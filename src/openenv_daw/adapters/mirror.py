"""MirrorBackend: one ``Backend`` implementation for every DAW.

Each call re-reads the DAW, dry-runs the tool on the simulator over that state (so errors are the
ones the model saw in training), sends the resulting edits to the ``Driver`` as one atomic apply,
reads back and checks the touched objects, and rolls back on a mismatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openenv_daw.backends.sim import SimBackend
from openenv_daw.errors import DawError
from openenv_daw.schema import Session, Track
from openenv_daw.sim import ops
from openenv_daw.sim.profiles import capabilities_for, profile_for, tools_for

from .changes import (
    AddClip,
    AddDevice,
    AddForeignDevice,
    AddTrack,
    Change,
    RemoveClip,
    RemoveDevice,
    RemoveTrack,
    SetNotes,
    SetParam,
    SetSend,
    SetTransport,
    UpdateClip,
    UpdateTrack,
    diff,
)
from .devices import DeviceMap
from .driver import SCRATCH_TRACK, ApplyResult, Driver
from .ids import IdMap

READ_ONLY = {"get_session", "list_devices", "get_device_params"}


@dataclass
class Tolerances:
    db: float = 0.1
    norm: float = 0.01
    beats: float = 1e-3
    bpm: float = 0.01


class MirrorBackend:
    def __init__(
        self,
        driver: Driver,
        profile: str | None = None,
        device_map: DeviceMap | None = None,
        tolerances: Tolerances | None = None,
        allow_destructive_load: bool = False,
    ) -> None:
        profile = profile or profile_for(driver.capabilities)  # unset: what the driver can serve
        wanted = tools_for(profile) | capabilities_for(profile)
        if missing := wanted - set(driver.capabilities):
            raise ValueError(
                f"driver {driver.name!r} cannot serve profile {profile!r}: {sorted(missing)}"
            )
        self.driver = driver
        self.profile = profile
        self.capabilities = tools_for(profile) | capabilities_for(profile)
        self.device_map = device_map
        self.masked = device_map.masked() if device_map else frozenset()
        self.tol = tolerances or Tolerances()
        self.allow_destructive_load = allow_destructive_load
        self.ids = IdMap()
        self.broken = False

    # --- reading ------------------------------------------------------------------------
    def _read(self, notes: set[str] | frozenset[str] | str = frozenset()) -> Session:
        """Canonical mirror of the DAW. ``notes``: canonical clip ids to fetch, or ``"all"``."""
        raw = self.driver.read()
        raw.tracks = [t for t in raw.tracks if t.name != SCRATCH_TRACK]
        s = self.ids.canonicalize(raw)
        for t in s.tracks:
            for c in t.clips:
                if notes == "all" or c.id in notes:
                    got = self.driver.read_notes(self.ids.native(c.id))
                    c.notes = sorted(got, key=lambda n: (n.start_beat, n.pitch))
        return s

    @property
    def session(self) -> Session:
        return self._read("all")

    # --- Backend API --------------------------------------------------------------------
    def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if self.broken:
            return self._error("inconsistent_state", "the DAW is in an unknown state; stop here")
        try:
            return self._call(tool, args)
        except DawError as e:
            return e.to_result()

    def _touched_clips(
        self, tool: str, args: dict[str, Any], s: Session
    ) -> set[str] | frozenset[str] | str:
        if tool == "get_session" and args.get("include_notes"):
            if args.get("track"):
                try:
                    return {c.id for c in ops.find_track(s, str(args["track"])).clips}
                except DawError:
                    return frozenset()
            return "all"
        if tool in ("insert_midi_notes", "remove_midi_notes", "delete_clip") and args.get("clip"):
            try:
                return {ops.find_clip(s, str(args["clip"]))[1].id}
            except DawError:
                return frozenset()
        return frozenset()

    def _call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        probe = self._read()
        before = self._read(self._touched_clips(tool, args, probe))
        sim = SimBackend(Session.restore(before.snapshot()), self.profile, self.masked)
        result = sim.call(tool, args)
        if not result.get("ok") or tool in READ_ONLY:
            return result
        after = sim.session
        changes = [c for c in diff(before, after) if not _foreign_only(c)]
        if not changes:
            return result
        try:
            applied = self.driver.apply(changes, self.ids)
        except Exception as e:
            # apply is atomic: a raised error means nothing was applied. Check that, and only
            # roll back when the DAW state says otherwise.
            why = f"the DAW rejected the edit: {e}"
            now = self._read(_clips_with_notes(changes))
            if not self._verify(before, now, changes, ApplyResult()):
                return self._error("execution_error", why)
            return self._rollback(before, changes, why)
        read_back = self._read(_clips_with_notes(changes))
        problems = self._verify(after, read_back, changes, applied)
        if problems:
            return self._rollback(
                before, changes, "DAW state differs from the request: " + "; ".join(problems[:3])
            )
        if applied.warnings:
            result["warnings"] = list(applied.warnings)
        return result

    def _error(self, code: str, message: str) -> dict[str, Any]:
        return {"ok": False, "error": {"code": code, "message": message}}

    # --- rollback -----------------------------------------------------------------------
    def _rollback(self, before: Session, changes: list[Change], why: str) -> dict[str, Any]:
        try:
            if self.driver.native_undo:
                self.driver.undo_last()
            else:
                current = self._read("all")
                self.driver.apply(diff(current, before), self.ids)
            now = self._read(_clips_with_notes(changes))
            leftovers = self._verify(before, now, changes, ApplyResult())
            if leftovers:
                raise RuntimeError("; ".join(leftovers[:3]))
        except Exception as e:
            self.broken = True
            return self._error("inconsistent_state", f"{why}; rollback failed ({e}); stop here")
        return self._error("execution_error", why + "; the change was undone")

    # --- verification -------------------------------------------------------------------
    def _verify(
        self, want: Session, got: Session, changes: list[Change], applied: ApplyResult
    ) -> list[str]:
        tol, bad = self.tol, []

        def close(a: float, b: float, t: float) -> bool:
            return abs(a - b) <= t

        gt = {t.id: t for t in got.tracks}
        wt = {t.id: t for t in want.tracks}
        track_ids, mixer, sends, clips_of, devs_of = set(), set(), set(), set(), set()
        foreign_of: set[str] = set()
        for c in changes:
            match c:
                case AddTrack() | RemoveTrack():
                    track_ids.add(c.id)
                    mixer.add(c.id)
                case UpdateTrack():
                    track_ids.add(c.id)
                    mixer.add(c.id)
                case SetSend():
                    sends.add(c.track)
                case AddClip() | RemoveClip() | UpdateClip() | SetNotes():
                    clips_of.add(c.track)
                case AddDevice() | RemoveDevice() | SetParam():
                    devs_of.add(c.track)
                case AddForeignDevice():
                    foreign_of.add(c.track)
        if any(isinstance(c, SetTransport) for c in changes):
            if not close(want.tempo_bpm, got.tempo_bpm, tol.bpm) or tuple(
                want.time_signature
            ) != tuple(got.time_signature):
                bad.append("transport")
        removed = {tid for tid in track_ids if tid not in wt}
        for tid in track_ids:
            if tid in removed:
                if tid in gt:
                    bad.append(f"track {tid} still exists")
                continue
            if tid not in gt:
                bad.append(f"track {tid} missing")
        # order of regular tracks
        wo = [t.id for t in want.tracks if t.kind != "group"]
        go = [t.id for t in got.tracks if t.kind != "group"]
        if any(isinstance(c, (AddTrack, RemoveTrack)) for c in changes) and wo != go:
            bad.append(f"track order {go} != {wo}")
        for tid in mixer - removed:
            w, g = wt.get(tid), gt.get(tid)
            if w is None or g is None or tid in applied.adjusted:
                continue
            if (w.name, w.kind) != (g.name, g.kind):
                bad.append(f"track {tid} name/kind")
            if not close(w.volume_db, g.volume_db, tol.db) and not (
                w.volume_db <= -70 and g.volume_db <= -69
            ):
                bad.append(f"track {tid} volume {g.volume_db} != {w.volume_db}")
            if not close(w.pan, g.pan, tol.norm) or w.mute != g.mute or w.solo != g.solo:
                bad.append(f"track {tid} pan/mute/solo")
        for tid in sends:
            w, g = wt.get(tid), gt.get(tid)
            if w is None or g is None or tid in applied.adjusted:
                continue
            for rid in set(w.sends) | set(g.sends):
                if rid in applied.adjusted or rid not in wt:
                    continue
                a, b = w.sends.get(rid, -70.0), g.sends.get(rid, -70.0)
                if not close(a, b, tol.db) and not (a <= -70 and b <= -69):
                    bad.append(f"send {tid}->{rid} {b} != {a}")
        for tid in clips_of:
            w, g = wt.get(tid), gt.get(tid)
            if w is None or g is None:
                continue
            bad += self._cmp_clips(w, g, applied)
        for tid in devs_of:
            w, g = wt.get(tid), gt.get(tid)
            if w is None or g is None:
                continue
            bad += self._cmp_devices(w, g, applied)
        for tid in foreign_of:
            w, g = wt.get(tid), gt.get(tid)
            if w is None or g is None:
                continue
            want_items = {(d.id, d.item) for d in w.foreign_devices if d.item}
            got_items = {(d.id, d.item) for d in g.foreign_devices if d.item}
            if want_items - got_items:
                bad.append(f"foreign devices of {tid}: {sorted(want_items - got_items)} missing")
        return bad

    def _cmp_clips(self, w: Track, g: Track, applied: ApplyResult) -> list[str]:
        tol, bad = self.tol, []
        if [c.id for c in w.clips] != [c.id for c in g.clips]:
            return [f"clips of {w.id}: {[c.id for c in g.clips]} != {[c.id for c in w.clips]}"]
        for a, b in zip(w.clips, g.clips, strict=True):
            if a.id in applied.adjusted:
                continue
            if (
                a.name != b.name
                or abs(a.start_beat - b.start_beat) > tol.beats
                or abs(a.length_beats - b.length_beats) > tol.beats
            ):
                bad.append(f"clip {a.id} placement/name")
            if b.notes or a.notes:
                an = sorted(
                    (n.pitch, round(n.start_beat, 3), round(n.duration_beats, 3), n.velocity)
                    for n in a.notes
                )
                bn = sorted(
                    (n.pitch, round(n.start_beat, 3), round(n.duration_beats, 3), n.velocity)
                    for n in b.notes
                )
                if len(an) != len(bn) or any(
                    x[0] != y[0]
                    or x[3] != y[3]
                    or abs(x[1] - y[1]) > tol.beats
                    or abs(x[2] - y[2]) > tol.beats
                    for x, y in zip(an, bn, strict=True)
                ):
                    bad.append(f"clip {a.id} notes")
        return bad

    def _cmp_devices(self, w: Track, g: Track, applied: ApplyResult) -> list[str]:
        bad = []
        if [(d.id, d.device) for d in w.devices] != [(d.id, d.device) for d in g.devices]:
            return [f"devices of {w.id} differ"]
        for a, b in zip(w.devices, g.devices, strict=True):
            if a.preset != b.preset and a.id not in applied.acknowledged:
                bad.append(f"device {a.id} preset {b.preset} != {a.preset}")
            if a.id in applied.adjusted:
                continue
            for k, v in a.params.items():
                if f"{a.device}.{k}" in self.masked:
                    continue
                if abs(v - b.params.get(k, v)) > self.tol.norm:
                    bad.append(f"device {a.id}.{k} {b.params.get(k)} != {v}")
        return bad

    # --- loading, snapshots -------------------------------------------------------------
    def load(self, session: Session) -> None:
        raw = self.driver.read()
        if not self.allow_destructive_load or not any(t.name == SCRATCH_TRACK for t in raw.tracks):
            raise DawError(
                "unsupported",
                f"refusing to overwrite the project: needs allow_destructive_load=True and a "
                f"track named {SCRATCH_TRACK!r}",
            )
        current = self._read("all")
        removal: list[Change] = [RemoveTrack(t.id) for t in current.tracks if t.kind != "group"]
        if removal:
            self.driver.apply(removal, self.ids)
        for t in current.tracks:
            if t.kind != "group":
                self.ids._c2n.pop(t.id, None)
        self.ids._n2c = {k: v for k, v in self.ids._n2c.items() if v in self.ids._c2n}
        for k, v in session.counters.items():
            self.ids.counters[k] = max(self.ids.counters.get(k, 0), v)
        base = self._read("all")
        target = Session.restore(session.snapshot())
        changes = [c for c in diff(base, target) if not _foreign_only(c)]
        if changes:
            self.driver.apply(changes, self.ids)

    def snapshot(self) -> dict[str, Any]:
        return self.session.snapshot()

    def restore(self, data: dict[str, Any]) -> None:
        raise DawError("unsupported", "restore is not supported by DAW adapters")

    def render(self, start_beat: float, length_beats: float) -> dict[str, Any]:
        return DawError("unsupported", "audio rendering is not available").to_result()

    def close(self) -> None:
        close = getattr(self.driver, "close", None)
        if callable(close):
            close()


def _foreign_only(c: Change) -> bool:
    return isinstance(c, AddTrack) and c.kind not in ("midi", "audio", "return")


def _clips_with_notes(changes: list[Change]) -> set[str]:
    return {c.clip for c in changes if isinstance(c, SetNotes)} | {
        c.id for c in changes if isinstance(c, AddClip)
    }
