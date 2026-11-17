"""MirrorBackend: one ``Backend`` implementation for every DAW.

Each call re-reads the DAW, dry-runs the tool on the simulator over that state (so errors are the
ones the model saw in training), and sends the resulting edits to the ``Driver`` as one atomic
apply. There is no read-back verification or rollback: a driver that fails raises ``DriverError``
and the call returns an ``execution_error``."""

from __future__ import annotations

from typing import Any

from openenv_daw.server.backends import SimBackend
from openenv_daw.server.sim import ops
from openenv_daw.server.sim.errors import DawError
from openenv_daw.server.sim.session import Session

from .changes import AddTrack, Change, RemoveTrack, diff
from .devices import DeviceMap
from .driver import SCRATCH_TRACK, Driver
from .ids import IdMap

READ_ONLY = {"get_session", "list_devices", "get_device_params"}


class MirrorBackend:
    def __init__(
        self,
        driver: Driver,
        device_map: DeviceMap | None = None,
        allow_destructive_load: bool = False,
    ) -> None:
        self.driver = driver
        self.device_map = device_map
        self.allow_destructive_load = allow_destructive_load
        self.ids = IdMap()

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
        sim = SimBackend(Session.restore(before.snapshot()))
        result = sim.call(tool, args)
        if not result.get("ok") or tool in READ_ONLY:
            return result
        changes = [c for c in diff(before, sim.session) if not _foreign_only(c)]
        if not changes:
            return result
        try:
            applied = self.driver.apply(changes, self.ids)
        except Exception as e:
            return {
                "ok": False,
                "error": {"code": "execution_error", "message": f"the DAW rejected the edit: {e}"},
            }
        if applied.warnings:
            result["warnings"] = list(applied.warnings)
        return result

    # --- loading ------------------------------------------------------------------------
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

    def close(self) -> None:
        close = getattr(self.driver, "close", None)
        if callable(close):
            close()


def _foreign_only(c: Change) -> bool:
    return isinstance(c, AddTrack) and c.kind not in ("midi", "audio", "return")
