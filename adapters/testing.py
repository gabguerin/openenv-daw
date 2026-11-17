"""Conformance kit for DAW drivers: an in-memory ``FakeDriver`` that behaves like a real DAW
(opaque index-addressed ids, optional lossy units, fault injection) and
``assert_driver_conformance`` that replays task oracles through ``MirrorBackend``."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from openenv_daw.server.backends import SimBackend
from openenv_daw.server.episode import Episode
from openenv_daw.server.sim.catalog import load_catalog, load_library
from openenv_daw.server.sim.session import Note, Session, Track
from openenv_daw.server.tasks.model import Task

from .changes import (
    AddClip,
    AddDevice,
    AddForeignDevice,
    AddTrack,
    Change,
    NoteSpec,
    RemoveClip,
    RemoveDevice,
    RemoveTrack,
    SetNotes,
    SetParam,
    SetSend,
    UpdateClip,
    UpdateTrack,
    apply_changes,
    same_session,
)
from .devices import DeviceMap
from .driver import SCRATCH_TRACK, ApplyResult, DriverError
from .ids import IdMap
from .mirror import MirrorBackend


class FakeDriver:
    name = "fake"

    def __init__(self, lossy: bool = False, scratch: bool = True) -> None:
        self.lossy = lossy
        self._s = Session()
        self._n = 0
        # the marker track is reported by read() but is not part of the editable track list
        self._scratch = (
            Track(id=self._key("trk"), name=SCRATCH_TRACK, kind="midi") if scratch else None
        )
        self.apply_calls = 0
        # fault injection
        self.fail_next_apply = False

    def _key(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix[0]}:{self._n}"  # opaque, index-addressed, differs from canonical ids

    # --- Driver ---------------------------------------------------------------------------
    def read(self) -> Session:
        s = Session.restore(self._s.snapshot())
        for t in s.tracks:
            for c in t.clips:
                c.notes = []
        if self._scratch is not None:
            s.tracks.append(self._scratch.model_copy(deep=True))
        return s

    def read_notes(self, clip: str) -> list[Note]:
        for t in self._s.tracks:
            for c in t.clips:
                if c.id == clip:
                    return [n.model_copy() for n in c.notes]
        raise DriverError(f"no clip {clip}")

    def apply(self, changes: list[Change], ids: IdMap) -> ApplyResult:
        self.apply_calls += 1
        if self.fail_next_apply:
            self.fail_next_apply = False
            raise DriverError("simulated DAW failure")
        native = [self._native(c, ids) for c in changes]
        self._s = apply_changes(self._s, native)
        return ApplyResult()

    # --- helpers ----------------------------------------------------------------------------
    def _q(self, v: float | None, step: float) -> float | None:
        return v if v is None or not self.lossy else round(round(v / step) * step, 6)

    def _native(self, ch: Change, ids: IdMap) -> Change:
        n = ids.native
        match ch:
            case AddTrack():
                key = self._key("trk")
                ids.bind(ch.id, key)
                return dataclasses.replace(ch, id=key)
            case RemoveTrack():
                return RemoveTrack(n(ch.id))
            case UpdateTrack():
                return dataclasses.replace(
                    ch,
                    id=n(ch.id),
                    volume_db=self._q(ch.volume_db, 0.01),
                    pan=self._q(ch.pan, 0.001),
                )
            case SetSend():
                return SetSend(n(ch.track), n(ch.return_track), self._q(ch.level_db, 0.01))
            case AddClip():
                key = self._key("clp")
                ids.bind(ch.id, key)
                return dataclasses.replace(
                    ch,
                    id=key,
                    track=n(ch.track),
                    start_beat=self._q(ch.start_beat, 1 / 64) or 0.0,
                    length_beats=self._q(ch.length_beats, 1 / 64) or 0.0,
                )
            case RemoveClip():
                return RemoveClip(n(ch.id), n(ch.track))
            case UpdateClip():
                return dataclasses.replace(ch, id=n(ch.id), track=n(ch.track))
            case SetNotes():
                q = lambda ns: tuple(  # noqa: E731
                    NoteSpec(
                        x.pitch,
                        self._q(x.start_beat, 1 / 64) or 0.0,
                        self._q(x.duration_beats, 1 / 64) or 0.0,
                        x.velocity,
                    )
                    for x in ns
                )
                return SetNotes(n(ch.clip), n(ch.track), q(ch.added), q(ch.removed), q(ch.final))
            case AddDevice():
                key = self._key("dev")
                ids.bind(ch.id, key)
                return dataclasses.replace(
                    ch,
                    id=key,
                    track=n(ch.track),
                    params={k: self._q(v, 0.001) or 0.0 for k, v in ch.params.items()},
                )
            case AddForeignDevice():
                key = self._key("fdv")
                ids.bind(ch.id, key)
                return dataclasses.replace(ch, id=key, track=n(ch.track))
            case RemoveDevice():
                return RemoveDevice(n(ch.id), n(ch.track))
            case SetParam():
                return dataclasses.replace(
                    ch,
                    device=n(ch.device),
                    track=n(ch.track),
                    value=self._q(ch.value, 0.001) or 0.0,
                )
        return ch

    def seed_foreign(self, session: Session, ids: IdMap) -> None:
        """Test hook: put the session's read-only group tracks (foreign content) into the DAW."""
        for i, t in enumerate(session.tracks):
            if t.kind != "group":
                continue
            native = Session.restore({"tracks": [t.model_dump(mode="json")]}).tracks[0]
            native.id = self._key("trk")
            ids.bind(t.id, native.id)
            for c in native.foreign_clips:
                key = self._key("fcl")
                ids.bind(c.id, key)
                c.id = key
            for d in native.foreign_devices:
                key = self._key("fdv")
                ids.bind(d.id, key)
                d.id = key
            self._s.tracks.insert(min(i, len(self._s.tracks)), native)


def _servable(task: Task, device_map: DeviceMap | None) -> bool:
    """Whether the device map covers every device, preset and parameter the oracle uses."""
    if device_map is None:
        return True
    lib, cat = load_library(), load_catalog()
    for st in task.oracle:
        dev = st.args.get("device")
        if st.tool == "insert_device" and dev in lib and dev not in device_map.library:
            return False
        if st.tool == "insert_device" and dev in cat and dev not in device_map.devices:
            return False
        if st.tool == "set_device_param":
            track = next(
                (
                    x
                    for x in task.oracle
                    if x.tool == "insert_device" and x.args.get("track") == st.args.get("track")
                ),
                None,
            )
            base = track.args.get("device") if track else dev
            base = lib[base].base if base in lib else base
            if (
                base in cat
                and device_map.param(base, str(st.args.get("param", "")).lower()) is None
            ):
                return False
    return True


def assert_driver_conformance(
    make_driver: Callable[[], Any],
    tasks: list[Task],
    *,
    device_map: DeviceMap | None = None,
    lossless: bool = True,
) -> dict[str, Any]:
    """Replay every task's oracle through ``MirrorBackend(make_driver())``: each must score 1.0
    and, for lossless drivers, end in the same state as the simulator. Tasks the device map does
    not cover are skipped. Returns a report with the device-map coverage."""
    failures: list[str] = []
    served = [t for t in tasks if _servable(t, device_map)]
    for task in served:
        driver = make_driver()
        mirror = MirrorBackend(driver, device_map, allow_destructive_load=True)
        initial = Session.restore(task.initial_state) if task.initial_state else Session()
        if hasattr(driver, "seed_foreign"):
            driver.seed_foreign(initial, mirror.ids)
        ep = Episode(task, 60, 5, backend_factory=lambda m=mirror: m)  # type: ignore[misc]
        ref = SimBackend(Session.restore(initial.snapshot()))
        for step in task.oracle:
            ep.run(step.tool, step.args)
            if step.tool != "submit":
                ref.call(step.tool, step.args)
        if ep.score is None or ep.score.reward != 1.0:
            failures.append(
                f"{task.task_id}: reward {ep.score and ep.score.reward} {ep.score and ep.score.failed_checks}"
            )
        elif lossless and not same_session(mirror.session, ref.session):
            failures.append(f"{task.task_id}: final state differs from the simulator")
        mirror.close()
    assert not failures, f"{len(failures)}/{len(served)} tasks failed:\n" + "\n".join(failures[:10])
    return {"tasks": len(served), "coverage": device_map.coverage() if device_map else None}
