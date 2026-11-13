"""JSON state simulator backend: no audio, pure Python, deterministic."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from openenv_daw.schema import Session
from openenv_daw.sim import ops
from openenv_daw.sim.errors import SimError, unsupported
from openenv_daw.sim.profiles import DEFAULT_PROFILE, check_allowed

_OPS: dict[str, Callable[..., dict[str, Any]]] = {
    name: getattr(ops, name)
    for name in (
        "get_session", "set_transport", "create_track", "delete_track", "rename_track",
        "create_midi_clip", "delete_clip", "insert_midi_notes", "remove_midi_notes",
        "set_mixer_channel", "set_send", "list_devices", "insert_device", "get_device_params",
        "set_device_param",
    )
}


class SimBackend:
    capabilities: frozenset[str] = frozenset()

    def __init__(self, session: Session | None = None, profile: str = DEFAULT_PROFILE) -> None:
        self._session = session or Session()
        self.profile = profile

    @property
    def session(self) -> Session:
        return self._session

    def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            check_allowed(self.profile, tool)
            fn = _OPS.get(tool)
            if fn is None:
                raise unsupported(f"unknown tool {tool!r}")
            try:
                return fn(self._session, **args)
            except TypeError as e:  # bad/missing kwargs
                raise SimError("invalid_args", str(e)) from None
        except SimError as e:
            return e.to_result()

    def snapshot(self) -> dict[str, Any]:
        return self._session.snapshot()

    def restore(self, data: dict[str, Any]) -> None:
        self._session = Session.restore(data)

    def render(self, start_beat: float, length_beats: float) -> dict[str, Any]:
        return unsupported("audio rendering needs a headless backend").to_result()
