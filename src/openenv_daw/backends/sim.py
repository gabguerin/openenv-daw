"""JSON state simulator backend: no audio, pure Python, deterministic."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from openenv_daw.schema import Session
from openenv_daw.sim import ops
from openenv_daw.sim.catalog import unavailable_items
from openenv_daw.sim.errors import SimError, unsupported
from openenv_daw.sim.profiles import (
    DEFAULT_PROFILE,
    capabilities_for,
    check_allowed,
    tools_for,
    unsupported_params_for,
)

_OPS: dict[str, Callable[..., dict[str, Any]]] = {
    name: getattr(ops, name)
    for name in (
        "get_session",
        "set_transport",
        "create_track",
        "delete_track",
        "rename_track",
        "create_midi_clip",
        "delete_clip",
        "insert_midi_notes",
        "remove_midi_notes",
        "set_mixer_channel",
        "set_send",
        "list_devices",
        "insert_device",
        "get_device_params",
        "set_device_param",
    )
}


class SimBackend:
    def __init__(
        self,
        session: Session | None = None,
        profile: str = DEFAULT_PROFILE,
        masked_params: frozenset[str] = frozenset(),
    ) -> None:
        # ``<device>.<param>`` entries this backend cannot set (profile gaps plus episode mask)
        self.masked = unsupported_params_for(profile) | frozenset(masked_params)
        self._session = session or Session()
        self.profile = profile
        self.capabilities = tools_for(profile) | capabilities_for(profile)

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
                if self.masked and tool == "set_device_param":
                    self._reject_masked(args)
                args = {k: v for k, v in args.items() if not k.startswith("_")}
                if tool in ("list_devices", "insert_device"):
                    args["_unavailable"] = unavailable_items(self.capabilities, self.masked)
                result = fn(self._session, **args)
            except TypeError as e:  # bad/missing kwargs
                raise SimError("invalid_args", str(e)) from None
            return self._annotate(tool, result) if self.masked else result
        except SimError as e:
            return e.to_result()

    def load(self, session: Session) -> None:
        self._session = Session.restore(session.snapshot())

    def close(self) -> None:
        return None

    def _catalog_device(self, track: str, device: str) -> str | None:
        try:
            return ops.find_device(ops.find_track(self._session, track), device).device
        except SimError:
            return None

    def _reject_masked(self, args: dict[str, Any]) -> None:
        dev = self._catalog_device(str(args.get("track")), str(args.get("device")))
        param = str(args.get("param", "")).lower()
        if dev and f"{dev}.{param}" in self.masked:
            raise unsupported(f"parameter {param!r} of {dev!r} is not supported by this DAW")

    def _annotate(self, tool: str, result: dict[str, Any]) -> dict[str, Any]:
        """Mark masked parameters: ``supported: false`` (only ever added, never ``true``)."""
        if not result.get("ok"):
            return result
        if tool == "get_device_params" and not result.get("foreign"):
            dev = self._catalog_device(result["track"], result["device"])
            for p in result["params"]:
                if f"{dev}.{p['name']}" in self.masked:
                    p["supported"] = False
        elif tool == "insert_device" or tool == "set_device_param":
            return result
        elif tool == "list_devices":
            for d in result["devices"]:
                dev = d.get("base") or d["device"]
                gaps = [p for p in d.get("params", ()) if f"{dev}.{p}" in self.masked]
                if gaps:
                    d["unsupported_params"] = gaps
        return result

    def snapshot(self) -> dict[str, Any]:
        return self._session.snapshot()

    def restore(self, data: dict[str, Any]) -> None:
        self._session = Session.restore(data)

    def render(self, start_beat: float, length_beats: float) -> dict[str, Any]:
        return unsupported("audio rendering needs a headless backend").to_result()
