"""Backends: the simulator, and the interface a DAW adapter (``openenv_daw.adapters``) implements.

``resolve_backend`` finds a backend factory by entry-point name (group ``openenv_daw.backends``)
or by a ``module:factory`` path."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

from openenv_daw.server.sim import ops
from openenv_daw.server.sim.errors import DawError, unsupported
from openenv_daw.server.sim.session import Session

GROUP = "openenv_daw.backends"

TOOLS = (
    "get_session",
    "set_transport",
    "create_track",
    "delete_track",
    "rename_track",
    "create_midi_clip",
    "delete_clip",
    "update_midi_clip",
    "insert_midi_notes",
    "remove_midi_notes",
    "set_mixer_channel",
    "set_send",
    "list_devices",
    "insert_device",
    "delete_device",
    "get_device_params",
    "set_device_param",
)


@runtime_checkable
class Backend(Protocol):
    @property
    def session(self) -> Session: ...

    def load(self, session: Session) -> None:
        """Make the backend's project equal to ``session`` (episode start)."""

    def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run a tool; always returns a result dict (``ok`` False + ``error`` on failure)."""

    def snapshot(self) -> dict[str, Any]: ...

    def restore(self, data: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class SimBackend:
    """JSON state simulator: no audio, pure Python, deterministic."""

    def __init__(self, session: Session | None = None) -> None:
        self._session = session or Session()

    @property
    def session(self) -> Session:
        return self._session

    def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            if tool not in TOOLS:
                raise unsupported(f"unknown tool {tool!r}")
            args = {k: v for k, v in args.items() if not k.startswith("_")}
            try:
                return getattr(ops, tool)(self._session, **args)
            except TypeError as e:  # bad or missing arguments
                raise DawError("invalid_args", str(e)) from None
        except DawError as e:
            return e.to_result()

    def load(self, session: Session) -> None:
        self._session = Session.restore(session.snapshot())

    def snapshot(self) -> dict[str, Any]:
        return self._session.snapshot()

    def restore(self, data: dict[str, Any]) -> None:
        self._session = Session.restore(data)

    def close(self) -> None:
        return None


def available_backends() -> list[str]:
    return sorted(ep.name for ep in entry_points(group=GROUP))


def resolve_backend(name_or_path: str) -> Callable[[], Backend]:
    """A ``factory() -> Backend`` from an entry-point name or a ``module:factory`` path."""
    if ":" in name_or_path:
        module, _, attr = name_or_path.partition(":")
        return getattr(importlib.import_module(module), attr)
    eps = [ep for ep in entry_points(group=GROUP) if ep.name == name_or_path]
    if not eps:
        raise KeyError(f"unknown backend {name_or_path!r}; have {available_backends()}")
    return eps[0].load()
