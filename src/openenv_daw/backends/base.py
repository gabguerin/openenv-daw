"""Backend interface (version 1). The simulator is the reference implementation; a DAW adapter
(see ``openenv_daw.adapters``) or a headless audio engine implements the same interface."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from openenv_daw.errors import DawError
from openenv_daw.schema import Session
from openenv_daw.sim.profiles import capabilities_for, check_allowed, tools_for

BACKEND_API_VERSION = 1


@runtime_checkable
class Backend(Protocol):
    profile: str
    capabilities: frozenset[str]  # tools and capability tags this backend actually supports

    @property
    def session(self) -> Session: ...

    def load(self, session: Session) -> None:
        """Make the backend's project equal to ``session`` (episode start)."""

    def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run a tool; always returns a result dict (``ok`` False + ``error`` on failure)."""

    def snapshot(self) -> dict[str, Any]: ...

    def restore(self, data: dict[str, Any]) -> None: ...

    def render(self, start_beat: float, length_beats: float) -> dict[str, Any]: ...

    def close(self) -> None: ...


class ProfileGuard:
    """Wraps any backend so that only the tools of ``profile`` can be called, and checks at
    construction that the backend supports the whole profile."""

    def __init__(self, backend: Backend, profile: str) -> None:
        missing = (tools_for(profile) | capabilities_for(profile)) - set(backend.capabilities)
        if missing:
            raise ValueError(
                f"backend does not support profile {profile!r}: missing {sorted(missing)}"
            )
        self.backend = backend
        self.profile = profile
        self.capabilities = backend.capabilities

    @property
    def session(self) -> Session:
        return self.backend.session

    def load(self, session: Session) -> None:
        self.backend.load(session)

    def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            check_allowed(self.profile, tool)
        except DawError as e:
            return e.to_result()
        return self.backend.call(tool, args)

    def snapshot(self) -> dict[str, Any]:
        return self.backend.snapshot()

    def restore(self, data: dict[str, Any]) -> None:
        self.backend.restore(data)

    def render(self, start_beat: float, length_beats: float) -> dict[str, Any]:
        return self.backend.render(start_beat, length_beats)

    def close(self) -> None:
        self.backend.close()


BackendFactory = Any  # Callable[[str], Backend]: receives the profile name
