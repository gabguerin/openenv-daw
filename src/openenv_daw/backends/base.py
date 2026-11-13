"""Backend interface. The simulator is the only implementation for now; a headless audio engine
or a DAW adapter would implement the same interface (``render`` is the reserved extension)."""

from __future__ import annotations

from typing import Any, Protocol

from openenv_daw.schema import Session


class Backend(Protocol):
    profile: str
    capabilities: frozenset[str]  # e.g. {"render"} once an audio backend exists

    @property
    def session(self) -> Session: ...

    def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run a tool; always returns a result dict (``ok`` False + ``error`` on failure)."""

    def snapshot(self) -> dict[str, Any]: ...

    def restore(self, data: dict[str, Any]) -> None: ...

    def render(self, start_beat: float, length_beats: float) -> dict[str, Any]: ...
