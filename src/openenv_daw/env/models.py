"""Wire types. Actions are OpenEnv's MCP actions (CallToolAction / ListToolsAction)."""

from __future__ import annotations

from typing import Any

from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv.core.env_server.types import Observation, ResetRequest, State
from pydantic import Field

__all__ = [
    "CallToolAction",
    "CallToolObservation",
    "DawResetObservation",
    "DawResetRequest",
    "DawState",
]


class DawResetRequest(ResetRequest):
    """Task selection: ``task_id``, else ``split`` + ``index``, else a seeded sample of
    ``split`` (optionally restricted to a ``family``)."""

    task_id: str | None = None
    split: str = "train"
    family: str | None = None
    index: int | None = None
    profile: str | None = None


class DawResetObservation(Observation):
    """The agent sees the request only; it must call ``get_session`` to see the project."""

    task_id: str = ""
    intent: str = ""
    profile: str = "portable"


class DawState(State):
    task_id: str | None = None
    profile: str = "portable"
    consecutive_errors: int = 0
    error_count: int = 0
    status: str | None = None
    done: bool = False
    success: bool | None = None
    score: dict[str, Any] | None = None
    state_hash: str | None = None
    step_log: list[dict[str, Any]] = Field(default_factory=list)
