"""Task definition: an intent, an initial session, declarative checks and an oracle solution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Check(BaseModel):
    kind: str
    args: dict[str, Any] = Field(default_factory=dict)
    critical: bool = False


class OracleStep(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    task_id: str
    family: str
    intent: str
    split: Literal["train", "heldout"] = "train"
    initial_state: dict[str, Any] | None = None  # Session snapshot; None = empty project
    checks: list[Check] = Field(default_factory=list)
    # Pre-existing objects the task may change: "transport", "*", "track:<name>" (everything on
    # the track) or "track:<name>:<mixer|sends|clips|devices>".
    touchable: list[str] = Field(default_factory=list)
    expected_status: Literal["done", "blocked"] = "done"
    oracle: list[OracleStep] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
