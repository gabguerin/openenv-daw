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


class MusicSpec(BaseModel):
    """What to score for musicality on a track: ``melody`` (needs ``key``), ``chords`` or ``drums``."""

    kind: Literal["melody", "chords", "drums"]
    track: str
    key: str | None = None
    bar_beats: float = 4.0


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
    # Names of tracks the task may create, and "device:<catalog id>" entries for devices it needs.
    # Anything else new is extraneous (see rewards.damage.extraneous_violations).
    allowed_new: list[str] = Field(default_factory=list)
    expected_status: Literal["done", "blocked"] = "done"
    oracle: list[OracleStep] = Field(default_factory=list)
    # Soft musicality scoring (never a gate); empty for tasks that are not about music.
    music: list[MusicSpec] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
