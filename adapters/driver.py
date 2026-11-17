"""The interface a DAW adapter implements. Everything else (tool rules, errors, id stability,
diffing) is done once by ``MirrorBackend``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from openenv_daw.server.sim.session import Note, Session

from .changes import Change
from .ids import IdMap

SCRATCH_TRACK = "__openenv_scratch__"  # marker track that allows destructive load


class DriverError(Exception):
    """The DAW refused or failed an edit."""


@dataclass
class ApplyResult:
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class Driver(Protocol):
    name: str

    def read(self) -> Session:
        """The DAW project without MIDI notes. Object ids are the DAW's own stable keys (strings
        unique per object kind). Unmapped plugins, audio clips and group tracks are reported as
        foreign objects. The scratch marker track may be included; the mirror hides it."""

    def read_notes(self, clip: str) -> list[Note]:
        """Notes of the clip with native key ``clip``, clip-relative."""

    def apply(self, changes: list[Change], ids: IdMap) -> ApplyResult:
        """Carry out the changes atomically (all or nothing, one undo step). Changes carry
        canonical ids: resolve existing objects with ``ids.native`` and register every object you
        create with ``ids.bind(canonical_id, native_key)``. Raise ``DriverError`` on failure."""
