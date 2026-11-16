"""The interface a DAW adapter implements. Everything else (tool rules, errors, id stability,
verification, rollback) is done once by ``MirrorBackend``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from openenv_daw.schema import Note, Session

from .changes import Change
from .ids import IdMap

SCRATCH_TRACK = "__openenv_scratch__"  # marker track that allows destructive load


class DriverError(Exception):
    """The DAW refused or failed an edit."""


@dataclass
class ApplyResult:
    warnings: list[str] = field(default_factory=list)
    # canonical ids whose DAW value legitimately differs from the request (clamping, rounding
    # beyond tolerance); the mirror adopts the DAW value instead of treating it as drift
    adjusted: set[str] = field(default_factory=set)
    # canonical ids of loaded library items (presets) that the DAW confirmed it loaded: the mirror
    # then checks only the base device on read-back, not the preset's reported name
    acknowledged: set[str] = field(default_factory=set)


@runtime_checkable
class Driver(Protocol):
    name: str
    capabilities: frozenset[str]  # tools and capability tags the driver can carry out
    native_undo: bool  # True when ``undo_last`` reverts exactly one ``apply``

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

    def undo_last(self) -> None:
        """Revert the last ``apply`` (only called when ``native_undo``)."""
