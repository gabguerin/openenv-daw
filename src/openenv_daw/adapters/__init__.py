"""Adapter SDK (provisional API): implement a ``Driver`` for a DAW and wrap it in a
``MirrorBackend`` to get a full ``Backend`` that behaves exactly like the training simulator."""
from .changes import (
    AddClip,
    AddDevice,
    AddTrack,
    Change,
    NoteSpec,
    RemoveClip,
    RemoveDevice,
    RemoveTrack,
    SetNotes,
    SetParam,
    SetSend,
    SetTransport,
    UpdateClip,
    UpdateTrack,
    apply_changes,
    diff,
    same_session,
)
from .devices import DeviceMap
from .driver import SCRATCH_TRACK, ApplyResult, Driver, DriverError
from .ids import IdMap
from .mirror import MirrorBackend, Tolerances

__all__ = [
    "SCRATCH_TRACK",
    "AddClip",
    "AddDevice",
    "AddTrack",
    "ApplyResult",
    "Change",
    "DeviceMap",
    "Driver",
    "DriverError",
    "IdMap",
    "MirrorBackend",
    "NoteSpec",
    "RemoveClip",
    "RemoveDevice",
    "RemoveTrack",
    "SetNotes",
    "SetParam",
    "SetSend",
    "SetTransport",
    "Tolerances",
    "UpdateClip",
    "UpdateTrack",
    "apply_changes",
    "diff",
    "same_session",
]
