"""DAW-agnostic session model (linear arrangement), inspired by the DAWproject 1.0 XSD."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --- units and normalization: beats, dB, pan, velocity ---
# Grid divisions per beat. 16 (default) keeps state hashes stable; 48 fits triplets and 1/16s.
GRID_DIVISIONS = int(os.getenv("OPENENV_DAW_GRID_DIVISIONS", "16"))
if GRID_DIVISIONS not in (16, 48):
    raise ValueError("OPENENV_DAW_GRID_DIVISIONS must be 16 or 48")
GRID = 1 / GRID_DIVISIONS  # beats; note and clip positions snap to this grid
DB_MIN = -70.0  # fader floor, treated as "-inf"
DB_MAX = 6.0
PAN_MIN, PAN_MAX = -1.0, 1.0
VEL_MIN, VEL_MAX = 1, 127
PITCH_MIN, PITCH_MAX = 0, 127
MAX_NOTES_PER_CALL = 512


def snap(beats: float) -> float:
    """Snap a beat position to the grid (exact in binary floating point for the 1/16 default)."""
    return round(beats / GRID) * GRID


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def r2(x: float) -> float:
    return round(x, 2)


TrackKind = Literal["midi", "audio", "return", "group"]  # group: read-only (foreign)


class Note(BaseModel):
    pitch: int
    start_beat: float  # clip-relative
    duration_beats: float
    velocity: int = 100


class Clip(BaseModel):
    id: str
    name: str
    start_beat: float  # position on the arrangement timeline
    length_beats: float
    notes: list[Note] = Field(default_factory=list)

    @property
    def end_beat(self) -> float:
        return self.start_beat + self.length_beats


class Device(BaseModel):
    id: str
    device: str  # catalog id
    name: str
    role: Literal["instrument", "audio_fx"]
    params: dict[str, float] = Field(default_factory=dict)  # normalized 0..1
    preset: str | None = None  # library id of the preset this device was loaded from


class ForeignParam(BaseModel):
    name: str
    display: str


class ForeignDevice(BaseModel):
    """A device the catalog does not know (third-party plugin, unmapped native device).
    Readable, never writable."""

    id: str
    name: str
    params: list[ForeignParam] = Field(default_factory=list)
    item: str | None = None  # library id when the agent loaded it (rack or plugin)
    role: Literal["instrument", "audio_fx"] | None = None  # known for library items only


class ForeignClip(BaseModel):
    """An audio or launcher clip the simulator cannot edit. Readable, never writable."""

    id: str
    name: str
    kind: Literal["audio", "launcher"]  # launcher: a clip in a clip-launcher / scene grid
    start_beat: float = 0.0
    length_beats: float = 0.0

    @field_validator("kind", mode="before")
    @classmethod
    def _legacy_kind(cls, v: Any) -> Any:
        return "launcher" if v == "session" else v  # pre-0.2 data


class Track(BaseModel):
    id: str
    name: str
    kind: TrackKind
    clips: list[Clip] = Field(default_factory=list)
    devices: list[Device] = Field(default_factory=list)
    volume_db: float = 0.0
    pan: float = 0.0
    mute: bool = False
    solo: bool = False
    sends: dict[str, float] = Field(default_factory=dict)  # return track id -> dB (absent = off)
    foreign_devices: list[ForeignDevice] = Field(default_factory=list)
    foreign_clips: list[ForeignClip] = Field(default_factory=list)


class Session(BaseModel):
    model_config = ConfigDict(validate_assignment=False)

    tempo_bpm: float = 120.0
    time_signature: tuple[int, int] = (4, 4)
    tracks: list[Track] = Field(default_factory=list)
    counters: dict[str, int] = Field(default_factory=lambda: {"trk": 0, "clp": 0, "dev": 0})

    def next_id(self, prefix: str) -> str:
        self.counters[prefix] = self.counters.get(prefix, 0) + 1
        return f"{prefix}_{self.counters[prefix]}"

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.model_dump(mode="json"))

    @classmethod
    def restore(cls, data: dict[str, Any]) -> Session:
        return cls.model_validate(copy.deepcopy(data))

    def state_hash(self) -> str:
        data = self.model_dump(mode="json")
        for t in data["tracks"]:  # empty foreign fields are excluded: pre-foreign hashes stay valid
            for k in ("foreign_devices", "foreign_clips"):
                if not t[k]:
                    del t[k]
            for d in t["devices"]:  # so are unset library fields (schema 1.1)
                if d["preset"] is None:
                    del d["preset"]
            for fd in t.get("foreign_devices", ()):
                for k in ("item", "role"):
                    if fd[k] is None:
                        del fd[k]
        blob = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()
