"""Units and normalization shared by the simulator: beats, dB, pan, velocity."""

from __future__ import annotations

import os

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
