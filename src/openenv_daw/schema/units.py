"""Units and normalization shared by the simulator: beats, dB, pan, velocity."""

from __future__ import annotations

GRID = 1 / 16  # beats; note and clip positions snap to this grid
DB_MIN = -70.0  # fader floor, treated as "-inf"
DB_MAX = 6.0
PAN_MIN, PAN_MAX = -1.0, 1.0
VEL_MIN, VEL_MAX = 1, 127
PITCH_MIN, PITCH_MAX = 0, 127
MAX_NOTES_PER_CALL = 512


def snap(beats: float) -> float:
    """Snap a beat position to the 1/16-beat grid (exact in binary floating point)."""
    return round(beats / GRID) * GRID


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def r2(x: float) -> float:
    return round(x, 2)
