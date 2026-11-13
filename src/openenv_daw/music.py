"""Music-theory helpers (ported and trimmed from ableton-sidekick ``core/music.py``)."""

from __future__ import annotations

ROOTS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
MAJOR = (0, 2, 4, 5, 7, 9, 11)
MINOR = (0, 2, 3, 5, 7, 8, 10)
PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# General MIDI drum pitches used by the drum_kit device.
KICK, SNARE, CLAP, CLOSED_HAT, OPEN_HAT = 36, 38, 39, 42, 46


def root_pc(root: str) -> int:
    """Pitch class of a root such as ``F``, ``F#`` or ``Bb``."""
    pc = ROOTS[root[0].upper()]
    for acc in root[1:]:
        pc += 1 if acc in "#♯" else -1 if acc in "b♭" else 0
    return pc % 12


def scale_classes(key: str) -> list[int]:
    """Pitch classes of ``"F# minor"`` style keys."""
    root, mode = key.split()
    steps = MAJOR if mode.lower() == "major" else MINOR
    r = root_pc(root)
    return sorted((r + s) % 12 for s in steps)


def scale_pitches(key: str, lo: int, hi: int) -> list[int]:
    ok = set(scale_classes(key))
    return [p for p in range(lo, hi + 1) if p % 12 in ok]


def triad(root: int, steps: tuple[int, ...], degree: int, octave: int = 3) -> list[int]:
    """Diatonic triad on scale ``degree`` (0-based) as MIDI pitches; octave 3 puts C at 48."""
    base = 12 * (octave + 1)
    return [base + root + steps[(degree + k) % 7] + 12 * ((degree + k) // 7) for k in (0, 2, 4)]


def note_name(pitch: int) -> str:
    return f"{PC_NAMES[pitch % 12]}{pitch // 12 - 1}"
