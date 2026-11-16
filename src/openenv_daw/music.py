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


NUMERALS = {
    "i": 0,
    "ii": 1,
    "iii": 2,
    "iv": 3,
    "v": 4,
    "vi": 5,
    "vii": 6,
}  # roman numeral -> scale degree


def roman_chord(key: str, numeral: str) -> list[int]:
    """Pitch classes (root, third, fifth) of a roman-numeral triad in ``key``. The degree comes
    from the numeral, the quality from its case: upper = major, lower = minor. Minor keys use the
    natural minor scale, so ``V`` is the (major) dominant on the fifth degree."""
    root, mode = key.split()
    steps = MAJOR if mode.lower() == "major" else MINOR
    deg = NUMERALS[numeral.lower()]
    r = (root_pc(root) + steps[deg]) % 12
    third = 4 if numeral.isupper() else 3
    return [r, (r + third) % 12, (r + 7) % 12]


def voice_lead(prev: list[int] | None, pcs: list[int], lo: int = 48, hi: int = 72) -> list[int]:
    """Voicing of the triad ``pcs`` between ``lo`` and ``hi`` that moves least from ``prev``
    (MIDI pitches); the first chord is the root-position triad at or above ``lo``."""
    from itertools import product

    options = [[p for p in range(lo, hi + 1) if p % 12 == pc] for pc in pcs]
    if prev is None:
        root = min(options[0])
        out = [root]
        for pc in pcs[1:]:
            out.append(min(p for p in range(out[-1] + 1, out[-1] + 13) if p % 12 == pc))
        return out

    def parallel(c: list[int]) -> bool:
        """Outer voices move together a perfect fifth or octave apart in both chords."""
        lo_iv, hi_iv = (prev[-1] - prev[0]) % 12, (c[-1] - c[0]) % 12
        return lo_iv in (0, 7) and hi_iv == lo_iv and (c[0] - prev[0]) * (c[-1] - prev[-1]) > 0

    cands = [sorted(c) for c in product(*options)]
    best = min(
        cands,
        key=lambda c: (
            parallel(c),
            sum(abs(a - b) for a, b in zip(c, sorted(prev), strict=True)),
            -len(set(c) & set(prev)),
            sum(c),
        ),
    )
    return best
