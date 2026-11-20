"""Symbolic musicality heuristics: a soft score in [0, 1] for melodies, chord progressions and
drum patterns, computed from the final session. It is never a gate: ``score_episode`` blends it
into the reward only after every correctness check passed (see ``rubrics.MUSIC_WEIGHT``).

The bands are calibrated on ``tests/fixtures/reference_melodies.json`` (folk tunes): human
melodies score >= 0.7 and random in-key notes <= 0.4."""

from __future__ import annotations

from collections import Counter
from typing import Any

from openenv_daw.server.music import KICK, root_pc, scale_classes
from openenv_daw.server.sim.session import Session, Track

AbsNote = tuple[float, int, float, int]  # start, pitch, duration, velocity


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _trapezoid(x: float, zero_lo: float, one_lo: float, one_hi: float, zero_hi: float) -> float:
    """0 below ``zero_lo``, rising to 1 at ``one_lo``, flat to ``one_hi``, falling to 0 at ``zero_hi``."""
    if x <= zero_lo or x >= zero_hi:
        return 0.0
    if x < one_lo:
        return (x - zero_lo) / (one_lo - zero_lo)
    if x > one_hi:
        return (zero_hi - x) / (zero_hi - one_hi)
    return 1.0


def track_notes(t: Track | None) -> list[AbsNote]:
    if t is None:
        return []
    out = [
        (c.start_beat + n.start_beat, n.pitch, n.duration_beats, n.velocity)
        for c in t.clips
        for n in c.notes
    ]
    return sorted(out)


def _line(notes: list[AbsNote]) -> list[AbsNote]:
    """One note per onset (the top note), so chords do not count as leaps."""
    top: dict[float, AbsNote] = {}
    for n in notes:
        k = round(n[0], 3)
        if k not in top or n[1] > top[k][1]:
            top[k] = n
    return [top[k] for k in sorted(top)]


# --- melody ---------------------------------------------------------------------------------


def melody_subscores(notes: list[AbsNote], key: str) -> dict[str, float]:
    line = _line(notes)
    if len(line) < 3:
        return {}
    pitches = [n[1] for n in line]
    ivs = [b - a for a, b in zip(pitches, pitches[1:], strict=False)]
    step_ratio = sum(1 for i in ivs if 0 < abs(i) <= 2) / len(ivs)
    leaps = [k for k, i in enumerate(ivs) if abs(i) >= 5]
    recovered = sum(
        1 for k in leaps if k + 1 < len(ivs) and ivs[k + 1] * ivs[k] < 0 and abs(ivs[k + 1]) <= 2
    )
    # a leap on the very last interval has nothing to recover into; ignore it
    scored = [k for k in leaps if k + 1 < len(ivs)]
    leap_recovery = (recovered / len(scored)) if scored else 1.0

    tonic = root_pc(key.split()[0])
    mode = key.split()[1].lower()
    triad = {tonic, (tonic + (4 if mode == "major" else 3)) % 12, (tonic + 7) % 12}
    cadence = 0.4 * (pitches[0] % 12 in triad) + 0.6 * (pitches[-1] % 12 == tonic)
    if line[-1][2] >= 1.5 - 1e-6:  # a held final note sounds like an ending
        cadence = min(1.0, cadence + 0.0)
    else:
        cadence *= 0.85

    gaps = {round(b[0] - a[0], 3) for a, b in zip(line, line[1:], strict=False)}
    durs = {round(n[2], 3) for n in line}
    kinds = len(gaps | durs)
    rhythm = 0.0 if kinds <= 1 else 0.75 if kinds == 2 else 1.0
    if kinds > 7:
        rhythm = 0.6

    if len(ivs) >= 7:
        grams = [tuple(ivs[k : k + 3]) for k in range(len(ivs) - 2)]
        rgaps = [round(b[0] - a[0], 3) for a, b in zip(line, line[1:], strict=False)]
        rgrams = [tuple(rgaps[k : k + 3]) for k in range(len(rgaps) - 2)]
        pc, rc = Counter(grams), Counter(rgrams)
        rep_p = sum(1 for g in grams if pc[g] > 1) / len(grams)
        rep_r = sum(1 for g in rgrams if rc[g] > 1) / len(rgrams)
        motif = _clamp((0.6 * rep_p + 0.4 * rep_r) / 0.45)
    else:
        motif = None  # too short to repeat anything: not scored

    span = max(pitches) - min(pitches)
    range_score = _trapezoid(span, 2, 5, 19, 31)

    run = best = 1
    for a, b in zip(pitches, pitches[1:], strict=False):
        run = run + 1 if a == b else 1
        best = max(best, run)
    repeats = _clamp(1 - (best - 3) / 5) if best > 3 else 1.0

    step_score = _trapezoid(step_ratio, 0.15, 0.5, 0.85, 1.0)
    subs = {
        "stepwise": step_score,
        "leap_recovery": leap_recovery,
        "cadence": cadence,
        "rhythm": rhythm,
        "motif": motif,
        "range": range_score,
        "repeats": repeats,
    }
    return {k: v for k, v in subs.items() if v is not None}


MELODY_WEIGHTS = {
    "stepwise": 1.0,
    "leap_recovery": 1.0,
    "cadence": 2.0,
    "rhythm": 1.0,
    "motif": 1.0,
    "range": 1.0,
    "repeats": 1.0,
}


def melody_score(notes: list[AbsNote], key: str) -> tuple[float, dict[str, float]]:
    subs = melody_subscores(notes, key)
    if not subs:
        return 0.0, {}
    total = sum(MELODY_WEIGHTS[k] for k in subs)
    return sum(MELODY_WEIGHTS[k] * v for k, v in subs.items()) / total, subs


# --- chords ---------------------------------------------------------------------------------


def _chords(notes: list[AbsNote], bar: float) -> list[list[int]]:
    by_bar: dict[int, list[int]] = {}
    for start, pitch, _d, _v in notes:
        by_bar.setdefault(int(start // bar + 1e-6), []).append(pitch)
    return [sorted(set(by_bar[k])) for k in sorted(by_bar)]


def chords_score(notes: list[AbsNote], bar: float = 4.0) -> tuple[float, dict[str, float]]:
    chords = _chords(notes, bar)
    if len(chords) < 2 or any(len(c) < 3 for c in chords):
        return 0.0, {}
    moves, common, parallel, pairs = [], [], 0, 0
    for a, b in zip(chords, chords[1:], strict=False):
        n = min(len(a), len(b))
        moves.append(sum(abs(x - y) for x, y in zip(a[:n], b[:n], strict=True)) / n)
        common.append(len(set(a) & set(b)) / max(len(a), len(b)))
        lo_iv, hi_iv = (a[-1] - a[0]) % 12, (b[-1] - b[0]) % 12
        same_dir = (b[0] - a[0]) * (b[-1] - a[-1]) > 0
        if lo_iv in (0, 7) and hi_iv == lo_iv and same_dir:
            parallel += 1
        pairs += 1
    smooth = _clamp(1 - (sum(moves) / len(moves) - 2) / 5)
    held = _clamp(sum(common) / len(common) / 0.34)
    no_parallel = 1 - parallel / pairs
    subs = {"smoothness": smooth, "common_tones": held, "no_parallels": no_parallel}
    return 0.55 * smooth + 0.1 * held + 0.35 * no_parallel, subs


# --- drums ----------------------------------------------------------------------------------


def drums_score(notes: list[AbsNote], bar: float = 4.0) -> tuple[float, dict[str, float]]:
    if not notes:
        return 0.0, {}
    vels = {n[3] for n in notes}
    varied = 1.0 if len(vels) >= 2 else 0.0
    kicks = [n for n in notes if n[1] == KICK]
    down = [n[3] for n in kicks if abs(n[0] % bar) < 1e-6]
    other = [n[3] for n in kicks if abs(n[0] % bar) >= 1e-6]
    if down and other:
        diff = sum(down) / len(down) - sum(other) / len(other)
        accent = 1.0 if diff >= 5 else 0.3 if diff >= 0 else 0.0
    else:
        accent = 0.5
    hats = [n for n in notes if n[1] != KICK]
    if hats:
        on = [n[3] for n in hats if abs(n[0] % 1) < 1e-6]
        off = [n[3] for n in hats if abs(n[0] % 1) >= 1e-6]
        if on and off:
            diff = sum(on) / len(on) - sum(off) / len(off)
            hat = 1.0 if diff >= 8 else 0.3 if diff >= 0 else 0.0
        else:
            hat = 0.5
        subs = {"varied": varied, "downbeat_accent": accent, "hat_accent": hat}
        return 0.3 * varied + 0.35 * accent + 0.35 * hat, subs
    subs = {"varied": varied, "downbeat_accent": accent}
    return 0.4 * varied + 0.6 * accent, subs


# --- task level -----------------------------------------------------------------------------


def _track(s: Session, ref: str) -> Track | None:
    for t in s.tracks:
        if t.id == ref or t.name.lower() == str(ref).lower():
            return t
    return None


def score_spec(spec: Any, session: Session) -> tuple[float, dict[str, float]]:
    notes = track_notes(_track(session, spec.track))
    if spec.kind == "melody":
        return melody_score(notes, spec.key)
    if spec.kind == "chords":
        return chords_score(notes, spec.bar_beats)
    if spec.kind == "drums":
        return drums_score(notes, spec.bar_beats)
    raise ValueError(f"unknown music spec kind {spec.kind!r}")


def musicality(specs: list[Any], session: Session) -> tuple[float | None, dict[str, Any]]:
    """Mean score over the task's music specs (``None`` when the task has none) and per-spec
    sub-scores."""
    if not specs:
        return None, {}
    scores, detail = [], {}
    for sp in specs:
        s, subs = score_spec(sp, session)
        scores.append(s)
        detail[f"{sp.kind}:{sp.track}"] = {
            "score": round(s, 4),
            **{k: round(v, 3) for k, v in subs.items()},
        }
    return sum(scores) / len(scores), detail


__all__ = [
    "chords_score",
    "drums_score",
    "melody_score",
    "musicality",
    "scale_classes",
    "track_notes",
]
