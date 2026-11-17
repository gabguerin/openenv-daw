"""Declarative state checks over a Session (ported in spirit from ableton-sidekick checkers,
re-keyed from scene slots to timeline clips). Tracks are referenced by name or id."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from openenv_daw.server.music import roman_chord, scale_classes
from openenv_daw.server.sim.catalog import load_library
from openenv_daw.server.sim.session import Session, Track

CheckFn = Callable[[Session, Session, dict[str, Any]], bool]
CHECKS: dict[str, CheckFn] = {}


def check(fn: CheckFn) -> CheckFn:
    """Register a built-in check under its function name."""
    CHECKS[fn.__name__] = fn
    return fn


def register_check(name: str) -> Callable[[CheckFn], CheckFn]:
    """Register a third-party check. ``name`` must be namespaced (``"pkg.name"``) and unique."""

    def deco(fn: CheckFn) -> CheckFn:
        if "." not in name:
            raise ValueError(f"check name {name!r} must be namespaced, e.g. 'mypkg.{name}'")
        if name in CHECKS:
            raise ValueError(f"check {name!r} is already registered")
        CHECKS[name] = fn
        return fn

    return deco


def _track(s: Session, ref: str) -> Track | None:
    for t in s.tracks:
        if t.id == ref or t.name.lower() == str(ref).lower():
            return t
    return None


def _notes(t: Track) -> list[Any]:
    return [n for c in t.clips for n in c.notes]


def _scoped(t: Track | None, a: dict[str, Any]) -> list[Any]:
    """Notes of the clip starting at ``a["start_beat"]`` when given, else of the whole track."""
    if t is None:
        return []
    if "start_beat" not in a:
        return _notes(t)
    return [n for c in t.clips if abs(c.start_beat - a["start_beat"]) < 1e-6 for n in c.notes]


def _abs_notes(t: Track) -> list[tuple[float, Any]]:
    return [(c.start_beat + n.start_beat, n) for c in t.clips for n in c.notes]


@check
def tempo_is(s, i, a):
    return abs(s.tempo_bpm - a["bpm"]) <= a.get("tol", 0.5)


@check
def time_signature_is(s, i, a):
    return tuple(s.time_signature) == (a["numerator"], a["denominator"])


@check
def track_exists(s, i, a):
    t = _track(s, a["track"])
    return t is not None and (a.get("kind") is None or t.kind == a["kind"])


@check
def track_absent(s, i, a):
    return _track(s, a["track"]) is None


@check
def track_count(s, i, a):
    kind = a.get("kind")
    return sum(1 for t in s.tracks if kind is None or t.kind == kind) == a["n"]


@check
def has_device(s, i, a):
    """The track holds a device matching every given argument: ``device`` (catalog id),
    ``preset`` or ``item`` (library id of a preset, or of a rack/plugin) and ``tags_any`` (the
    loaded library item carries at least one of these tags)."""
    t = _track(s, a["track"])
    if t is None:
        return False
    lib = load_library()
    tags = set(a.get("tags_any", ()))
    loaded = [(d.device, d.preset, None) for d in t.devices] + [
        (None, None, d.item) for d in t.foreign_devices
    ]

    def matches(device: str | None, preset: str | None, item: str | None) -> bool:
        if "device" in a and device != a["device"]:
            return False
        if "preset" in a and preset != a["preset"]:
            return False
        if "item" in a and item != a["item"]:
            return False
        if tags:
            lib_id = preset or item
            return lib_id in lib and bool(tags & set(lib[lib_id].tags))
        return True

    return any(matches(*c) for c in loaded)


@check
def device_param_in(s, i, a):
    t = _track(s, a["track"])
    if t is None:
        return False
    return any(
        d.device == a["device"] and a["min"] <= d.params.get(a["param"], -1) <= a["max"]
        for d in t.devices
    )


@check
def device_param_display_in(s, i, a):
    """The parameter's display value (Hz, ms, dB, ...) is within ``tol_pct`` percent of
    ``target``."""
    from openenv_daw.server.sim.catalog import load_catalog

    t = _track(s, a["track"])
    if t is None:
        return False
    spec = load_catalog().get(a["device"])
    for d in t.devices:
        if d.device == a["device"] and spec and a["param"] in spec.params:
            ps = spec.params[a["param"]]
            v = ps.to_value(d.params.get(a["param"], ps.default))
            return abs(v - a["target"]) <= abs(a["target"]) * a.get("tol_pct", 5.0) / 100
    return False


@check
def clip_count(s, i, a):
    t = _track(s, a["track"])
    return t is not None and len(t.clips) == a["n"]


@check
def clip_at(s, i, a):
    t = _track(s, a["track"])
    if t is None:
        return False
    return any(
        abs(c.start_beat - a["start_beat"]) < 1e-6
        and (a.get("length_beats") is None or abs(c.length_beats - a["length_beats"]) < 1e-6)
        for c in t.clips
    )


@check
def note_count_at_least(s, i, a):
    t = _track(s, a["track"])
    return t is not None and len(_scoped(t, a)) >= a["n"]


@check
def note_count_is(s, i, a):
    t = _track(s, a["track"])
    return t is not None and len(_notes(t)) == a["n"]


@check
def notes_in_key(s, i, a):
    t = _track(s, a["track"])
    ns = _notes(t) if t else []
    ok = set(scale_classes(a["key"]))
    return bool(ns) and all(n.pitch % 12 in ok for n in ns)


@check
def notes_pitch_range(s, i, a):
    t = _track(s, a["track"])
    ns = _scoped(t, a)
    return bool(ns) and all(a["min"] <= n.pitch <= a["max"] for n in ns)


@check
def notes_monophonic(s, i, a):
    t = _track(s, a["track"])
    ns = sorted(_abs_notes(t) if t else [], key=lambda x: x[0])
    if not ns:
        return False
    return all(
        ns[k][0] + ns[k][1].duration_beats <= ns[k + 1][0] + 1e-6 for k in range(len(ns) - 1)
    )


@check
def hits_on_beats(s, i, a):
    """Every listed absolute beat has a note of ``pitch`` starting within ``tol`` of it."""
    t = _track(s, a["track"])
    if t is None:
        return False
    starts = [b for b, n in _abs_notes(t) if n.pitch == a["pitch"]]
    tol = a.get("tol", 1e-6)
    return all(any(abs(b - x) <= tol for x in starts) for b in a["beats"])


@check
def no_notes_of_pitch_off_beats(s, i, a):
    """Notes of ``pitch`` only start on the listed beats."""
    t = _track(s, a["track"])
    if t is None:
        return False
    return all(
        any(abs(b - x) <= 1e-6 for x in a["beats"])
        for b, n in _abs_notes(t)
        if n.pitch == a["pitch"]
    )


@check
def hits_only_on(s, i, a):
    """Notes of ``pitch`` start only on the listed absolute beats (alias of the older check)."""
    return no_notes_of_pitch_off_beats(s, i, a)


@check
def only_pitches(s, i, a):
    """The track holds notes, and every one uses a pitch from ``pitches``."""
    t = _track(s, a["track"])
    ns = _notes(t) if t else []
    return bool(ns) and all(n.pitch in set(a["pitches"]) for n in ns)


@check
def distinct_pitches_at_least(s, i, a):
    t = _track(s, a["track"])
    return t is not None and len({n.pitch for n in _notes(t)}) >= a["n"]


@check
def notes_span(s, i, a):
    """Notes reach the last bar of every clip on the track (``bar_beats`` = beats per bar)."""
    t = _track(s, a["track"])
    if t is None or not t.clips:
        return False
    bar = a.get("bar_beats", 4.0)
    return all(any(n.start_beat >= c.length_beats - bar - 1e-6 for n in c.notes) for c in t.clips)


@check
def pitch_classes_within(s, i, a):
    """The track holds notes and every note's pitch class is in ``classes``."""
    t = _track(s, a["track"])
    ns = _scoped(t, a)
    return bool(ns) and all(n.pitch % 12 in set(a["classes"]) for n in ns)


@check
def existing_notes_kept(s, i, a):
    """Every note the track had initially is still there (same pitch, start, duration)."""
    t, t0 = _track(s, a["track"]), _track(i, a["track"])
    if t is None or t0 is None:
        return False
    now = {
        (c.start_beat, n.pitch, n.start_beat, n.duration_beats) for c in t.clips for n in c.notes
    }
    return all(
        (c.start_beat, n.pitch, n.start_beat, n.duration_beats) in now
        for c in t0.clips
        for n in c.notes
    )


@check
def clip_named(s, i, a):
    t = _track(s, a["track"])
    return t is not None and any(
        abs(c.start_beat - a["start_beat"]) < 1e-6 and c.name.lower() == a["name"].lower()
        for c in t.clips
    )


def chord_pitch_classes(key: str, numeral: str) -> set[int]:
    """Triad (pitch classes) for a roman numeral in ``key``: the degree comes from the numeral,
    the quality from its case (upper = major, lower = minor)."""
    return set(roman_chord(key, numeral))


@check
def chord_progression(s, i, a):
    """One block triad per bar: the notes starting in bar k have exactly the pitch classes of the
    k-th numeral of ``numerals`` in ``key`` (case gives the quality), start together (within
    ``spread`` beats) and are held for at least half a bar."""
    t = _track(s, a["track"])
    if t is None:
        return False
    bar = a.get("bar_beats", 4.0)
    spread = a.get("spread", 0.25)
    notes = [(b, n) for b, n in _abs_notes(t)]
    base = a.get("start_beat", min((c.start_beat for c in t.clips), default=0.0))
    for k, num in enumerate(a["numerals"]):
        lo, hi = base + k * bar - 1e-6, base + (k + 1) * bar - 1e-6
        win = [(b, n) for b, n in notes if lo <= b < hi]
        if len(win) < 3 or {n.pitch % 12 for _, n in win} != chord_pitch_classes(a["key"], num):
            return False
        if max(b for b, _ in win) - min(b for b, _ in win) > spread + 1e-6:
            return False
        if any(n.duration_beats < bar / 2 - 1e-6 for _, n in win):
            return False
    return True


@check
def notes_every_bar(s, i, a):
    """Every bar of every clip on the track has at least one note starting in it."""
    t = _track(s, a["track"])
    if t is None or not t.clips:
        return False
    bar = a.get("bar_beats", 4.0)
    for c in t.clips:
        for k in range(int(round(c.length_beats / bar))):
            if not any(k * bar - 1e-6 <= n.start_beat < (k + 1) * bar - 1e-6 for n in c.notes):
                return False
    return True


def _note_keys(t: Track) -> list[tuple[float, int, float, float]]:
    return [
        (c.start_beat, n.pitch, n.start_beat, n.duration_beats) for c in t.clips for n in c.notes
    ]


@check
def notes_transposed(s, i, a):
    """The track holds exactly its initial notes shifted by ``offset`` semitones, with the same
    starts and durations."""
    t, t0 = _track(s, a["track"]), _track(i, a["track"])
    if t is None or t0 is None:
        return False
    want = sorted((c, p + a["offset"], st, d) for c, p, st, d in _note_keys(t0))
    return sorted(_note_keys(t)) == want


@check
def existing_notes_kept_except(s, i, a):
    """Every initial note that is not of ``pitch`` is unchanged, and no note of ``pitch`` is
    left; nothing else was added."""
    t, t0 = _track(s, a["track"]), _track(i, a["track"])
    if t is None or t0 is None:
        return False
    want = sorted(k for k in _note_keys(t0) if k[1] != a["pitch"])
    return sorted(_note_keys(t)) == want


@check
def volume_in(s, i, a):
    t = _track(s, a["track"])
    return t is not None and a["min"] <= t.volume_db <= a["max"]


@check
def pan_in(s, i, a):
    t = _track(s, a["track"])
    return t is not None and a["min"] <= t.pan <= a["max"]


@check
def mute_is(s, i, a):
    t = _track(s, a["track"])
    return t is not None and t.mute == a["value"]


@check
def solo_is(s, i, a):
    t = _track(s, a["track"])
    return t is not None and t.solo == a["value"]


@check
def louder_than(s, i, a):
    t, o = _track(s, a["track"]), _track(s, a["other"])
    return t is not None and o is not None and t.volume_db >= o.volume_db + a.get("margin_db", 0.1)


@check
def no_fader_above(s, i, a):
    return all(t.volume_db <= a["db"] for t in s.tracks)


@check
def send_in(s, i, a):
    t, r = _track(s, a["track"]), _track(s, a["return_track"])
    if t is None or r is None:
        return False
    return a["min"] <= t.sends.get(r.id, -70.0) <= a["max"]


@check
def state_unchanged(s, i, a):
    return s.snapshot() == i.snapshot()


def run_check(kind: str, session: Session, initial: Session, args: dict[str, Any]) -> bool:
    fn = CHECKS.get(kind)
    if fn is None:
        raise KeyError(f"unknown check kind {kind!r}")
    return bool(fn(session, initial, args))
