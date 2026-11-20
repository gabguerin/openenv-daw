"""Seeded task generators. Each family loads its YAML (slot pools + intent templates) and builds
the initial session, declarative checks, touchable set and oracle solution from sampled slots.

Train tasks use train templates and train slot values; heldout tasks use heldout templates and
heldout slot values, so heldout measures generalization to unseen phrasing and values."""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from functools import cache
from importlib import resources
from typing import Any, Literal

import yaml
from openenv_daw.server.backends import SimBackend
from openenv_daw.server.music import (
    CLOSED_HAT,
    KICK,
    note_name,
    roman_chord,
    root_pc,
    scale_pitches,
    voice_lead,
)
from openenv_daw.server.sim.catalog import find_item, load_catalog, load_library
from openenv_daw.server.tasks.model import Check, MusicSpec, OracleStep, Task

SUBMIT_BLOCKED_STEP = OracleStep(
    tool="submit", args={"status": "blocked", "reason": "not possible with the available tools"}
)


def blocked_form(task: Task) -> Task:
    """An infeasible request: nothing may change and the right answer is ``submit("blocked")``."""
    return task.model_copy(
        update={
            "expected_status": "blocked",
            "checks": [Check(kind="state_unchanged", critical=True)],
            "oracle": [*(s for s in task.oracle if s.tool == "list_devices"), SUBMIT_BLOCKED_STEP],
            "touchable": [],
            "allowed_new": [],
            "music": [],
            "tags": task.tags if "blocked" in task.tags else [*task.tags, "blocked"],
        }
    )


PLACEHOLDER = re.compile(r"{(\w+)}")
Parts = dict[str, Any]


@cache
def load_templates() -> dict[str, dict[str, Any]]:
    return yaml.safe_load(
        resources.files("openenv_daw.server.tasks").joinpath("templates.yaml").read_text()
    )


def load_family(family: str) -> dict[str, Any]:
    return load_templates()[family]


def families() -> dict[str, float]:
    names = [
        "session_setup",
        "melody_in_key",
        "four_on_floor_beat",
        "reverb_send",
        "mix_balance",
        "near_miss",
        "infeasible",
        "chord_progression",
        "edit_notes",
        "device_tweak",
        "disambiguation",
        "compositional",
        "load_sound",
        "remove_device",
        "clip_edit",
    ]
    return {n: float(load_family(n)["weight"]) for n in names}


def build_session(steps: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    b = SimBackend()
    for tool, args in steps:
        r = b.call(tool, args)
        assert r["ok"], (tool, args, r)
    return b.snapshot()


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _step(tool: str, **args: Any) -> OracleStep:
    return OracleStep(tool=tool, args=args)


SUBMIT_DONE = _step("submit", status="done")


# --- families: (rng, pool, used) -> Parts ---------------------------------------------------


def gen_session_setup(rng: random.Random, pool: dict[str, list[Any]], used: set[str]) -> Parts:
    vals: dict[str, str] = {}
    checks: list[Check] = []
    oracle: list[OracleStep] = []
    touch: list[str] = ["transport"]
    if "bpm" in used:
        bpm = rng.choice(pool["bpm"])
        vals["bpm"] = str(bpm)
        checks.append(Check(kind="tempo_is", args={"bpm": bpm}, critical=True))
        oracle.append(_step("set_transport", tempo_bpm=bpm))
    if "ts" in used:
        ts = rng.choice(pool["ts"])
        if "names" not in used and ts == "4/4" and vals.get("bpm") == "120":
            ts = rng.choice([x for x in pool["ts"] if x != "4/4"])  # never the default session
        num, den = (int(x) for x in ts.split("/"))
        vals["ts"] = ts
        checks.append(
            Check(
                kind="time_signature_is", args={"numerator": num, "denominator": den}, critical=True
            )
        )
        oracle.append(_step("set_transport", time_signature=[num, den]))
    if "names" in used:
        names = rng.sample(pool["names"], rng.randint(2, 3))
        vals["names"] = _join(names)
        for n in names:
            checks.append(Check(kind="track_exists", args={"track": n, "kind": "midi"}))
            oracle.append(_step("create_track", name=n, kind="midi"))
        checks.append(Check(kind="track_count", args={"kind": "midi", "n": len(names)}))
    return {
        "vals": vals,
        "checks": checks,
        "oracle": oracle,
        "touchable": touch,
        "allowed_new": ([*names, *(f"clip:{n}:0" for n in names)] if "names" in used else []),
    }


_RHYTHM_CELLS = [
    [1.0, 1.0, 1.0, 1.0],
    [1.5, 0.5, 1.0, 1.0],
    [1.0, 0.5, 0.5, 1.0, 1.0],
    [0.5, 0.5, 1.0, 2.0],
    [1.0, 1.0, 2.0],
    [0.5, 0.5, 0.5, 0.5, 1.0, 1.0],
    [2.0, 1.0, 1.0],
]
_CLOSING_CELLS = [[1.0, 1.0, 2.0], [1.0, 0.5, 0.5, 2.0], [0.5, 0.5, 1.0, 2.0]]

# Rhythm and closing cells (each sums to the bar) for bars that are not 4 beats long.
_METER_CELLS: dict[float, tuple[list[list[float]], list[list[float]]]] = {
    2.0: (
        [[1.0, 1.0], [0.5, 0.5, 1.0], [1.0, 0.5, 0.5], [1.5, 0.5]],
        [[1.0, 1.0], [0.5, 0.5, 1.0]],
    ),
    3.0: (
        [[1.0, 1.0, 1.0], [1.5, 0.5, 1.0], [1.0, 0.5, 0.5, 1.0], [2.0, 1.0], [1.0, 2.0]],
        [[1.0, 2.0], [1.0, 0.5, 1.5], [0.5, 0.5, 2.0]],
    ),
    3.5: (
        [[1.0, 1.0, 1.0, 0.5], [1.5, 0.5, 1.0, 0.5], [1.0, 1.0, 1.5], [0.5, 0.5, 1.0, 1.5]],
        [[1.0, 1.0, 1.5], [1.0, 0.5, 0.5, 1.5]],
    ),
    5.0: (
        [[1.0] * 5, [1.5, 0.5, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 2.0], [2.0, 1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0, 2.0], [1.0, 0.5, 0.5, 1.0, 2.0], [2.0, 1.0, 2.0]],
    ),
}


def meter_cells(bar_beats: float) -> tuple[list[list[float]], list[list[float]]]:
    """Rhythm cells and closing cells for a bar of ``bar_beats`` beats (4/4 keeps the originals)."""
    if bar_beats == 4.0:
        return _RHYTHM_CELLS, _CLOSING_CELLS
    if bar_beats in _METER_CELLS:
        return _METER_CELLS[bar_beats]
    raise ValueError(f"no rhythm cells for a bar of {bar_beats} beats")


def compose_melody(
    rng: random.Random, key: str, bars: int, bar_beats: float = 4.0
) -> list[dict[str, Any]]:
    """A short, singable line in ``key``: a one-bar motif that returns (transposed by scale
    steps) across the bars, mostly stepwise, leaps answered by a step back, a held tonic at the
    end. The best of several seeded candidates by the musicality heuristics is kept."""
    from openenv_daw.server.rewards.musicality import melody_score

    sp = scale_pitches(key, 57, 81)
    tonic = root_pc(key.split()[0])
    tonics = [i for i, p in enumerate(sp) if p % 12 == tonic and 60 <= p <= 72]
    best: tuple[float, list[dict[str, Any]]] = (-1.0, [])
    rhythm_cells, closing_cells = meter_cells(bar_beats)
    for _ in range(40):
        motif_cell = rng.choice(rhythm_cells)
        answer_cell = rng.choice(rhythm_cells)
        closing = rng.choice(closing_cells)
        steps = [rng.choice([1, 1, -1, -1, 2, -2, 3, -3]) for _ in motif_cell[:-1]]
        for k, st in enumerate(steps[:-1]):  # a leap is answered by a step the other way
            if abs(st) >= 3:
                steps[k + 1] = -1 if st > 0 else 1
        idx = rng.choice([tonics[0], tonics[0] + 2, tonics[0] + 4])
        shifts = [0] + [rng.choice([0, 1, -1, 2]) for _ in range(max(0, bars - 2))]
        notes: list[dict[str, Any]] = []
        beat = 0.0
        for bar in range(bars):
            last = bar == bars - 1
            cell = closing if last else motif_cell if bar % 2 == 0 else answer_cell
            idx0 = idx
            for j, d in enumerate(cell):
                i = max(0, min(len(sp) - 1, idx))
                if last and j == len(cell) - 1:
                    i = min(tonics, key=lambda t: abs(t - idx))
                p = sp[i]
                vel = 100 if abs(beat % bar_beats) < 1e-6 else 88
                notes.append({"pitch": p, "start_beat": beat, "duration_beats": d, "velocity": vel})
                beat += d
                step = steps[j % len(steps)] if steps else 1
                if bar % 2 == 1 and not last:
                    step = -step  # the answer mirrors the motif
                idx = i + step
            if bars > 1 and not last:
                idx = max(0, min(len(sp) - 1, idx0 + shifts[min(bar + 1, len(shifts) - 1)]))
        score, _ = melody_score(
            [(n["start_beat"], n["pitch"], n["duration_beats"], n["velocity"]) for n in notes],
            key,
        )
        if score > best[0]:
            best = (score, notes)
        if score >= 0.9:
            break
    return best[1]


def gen_melody_in_key(
    rng: random.Random, pool: dict[str, list[Any]], used: set[str], bar_beats: float = 4.0
) -> Parts:
    key, bars, track = rng.choice(pool["key"]), rng.choice(pool["bars"]), rng.choice(pool["track"])
    length: float = bars * bar_beats
    if length.is_integer():
        length = int(length)  # keep 4/4 oracles (and SFT text) as they were: 16, not 16.0
    notes = compose_melody(rng, key, bars, bar_beats)
    checks = [
        Check(kind="track_exists", args={"track": track, "kind": "midi"}, critical=True),
        Check(kind="has_device", args={"track": track, "device": "simple_synth"}, critical=True),
        Check(kind="notes_in_key", args={"track": track, "key": key}, critical=True),
        Check(kind="distinct_pitches_at_least", args={"track": track, "n": 3}, critical=True),
        Check(kind="notes_span", args={"track": track, "bar_beats": bar_beats}, critical=True),
        Check(
            kind="clip_at",
            args={"track": track, "start_beat": 0.0, "length_beats": float(length)},
            critical=True,
        ),
        Check(kind="notes_every_bar", args={"track": track, "bar_beats": bar_beats}, critical=True),
        Check(kind="note_count_at_least", args={"track": track, "n": max(2, int(length // 2))}),
        Check(kind="notes_monophonic", args={"track": track}),
        Check(
            kind="notes_pitch_range",
            args={"track": track, "min": 48, "max": 84},
            critical=True,
        ),
    ]
    oracle = [
        _step("create_track", name=track, kind="midi"),
        _step("insert_device", track=track, device="simple_synth"),
        _step("create_midi_clip", track=track, start_beat=0, length_beats=length, name="Melody"),
        _step("insert_midi_notes", clip="Melody", notes=notes),
    ]
    return {
        "vals": {"key": key, "bars": str(bars), "track": track},
        "checks": checks,
        "oracle": oracle,
        "touchable": [],
        "music": [MusicSpec(kind="melody", track=track, key=key, bar_beats=bar_beats)],
        "allowed_new": [track, "device:simple_synth", f"clip:{track}:1"],
    }


def gen_four_on_floor(
    rng: random.Random, pool: dict[str, list[Any]], used: set[str], bar_beats: float = 4.0
) -> Parts:
    if bar_beats % 1:
        raise ValueError("a quarter-note kick needs a bar of whole beats")
    bars, track = rng.choice(pool["bars"]), rng.choice(pool["track"])
    clause = rng.choice(pool["hat_clause"]) if "hat_clause" in used else ""
    length = int(bars * bar_beats)
    kicks = [float(b) for b in range(length)]
    hats = [i * 0.5 for i in range(length * 2)]
    notes = [
        {
            "pitch": KICK,
            "start_beat": b,
            "duration_beats": 0.25,
            "velocity": 115 if b % bar_beats == 0 else 100,
        }
        for b in kicks
    ]
    checks = [
        Check(kind="track_exists", args={"track": track, "kind": "midi"}, critical=True),
        Check(kind="has_device", args={"track": track, "device": "drum_kit"}, critical=True),
        Check(
            kind="hits_on_beats",
            args={"track": track, "pitch": KICK, "beats": kicks},
            critical=True,
        ),
        Check(
            kind="hits_only_on",
            args={"track": track, "pitch": KICK, "beats": kicks},
            critical=True,
        ),
        Check(
            kind="only_pitches",
            args={"track": track, "pitches": [KICK, CLOSED_HAT] if clause else [KICK]},
            critical=True,
        ),
        Check(
            kind="clip_at", args={"track": track, "start_beat": 0.0, "length_beats": float(length)}
        ),
    ]
    oracle = [
        _step("create_track", name=track, kind="midi"),
        _step("insert_device", track=track, device="drum_kit"),
        _step("create_midi_clip", track=track, start_beat=0, length_beats=length, name="Beat"),
    ]
    vals = {"bars": str(bars), "track": track, "hat_clause": clause}
    if clause:
        notes += [
            {
                "pitch": CLOSED_HAT,
                "start_beat": b,
                "duration_beats": 0.25,
                "velocity": 90 if b % 1 == 0 else 68,
            }
            for b in hats
        ]
        checks.append(
            Check(
                kind="hits_on_beats",
                args={"track": track, "pitch": CLOSED_HAT, "beats": hats},
                critical=True,
            )
        )
        checks.append(
            Check(
                kind="hits_only_on",
                args={"track": track, "pitch": CLOSED_HAT, "beats": hats},
                critical=True,
            )
        )
    oracle.append(_step("insert_midi_notes", clip="Beat", notes=notes))
    touch: list[str] = []
    if "bpm" in used:
        bpm = rng.choice(pool["bpm"])
        vals["bpm"] = str(bpm)
        checks.append(Check(kind="tempo_is", args={"bpm": bpm}, critical=True))
        oracle.append(_step("set_transport", tempo_bpm=bpm))
        touch.append("transport")
    return {
        "vals": vals,
        "checks": checks,
        "oracle": oracle,
        "touchable": touch,
        "music": [MusicSpec(kind="drums", track=track, bar_beats=bar_beats)],
        "allowed_new": [track, "device:drum_kit", f"clip:{track}:1"],
    }


def gen_reverb_send(rng: random.Random, pool: dict[str, list[Any]], used: set[str]) -> Parts:
    names = rng.sample(pool["names"], rng.randint(2, min(3, len(pool["names"]))))
    source, ret = rng.choice(names), rng.choice(pool["return_name"])
    db = rng.choice(pool["send_db"])
    initial = build_session([("create_track", {"name": n, "kind": "midi"}) for n in names])
    checks = [
        Check(kind="track_exists", args={"track": ret, "kind": "return"}, critical=True),
        Check(kind="has_device", args={"track": ret, "device": "reverb"}, critical=True),
        Check(
            kind="send_in",
            args={"track": source, "return_track": ret, "min": db - 0.5, "max": db + 0.5},
            critical=True,
        ),
    ]
    oracle = [
        _step("create_track", name=ret, kind="return"),
        _step("insert_device", track=ret, device="reverb"),
        _step("set_send", track=source, return_track=ret, level_db=db),
    ]
    vals: dict[str, str] = {"source": source, "return_name": ret, "send_db": str(db)}
    if "mix" in used:
        mix = rng.choice(pool["mix"])
        vals["mix"] = str(mix)
        checks.append(
            Check(
                kind="device_param_in",
                args={
                    "track": ret,
                    "device": "reverb",
                    "param": "mix",
                    "min": mix / 100 - 0.03,
                    "max": mix / 100 + 0.03,
                },
            )
        )
        oracle.append(
            _step("set_device_param", track=ret, device="reverb", param="mix", value=mix / 100)
        )
    return {
        "vals": vals,
        "checks": checks,
        "oracle": oracle,
        "initial": initial,
        "touchable": [f"track:{source}:sends"],
        "allowed_new": [ret, "device:reverb", f"clip:{ret}:0"],
    }


def gen_mix_balance(rng: random.Random, pool: dict[str, list[Any]], used: set[str]) -> Parts:
    names = rng.sample(pool["names"], 4)
    a, p, m = names[0], names[1], names[2]
    initial = build_session([("create_track", {"name": n, "kind": "midi"}) for n in names])
    vals: dict[str, str] = {}
    checks: list[Check] = [Check(kind="no_fader_above", args={"db": 0.0})]
    oracle: list[OracleStep] = []
    touch: list[str] = []
    if "db_a" in used:
        db = rng.choice(pool["db_a"])
        vals.update(track_a=a, db_a=str(db))
        checks.append(
            Check(
                kind="volume_in", args={"track": a, "min": db - 0.5, "max": db + 0.5}, critical=True
            )
        )
        oracle.append(_step("set_mixer_channel", track=a, volume_db=db))
        touch.append(f"track:{a}:mixer")
    if "side" in used:
        side = rng.choice(pool["side"])
        lo, hi = (-1.0, -0.9) if side == "left" else (0.9, 1.0)
        vals.update(pan_track=p, side=side)
        checks.append(Check(kind="pan_in", args={"track": p, "min": lo, "max": hi}, critical=True))
        oracle.append(_step("set_mixer_channel", track=p, pan=-1.0 if side == "left" else 1.0))
        touch.append(f"track:{p}:mixer")
    if "mute_track" in used:
        vals["mute_track"] = m
        checks.append(Check(kind="mute_is", args={"track": m, "value": True}, critical=True))
        oracle.append(_step("set_mixer_channel", track=m, mute=True))
        touch.append(f"track:{m}:mixer")
    return {
        "vals": vals,
        "checks": checks,
        "oracle": oracle,
        "initial": initial,
        "touchable": touch,
    }


def gen_near_miss(rng: random.Random, pool: dict[str, list[Any]], used: set[str]) -> Parts:
    """Feasible requests that mention infeasible-sounding words (group, plugin names, reverb), so
    a policy cannot refuse by keyword matching."""
    names = rng.sample(pool["names"], 3)
    a, b = names[0], names[1]
    initial = build_session([("create_track", {"name": n, "kind": "midi"}) for n in names])
    vals = {"a": a, "b": b}
    base: Parts = {"vals": vals, "initial": initial, "allowed_new": []}
    if "group" in used:
        g = rng.choice(pool["group"])
        vals["group"] = g
        return base | {
            "checks": [
                Check(kind="track_exists", args={"track": g, "kind": "midi"}, critical=True)
            ],
            "oracle": [_step("create_track", name=g, kind="midi")],
            "touchable": [],
            "allowed_new": [g],
        }
    if "plugin" in used:
        p = rng.choice(pool["plugin"])
        vals["plugin"] = p
        return base | {
            "checks": [
                Check(kind="track_exists", args={"track": p, "kind": "midi"}, critical=True),
                Check(kind="track_absent", args={"track": a}, critical=True),
            ],
            "oracle": [_step("rename_track", track=a, name=p)],
            "touchable": [f"track:{a}:mixer"],
        }
    if "db" in used:
        db = rng.choice(pool["db"])
        vals["db"] = str(db)
        return base | {
            "checks": [
                Check(
                    kind="volume_in",
                    args={"track": a, "min": db - 0.5, "max": db + 0.5},
                    critical=True,
                )
            ],
            "oracle": [_step("set_mixer_channel", track=a, volume_db=db)],
            "touchable": [f"track:{a}:mixer"],
        }
    if "b" in used:
        return base | {
            "checks": [
                Check(kind="mute_is", args={"track": a, "value": True}, critical=True),
                Check(kind="mute_is", args={"track": b, "value": True}, critical=True),
            ],
            "oracle": [
                _step("set_mixer_channel", track=a, mute=True),
                _step("set_mixer_channel", track=b, mute=True),
            ],
            "touchable": [f"track:{a}:mixer", f"track:{b}:mixer"],
        }
    return base | {
        "checks": [Check(kind="has_device", args={"track": a, "device": "reverb"}, critical=True)],
        "oracle": [_step("insert_device", track=a, device="reverb")],
        "touchable": [f"track:{a}:devices"],
        "allowed_new": ["device:reverb"],
    }


def gen_infeasible(rng: random.Random, pool: dict[str, list[Any]], used: set[str]) -> Parts:
    names = rng.sample(pool["names"], 3)
    initial = build_session([("create_track", {"name": n, "kind": "midi"}) for n in names])
    vals = {"a": names[0], "b": names[1]}
    for k in ("group", "plugin"):
        if k in used:
            vals[k] = rng.choice(pool[k])
    return {
        "vals": vals,
        "checks": [Check(kind="state_unchanged", critical=True)],
        # a plugin request is refused after looking it up in the library (read-only)
        "oracle": [_step("list_devices", query=vals["plugin"])] if "plugin" in used else [],
        "initial": initial,
        "touchable": [],
    }


# --- newer families -------------------------------------------------------------------------


def _pname(p: int) -> str:
    return note_name(p)


_PNAME_TO_PITCH = {note_name(p): p for p in range(24, 100)}
_IDEA_NOTES = [60, 62, 64, 65, 67, 69, 71, 72]  # one octave of C major, one note per beat


def _idea_session(track: str, length: float = 16.0) -> dict[str, Any]:
    notes = [
        {"pitch": p, "start_beat": float(i), "duration_beats": 0.5, "velocity": 100}
        for i, p in enumerate(_IDEA_NOTES)
    ]
    return build_session(
        [
            ("create_track", {"name": track, "kind": "midi"}),
            ("insert_device", {"track": track, "device": "simple_synth"}),
            (
                "create_midi_clip",
                {"track": track, "start_beat": 0, "length_beats": length, "name": "Idea"},
            ),
            ("insert_midi_notes", {"clip": "Idea", "notes": notes}),
        ]
    )


def gen_chord_progression(rng: random.Random, pool: dict[str, list[Any]], used: set[str]) -> Parts:
    key, track = rng.choice(pool["key"]), rng.choice(pool["track"])
    mode = key.split()[1].lower()
    numerals = rng.choice(pool["prog_major" if mode == "major" else "prog_minor"]).split("-")
    length = 4 * len(numerals)
    notes = []
    prev: list[int] | None = None
    for k, num in enumerate(numerals):
        prev = voice_lead(prev, roman_chord(key, num))
        notes += [
            {"pitch": p, "start_beat": 4.0 * k, "duration_beats": 4.0, "velocity": 84 + 8 * (k % 2)}
            for p in prev
        ]
    checks = [
        Check(kind="track_exists", args={"track": track, "kind": "midi"}, critical=True),
        Check(kind="has_device", args={"track": track, "device": "simple_synth"}, critical=True),
        Check(
            kind="chord_progression",
            args={"track": track, "key": key, "numerals": numerals, "start_beat": 0.0},
            critical=True,
        ),
        Check(
            kind="clip_at", args={"track": track, "start_beat": 0.0, "length_beats": float(length)}
        ),
        Check(kind="clip_count", args={"track": track, "n": 1}),
    ]
    oracle = [
        _step("create_track", name=track, kind="midi"),
        _step("insert_device", track=track, device="simple_synth"),
        _step("create_midi_clip", track=track, start_beat=0, length_beats=length, name="Chords"),
        _step("insert_midi_notes", clip="Chords", notes=notes),
    ]
    return {
        "vals": {
            "key": key,
            "track": track,
            "prog": "-".join(numerals),
            "bars": str(len(numerals)),
        },
        "checks": checks,
        "oracle": oracle,
        "touchable": [],
        "music": [MusicSpec(kind="chords", track=track)],
        "allowed_new": [track, "device:simple_synth", f"clip:{track}:1"],
    }


def gen_edit_notes(rng: random.Random, pool: dict[str, list[Any]], used: set[str]) -> Parts:
    track = rng.choice(pool["track"])
    initial = _idea_session(track)
    touch = [f"track:{track}:clips"]
    base = [
        {"pitch": p, "start_beat": float(i), "duration_beats": 0.5, "velocity": 100}
        for i, p in enumerate(_IDEA_NOTES)
    ]
    vals: dict[str, str] = {"track": track}
    if "pitch" in used:  # delete every note of one pitch
        name = rng.choice(pool["pitch"])
        p = _PNAME_TO_PITCH[name]
        vals["pitch"] = name
        left = sorted({q for q in _IDEA_NOTES if q != p})
        checks = [
            Check(kind="only_pitches", args={"track": track, "pitches": left}, critical=True),
            Check(kind="note_count_is", args={"track": track, "n": len(left)}, critical=True),
            Check(
                kind="existing_notes_kept_except", args={"track": track, "pitch": p}, critical=True
            ),
            Check(kind="clip_count", args={"track": track, "n": 1}),
        ]
        oracle = [_step("remove_midi_notes", clip="Idea", pitch=p)]
    elif "beat" in used:  # add a note, keep the rest
        name, beat = rng.choice(pool["add"]), rng.choice(pool["beat"])
        p = _PNAME_TO_PITCH[name]
        vals.update(add=name, beat=str(beat))
        checks = [
            Check(kind="existing_notes_kept", args={"track": track}, critical=True),
            Check(
                kind="hits_on_beats",
                args={"track": track, "pitch": p, "beats": [float(beat)]},
                critical=True,
            ),
            Check(
                kind="note_count_is",
                args={"track": track, "n": len(_IDEA_NOTES) + 1},
                critical=True,
            ),
            Check(kind="clip_count", args={"track": track, "n": 1}),
        ]
        oracle = [
            _step(
                "insert_midi_notes",
                clip="Idea",
                notes=[
                    {"pitch": p, "start_beat": float(beat), "duration_beats": 1.0, "velocity": 100}
                ],
            )
        ]
    else:  # transpose everything up an octave
        up = [q + 12 for q in _IDEA_NOTES]
        checks = [
            Check(kind="only_pitches", args={"track": track, "pitches": up}, critical=True),
            Check(kind="note_count_is", args={"track": track, "n": len(up)}, critical=True),
            Check(kind="notes_transposed", args={"track": track, "offset": 12}, critical=True),
            Check(kind="clip_count", args={"track": track, "n": 1}),
        ]
        oracle = [
            _step("remove_midi_notes", clip="Idea"),
            _step(
                "insert_midi_notes",
                clip="Idea",
                notes=[{**n, "pitch": n["pitch"] + 12} for n in base],
            ),
        ]
    return {
        "vals": vals,
        "checks": checks,
        "oracle": oracle,
        "initial": initial,
        "touchable": touch,
    }


def gen_device_tweak(rng: random.Random, pool: dict[str, list[Any]], used: set[str]) -> Parts:
    track = rng.choice(pool["track"])
    cat = load_catalog()

    def _is_default(setting: str) -> bool:
        dev, prm, tgt_, _ = setting.split("|")
        ps = cat[dev].params[prm]
        return abs(ps.to_value(ps.default) - float(tgt_)) <= abs(float(tgt_)) * 0.05

    options = [x for x in pool["setting"] if not _is_default(x)] or pool["setting"]
    device, param, target, text = rng.choice(options).split("|")
    spec = cat[device].params[param]
    tgt = float(target)
    initial = build_session(
        [
            ("create_track", {"name": track, "kind": "midi"}),
            ("insert_device", {"track": track, "device": device}),
        ]
    )
    return {
        "vals": {"track": track, "device": device, "param": param, "text": text},
        "checks": [
            Check(kind="has_device", args={"track": track, "device": device}, critical=True),
            Check(
                kind="device_param_display_in",
                args={
                    "track": track,
                    "device": device,
                    "param": param,
                    "target": tgt,
                    "tol_pct": 5.0,
                },
                critical=True,
            ),
        ],
        "oracle": [
            _step(
                "set_device_param",
                track=track,
                device=device,
                param=param,
                value=round(spec.to_norm(tgt), 6),
            )
        ],
        "initial": initial,
        "touchable": [f"track:{track}:devices"],
    }


def gen_disambiguation(rng: random.Random, pool: dict[str, list[Any]], used: set[str]) -> Parts:
    a, b = rng.sample(pool["names"], 2)
    initial = build_session(
        [("create_track", {"name": n, "kind": "midi"}) for n in (a, b)]
        + [
            ("create_midi_clip", {"track": n, "start_beat": 0, "length_beats": 4, "name": "Loop"})
            for n in (a, b)
        ]
    )
    b_clip = next(c["id"] for t in initial["tracks"] if t["name"] == b for c in t["clips"])
    vals = {"a": a, "b": b}
    if "pitch" in used:
        name = rng.choice(pool["pitch"])
        vals["pitch"] = name
        checks = [
            Check(
                kind="hits_on_beats",
                args={"track": b, "pitch": _PNAME_TO_PITCH[name], "beats": [0.0]},
                critical=True,
            ),
            Check(kind="note_count_is", args={"track": b, "n": 1}, critical=True),
            Check(kind="note_count_is", args={"track": a, "n": 0}, critical=True),
        ]
        oracle = [
            _step(
                "insert_midi_notes",
                clip=b_clip,
                notes=[
                    {
                        "pitch": _PNAME_TO_PITCH[name],
                        "start_beat": 0.0,
                        "duration_beats": 1.0,
                        "velocity": 100,
                    }
                ],
            )
        ]
    else:
        checks = [
            Check(kind="clip_count", args={"track": b, "n": 0}, critical=True),
            Check(kind="clip_count", args={"track": a, "n": 1}, critical=True),
        ]
        oracle = [_step("delete_clip", clip=b_clip)]
    return {
        "vals": vals,
        "checks": checks,
        "oracle": oracle,
        "initial": initial,
        "touchable": [f"track:{b}:clips"],
    }


def gen_load_sound(rng: random.Random, pool: dict[str, list[Any]], used: set[str]) -> Parts:
    """Find a named or described sound, rack or plugin with list_devices(query), then load it."""
    track = rng.choice(pool["track"])
    vals: dict[str, str] = {"track": track}
    lib = load_library()
    tag: str | None = None
    if "sound" in used:
        item = lib[rng.choice(pool["sound"])]
        vals["sound"] = item.name
        query = item.name
    elif "style" in used:
        tag, phrase, iid = rng.choice(pool["style"]).split("|")
        item = lib[iid]
        vals["style"] = phrase
        query = tag
    elif "rack" in used:
        item = lib[rng.choice(pool["rack"])]
        vals["rack"] = item.name
        query = item.name
    else:
        item = find_item(rng.choice(pool["plugin"]))  # type: ignore[assignment]
        vals["plugin"] = item.name
        query = item.name
    if tag is not None:
        match: dict[str, Any] = {"tags_any": [tag]}
    elif item.kind == "preset":
        match = {"preset": item.id}
    else:
        match = {"item": item.id}
    allowed = [track]
    allowed.append(f"device:{item.base or item.id}")  # racks and plugins load as foreign devices
    return {
        "vals": vals,
        "checks": [
            Check(kind="track_exists", args={"track": track, "kind": "midi"}, critical=True),
            Check(kind="has_device", args={"track": track, **match}, critical=True),
        ],
        "oracle": [
            _step("create_track", name=track, kind="midi"),
            _step("list_devices", query=query),
            _step("insert_device", track=track, device=item.id),
        ],
        "touchable": [],
        "allowed_new": allowed,
    }


_FX = ["eq", "compressor", "reverb", "delay", "limiter"]


def gen_remove_device(rng: random.Random, pool: dict[str, list[Any]], used: set[str]) -> Parts:
    """delete_device: remove one effect, swap it for another, remove it on one of two tracks, or
    (blocked) remove a rack, which is read-only."""
    track = rng.choice(pool["track"])

    def chain(name: str, effects: list[str]) -> list[tuple[str, dict[str, Any]]]:
        steps: list[tuple[str, dict[str, Any]]] = [
            ("create_track", {"name": name, "kind": "midi"}),
            ("insert_device", {"track": name, "device": "simple_synth"}),
        ]
        return steps + [("insert_device", {"track": name, "device": e}) for e in effects]

    if "rack" in used:
        item = load_library()[rng.choice(pool["rack"])]
        initial = build_session(
            [
                *chain(track, rng.sample(_FX, 1))[:2],
                ("insert_device", {"track": track, "device": item.id}),
            ]
        )
        return {
            "vals": {"track": track, "rack": item.name},
            "checks": [],
            "oracle": [],
            "initial": initial,
            "touchable": [],
            "blocked": True,
        }
    effect = rng.choice(pool["effect"])
    new = rng.choice(pool["new"]) if "new" in used else None
    others = rng.sample([e for e in _FX if e not in (effect, new)], rng.randint(1, 2))
    vals = {"track": track, "effect": effect}
    keep = [
        Check(kind="has_device", args={"track": track, "device": "simple_synth"}, critical=True)
    ]
    keep += [
        Check(kind="has_device", args={"track": track, "device": e}, critical=True) for e in others
    ]
    checks = [Check(kind="device_absent", args={"track": track, "device": effect}, critical=True)]
    steps = chain(track, [effect, *others])
    oracle = [_step("delete_device", track=track, device=effect)]
    allowed: list[str] = []
    if new is not None:
        vals["new"] = new
        checks.append(Check(kind="has_device", args={"track": track, "device": new}, critical=True))
        oracle.append(_step("insert_device", track=track, device=new))
        allowed = [f"device:{new}"]
    if "b" in used:
        other = rng.choice(pool["b"])
        vals["b"] = other
        steps += chain(other, [effect])
        # the same effect must stay on the other track; the one on `other` is the one to remove
        checks = [
            Check(kind="device_absent", args={"track": other, "device": effect}, critical=True)
        ]
        keep = [
            Check(kind="has_device", args={"track": track, "device": effect}, critical=True),
            Check(
                kind="has_device", args={"track": other, "device": "simple_synth"}, critical=True
            ),
        ] + [
            Check(kind="has_device", args={"track": track, "device": e}, critical=True)
            for e in others
        ]
        oracle = [_step("delete_device", track=other, device=effect)]
        touch = [f"track:{other}:devices"]
    else:
        touch = [f"track:{track}:devices"]
    return {
        "vals": vals,
        "checks": checks + keep,
        "oracle": oracle,
        "initial": build_session(steps),
        "touchable": touch,
        "allowed_new": allowed,
    }


def gen_clip_edit(rng: random.Random, pool: dict[str, list[Any]], used: set[str]) -> Parts:
    """update_midi_clip: rename, move, extend or shorten the one clip of a track (clip id kept)."""
    track = rng.choice(pool["track"])
    initial = _idea_session(track)
    n_notes = len(_IDEA_NOTES)
    checks = [Check(kind="clip_kept", args={"track": track}, critical=True)]
    vals: dict[str, str] = {"track": track}
    start, length, name = 0.0, 16.0, "Idea"
    update: dict[str, Any] = {}
    oracle: list[OracleStep] = []
    kept_before: float | None = None
    if "name" in used:
        name = rng.choice(pool["name"])
        vals["name"] = name
        update = {"name": name}
    elif "bar" in used:
        bar = rng.choice(pool["bar"])
        vals["bar"] = str(bar)
        start = 4.0 * (bar - 1)
        update = {"start_beat": start}
    elif "ebars" in used:
        bars = rng.choice(pool["ebars"])
        vals["ebars"] = str(bars)
        length = 4.0 * bars
        update = {"length_beats": length}
    else:
        bars = rng.choice(pool["sbars"])
        vals["sbars"] = str(bars)
        length = 4.0 * bars
        update = {"length_beats": length}
        if length < n_notes:  # notes starting at or after the new end must go first
            kept_before = length
            n_notes = int(length)
            oracle.append(_step("remove_midi_notes", clip="Idea", start_beat=length))
    oracle.append(_step("update_midi_clip", clip="Idea", **update))
    checks += [
        Check(
            kind="clip_at",
            args={"track": track, "start_beat": start, "length_beats": length},
            critical=True,
        ),
        Check(
            kind="clip_named",
            args={"track": track, "name": name},
            critical=True,
        ),
        Check(
            kind="clip_notes_kept", args={"track": track, "before_beat": kept_before}, critical=True
        ),
        Check(kind="note_count_is", args={"track": track, "n": n_notes}, critical=True),
        Check(kind="clip_count", args={"track": track, "n": 1}),
    ]
    return {
        "vals": vals,
        "checks": checks,
        "oracle": oracle,
        "initial": initial,
        "touchable": [f"track:{track}:clips"],
    }


GENERATORS: dict[str, Callable[[random.Random, dict[str, list[Any]], set[str]], Parts]] = {
    "session_setup": gen_session_setup,
    "melody_in_key": gen_melody_in_key,
    "four_on_floor_beat": gen_four_on_floor,
    "reverb_send": gen_reverb_send,
    "mix_balance": gen_mix_balance,
    "near_miss": gen_near_miss,
    "infeasible": gen_infeasible,
    "chord_progression": gen_chord_progression,
    "edit_notes": gen_edit_notes,
    "device_tweak": gen_device_tweak,
    "disambiguation": gen_disambiguation,
    "load_sound": gen_load_sound,
    "remove_device": gen_remove_device,
    "clip_edit": gen_clip_edit,
}


def generate(
    family: str,
    seed: int,
    split: Literal["train", "heldout"] = "train",
) -> Task:
    """Deterministically build a task from (family, split, seed)."""
    if family == "compositional":
        return _generate_compositional(seed, split)
    data = load_family(family)
    rng = random.Random(f"{family}:{split}:{seed}")
    templates = [t for t in data["templates"] if t["split"] == split]
    tpl = rng.choice(templates)
    used = set(PLACEHOLDER.findall(tpl["text"]))
    pool = {k: v[split] for k, v in data["slots"].items()}
    parts = GENERATORS[family](rng, pool, used)
    intent = tpl["text"].format(**parts["vals"])
    promoted = set(tpl.get("critical", []))  # wording the template promises, enforced hard
    for c in parts["checks"]:
        if c.kind in promoted:
            c.critical = True
    oracle = parts["oracle"] + [SUBMIT_DONE]
    if family == "infeasible":
        oracle = parts["oracle"] + [SUBMIT_BLOCKED_STEP]
    task = Task(
        task_id=f"{family}-{split}-{seed}",
        family=family,
        intent=intent,
        split=split,
        initial_state=parts.get("initial"),
        checks=parts["checks"],
        touchable=parts["touchable"],
        allowed_new=parts.get("allowed_new", []),
        oracle=oracle,
        music=parts.get("music", []),
        tags=[difficulty(len(oracle) - 1)] + (["blocked"] if family == "infeasible" else []),
        expected_status="blocked" if family == "infeasible" else "done",
    )
    return blocked_form(task) if parts.get("blocked") else task


def difficulty(n_calls: int, parts: int = 1) -> str:
    """L1: up to 3 oracle calls, L2: up to 6, else L3; compositions are always L3."""
    return "L3" if parts > 1 or n_calls > 6 else "L2" if n_calls > 3 else "L1"


_SIMPLE = ("session_setup", "melody_in_key", "four_on_floor_beat")  # need no initial session


_METERED = ("melody_in_key", "four_on_floor_beat")  # generators that take the bar length


def _bar_beats(parts_list: list[Parts]) -> float:
    """Beats per bar after the time signature the parts so far set (4/4 when none)."""
    bar = 4.0
    for p in parts_list:
        for o in p["oracle"]:
            if o.tool == "set_transport" and "time_signature" in o.args:
                num, den = o.args["time_signature"]
                bar = num * 4.0 / den
    return bar


def _name_conflict(parts_list: list[Parts]) -> bool:
    """Same track created twice, or two tempo/time-signature settings."""
    seen: set[str] = set()
    singles: set[str] = set()
    for p in parts_list:
        for o in p["oracle"]:
            if o.tool == "create_track":
                n = o.args["name"].lower()
                if n in seen:
                    return True
                seen.add(n)
            if o.tool == "set_transport":
                for k in o.args:
                    if k in singles:
                        return True
                    singles.add(k)
    return False


def _generate_compositional(seed: int, split: Literal["train", "heldout"]) -> Task:
    """Chain 2-3 simple families into one request. A part that is infeasible (``infeasible``)
    makes the whole task blocked: no changes and ``submit("blocked")``."""
    data = load_family("compositional")
    rng = random.Random(f"compositional:{split}:{seed}")
    combo: list[str] = rng.choice(data["combos"][split])
    for attempt in range(200):
        sub = random.Random(f"compositional:{split}:{seed}:{attempt}")
        picked: list[tuple[str, dict[str, Any], Parts]] = []
        parts_list: list[Parts] = []
        for fam in combo:
            fdata = load_family(fam)
            tpl = sub.choice([t for t in fdata["templates"] if t["split"] == split])
            used = set(PLACEHOLDER.findall(tpl["text"]))
            pool = {k: v[split] for k, v in fdata["slots"].items()}
            bar_beats = _bar_beats(parts_list)
            if fam in _METERED:
                if fam == "four_on_floor_beat" and bar_beats % 1:
                    break  # a quarter-note kick does not divide this bar: draw again
                parts = GENERATORS[fam](sub, pool, used, bar_beats=bar_beats)  # type: ignore[call-arg]
            else:
                parts = GENERATORS[fam](sub, pool, used)
            picked.append((fam, tpl, parts))
            parts_list.append(parts)
        else:
            if not _name_conflict(parts_list):
                break
    else:
        raise RuntimeError(f"no conflict-free composition for seed {seed}")
    intent = " ".join(t["text"].format(**p["vals"]) for _, t, p in picked)
    checks: list[Check] = []
    oracle: list[OracleStep] = []
    touch: list[str] = []
    allowed: list[str] = []
    initial = None
    for _fam, _tpl, p in picked:
        checks += [c for c in p["checks"] if c.kind != "track_count"]  # counts are per family
        oracle += p["oracle"]
        touch += p["touchable"]
        allowed += p.get("allowed_new", [])
        initial = initial or p.get("initial")
    oracle.append(SUBMIT_DONE)
    tags = ["compositional", f"parts{len(combo)}", "L3"]
    task = Task(
        task_id=f"compositional-{split}-{seed}",
        family="compositional",
        intent=intent,
        split=split,
        initial_state=initial,
        checks=checks,
        touchable=touch,
        allowed_new=allowed,
        oracle=oracle,
        music=[m for _f, _t, p in picked for m in p.get("music", [])],
        tags=tags,
    )
    return blocked_form(task) if "infeasible" in combo else task
