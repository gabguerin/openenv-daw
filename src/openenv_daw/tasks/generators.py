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

from openenv_daw.backends import SimBackend
from openenv_daw.music import CLOSED_HAT, KICK, scale_pitches
from openenv_daw.tasks.model import Check, OracleStep, Task

PLACEHOLDER = re.compile(r"{(\w+)}")
Parts = dict[str, Any]


@cache
def load_family(family: str) -> dict[str, Any]:
    text = resources.files("openenv_daw.tasks.templates").joinpath(f"{family}.yaml").read_text()
    data = yaml.safe_load(text)
    extra = resources.files("openenv_daw.tasks.templates").joinpath(f"paraphrased/{family}.yaml")
    if extra.is_file():  # offline paraphrases of train templates (scripts/paraphrase_templates.py)
        data["templates"] = data["templates"] + yaml.safe_load(extra.read_text())["templates"]
    return data


def families() -> dict[str, float]:
    names = [
        "session_setup",
        "melody_in_key",
        "four_on_floor_beat",
        "reverb_send",
        "mix_balance",
        "infeasible",
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
SUBMIT_BLOCKED = _step("submit", status="blocked", reason="not possible with the available tools")


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
    return {"vals": vals, "checks": checks, "oracle": oracle, "touchable": touch}


def gen_melody_in_key(rng: random.Random, pool: dict[str, list[Any]], used: set[str]) -> Parts:
    key, bars, track = rng.choice(pool["key"]), rng.choice(pool["bars"]), rng.choice(pool["track"])
    length = bars * 4
    sp = scale_pitches(key, 60, 83)
    notes = [
        {
            "pitch": sp[(i * 2) % len(sp)],
            "start_beat": float(i),
            "duration_beats": 1.0,
            "velocity": 90,
        }
        for i in range(length)
    ]
    checks = [
        Check(kind="track_exists", args={"track": track, "kind": "midi"}, critical=True),
        Check(kind="has_device", args={"track": track, "device": "simple_synth"}, critical=True),
        Check(kind="notes_in_key", args={"track": track, "key": key}, critical=True),
        Check(
            kind="clip_at", args={"track": track, "start_beat": 0.0, "length_beats": float(length)}
        ),
        Check(kind="note_count_at_least", args={"track": track, "n": max(2, length // 2)}),
        Check(kind="notes_monophonic", args={"track": track}),
        Check(kind="notes_pitch_range", args={"track": track, "min": 48, "max": 84}),
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
    }


def gen_four_on_floor(rng: random.Random, pool: dict[str, list[Any]], used: set[str]) -> Parts:
    bars, track = rng.choice(pool["bars"]), rng.choice(pool["track"])
    clause = rng.choice(pool["hat_clause"]) if "hat_clause" in used else ""
    length = bars * 4
    kicks = [float(b) for b in range(length)]
    hats = [i * 0.5 for i in range(length * 2)]
    notes = [
        {"pitch": KICK, "start_beat": b, "duration_beats": 0.25, "velocity": 110} for b in kicks
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
            kind="no_notes_of_pitch_off_beats", args={"track": track, "pitch": KICK, "beats": kicks}
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
            {"pitch": CLOSED_HAT, "start_beat": b, "duration_beats": 0.25, "velocity": 80}
            for b in hats
        ]
        checks.append(
            Check(kind="hits_on_beats", args={"track": track, "pitch": CLOSED_HAT, "beats": hats})
        )
    oracle.append(_step("insert_midi_notes", clip="Beat", notes=notes))
    touch: list[str] = []
    if "bpm" in used:
        bpm = rng.choice(pool["bpm"])
        vals["bpm"] = str(bpm)
        checks.append(Check(kind="tempo_is", args={"bpm": bpm}, critical=True))
        oracle.append(_step("set_transport", tempo_bpm=bpm))
        touch.append("transport")
    return {"vals": vals, "checks": checks, "oracle": oracle, "touchable": touch}


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
        "oracle": [],
        "initial": initial,
        "touchable": [],
        "blocked": True,
    }


GENERATORS: dict[str, Callable[[random.Random, dict[str, list[Any]], set[str]], Parts]] = {
    "session_setup": gen_session_setup,
    "melody_in_key": gen_melody_in_key,
    "four_on_floor_beat": gen_four_on_floor,
    "reverb_send": gen_reverb_send,
    "mix_balance": gen_mix_balance,
    "infeasible": gen_infeasible,
}


def generate(family: str, seed: int, split: Literal["train", "heldout"] = "train") -> Task:
    """Deterministically build a task from (family, split, seed)."""
    data = load_family(family)
    rng = random.Random(f"{family}:{split}:{seed}")
    templates = [t for t in data["templates"] if t["split"] == split]
    tpl = rng.choice(templates)
    used = set(PLACEHOLDER.findall(tpl["text"]))
    pool = {k: v[split] for k, v in data["slots"].items()}
    parts = GENERATORS[family](rng, pool, used)
    intent = tpl["text"].format(**parts["vals"])
    blocked = parts.get("blocked", False)
    oracle = parts["oracle"] + [SUBMIT_BLOCKED if blocked else SUBMIT_DONE]
    return Task(
        task_id=f"{family}-{split}-{seed}",
        family=family,
        intent=intent,
        split=split,
        initial_state=parts.get("initial"),
        checks=parts["checks"],
        touchable=parts["touchable"],
        expected_status="blocked" if blocked else "done",
        oracle=oracle,
        tags=["blocked"] if blocked else [],
    )
