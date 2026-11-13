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
from openenv_daw.schema import Session, Track
from openenv_daw.schema.session import ForeignClip, ForeignDevice, ForeignParam
from openenv_daw.sim.catalog import load_catalog
from openenv_daw.tasks.model import Check, OracleStep, Task
from openenv_daw.tasks.resolve import resolve_for_profile

PLACEHOLDER = re.compile(r"{(\w+)}")
Parts = dict[str, Any]


@cache
def load_family(family: str) -> dict[str, Any]:
    text = resources.files("openenv_daw.tasks.templates").joinpath(f"{family}.yaml").read_text()
    data = yaml.safe_load(text)
    return data


def families() -> dict[str, float]:
    names = [
        "session_setup",
        "melody_in_key",
        "four_on_floor_beat",
        "reverb_send",
        "mix_balance",
        "near_miss",
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
    return {
        "vals": vals,
        "checks": checks,
        "oracle": oracle,
        "touchable": touch,
        "allowed_new": list(names) if "names" in used else [],
    }


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
        Check(kind="distinct_pitches_at_least", args={"track": track, "n": 3}, critical=True),
        Check(kind="notes_span", args={"track": track, "bar_beats": 4.0}, critical=True),
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
        "allowed_new": [track, "device:simple_synth"],
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
            {"pitch": CLOSED_HAT, "start_beat": b, "duration_beats": 0.25, "velocity": 80}
            for b in hats
        ]
        checks.append(
            Check(kind="hits_on_beats", args={"track": track, "pitch": CLOSED_HAT, "beats": hats})
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
        "allowed_new": [track, "device:drum_kit"],
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
        "allowed_new": [ret, "device:reverb"],
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
    # checks/oracle are the blocked form; ``requires`` (capability tags from the template) makes
    # resolve_for_profile keep them
    return {
        "vals": vals,
        "checks": [Check(kind="state_unchanged", critical=True)],
        "oracle": [],
        "initial": initial,
        "touchable": [],
    }


GENERATORS: dict[str, Callable[[random.Random, dict[str, list[Any]], set[str]], Parts]] = {
    "session_setup": gen_session_setup,
    "melody_in_key": gen_melody_in_key,
    "four_on_floor_beat": gen_four_on_floor,
    "reverb_send": gen_reverb_send,
    "mix_balance": gen_mix_balance,
    "near_miss": gen_near_miss,
    "infeasible": gen_infeasible,
}


FOREIGN_PROB = 0.2
_FOREIGN_DEVICES = [
    ("Serum", [("Cutoff", "1.2 kHz"), ("Wavetable", "Basic Shapes")]),
    ("Kontakt", [("Volume", "-3.0 dB")]),
    ("Pigments", [("Macro 1", "42 %"), ("Macro 2", "10 %")]),
]


def inject_foreign(initial: dict[str, Any] | None, rng: random.Random) -> dict[str, Any]:
    """Add an untouchable read-only group track (foreign plugin and audio/launcher clips)."""
    s = Session.restore(initial) if initial else Session()
    name, params = rng.choice(_FOREIGN_DEVICES)
    s.tracks.append(
        Track(
            id=s.next_id("trk"),
            name="Legacy Bus",
            kind="group",
            foreign_devices=[
                ForeignDevice(
                    id=s.next_id("fdv"),
                    name=name,
                    params=[ForeignParam(name=n, display=d) for n, d in params],
                )
            ],
            foreign_clips=[
                ForeignClip(
                    id=s.next_id("fcl"), name="Vocal Take", kind="audio", length_beats=16.0
                ),
                ForeignClip(
                    id=s.next_id("fcl"), name="Scene Loop", kind="launcher", length_beats=8.0
                ),
            ],
        )
    )
    return s.snapshot()


def generate(family: str, seed: int, split: Literal["train", "heldout"] = "train") -> Task:
    """Deterministically build a task from (family, split, seed). The result is
    profile-neutral: ``requires`` is what ``resolve_for_profile`` checks. It is stored in its
    form for the default (portable) profile."""
    data = load_family(family)
    rng = random.Random(f"{family}:{split}:{seed}")
    templates = [t for t in data["templates"] if t["split"] == split]
    tpl = rng.choice(templates)
    used = set(PLACEHOLDER.findall(tpl["text"]))
    pool = {k: v[split] for k, v in data["slots"].items()}
    parts = GENERATORS[family](rng, pool, used)
    intent = tpl["text"].format(**parts["vals"])
    oracle = parts["oracle"] + [SUBMIT_DONE]
    tools = sorted({o.tool for o in oracle if o.tool != "submit"})
    requires = tools + [c for c in tpl.get("requires", []) if c not in tools]
    for o in oracle:
        if o.tool == "set_device_param" and o.args.get("device") in load_catalog():
            key = f"param:{o.args['device']}.{str(o.args['param']).lower()}"
            if key not in requires:
                requires.append(key)
    initial = parts.get("initial")
    tags: list[str] = []
    if random.Random(f"foreign:{family}:{split}:{seed}").random() < FOREIGN_PROB:
        initial = inject_foreign(initial, random.Random(f"foreign-pick:{family}:{split}:{seed}"))
        tags.append("foreign")
    task = Task(
        task_id=f"{family}-{split}-{seed}",
        family=family,
        intent=intent,
        split=split,
        initial_state=initial,
        checks=parts["checks"],
        touchable=parts["touchable"],
        allowed_new=parts.get("allowed_new", []),
        requires=requires,
        oracle=oracle,
        tags=tags,
    )
    return resolve_for_profile(task)
