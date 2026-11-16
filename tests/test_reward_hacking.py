"""Probe policies that used to score 1.0 for degenerate behaviour. Each must now fail."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from openenv_daw.env.episode import Episode
from openenv_daw.music import CLAP, CLOSED_HAT, KICK, SNARE
from openenv_daw.tasks.generators import generate
from openenv_daw.tasks.model import Task


def run(task: Task, steps: list[dict[str, Any]], profile: str = "portable") -> Episode:
    ep = Episode(task, profile, max_steps=30, max_consecutive_errors=3)
    for s in steps:
        if ep.done:
            break
        ep.run(s["tool"], s["args"])
    if not ep.done:
        ep.finalize(True, "no_submit")
    return ep


def oracle(task: Task) -> list[dict[str, Any]]:
    return [{"tool": o.tool, "args": copy.deepcopy(o.args)} for o in task.oracle]


def first(family: str, pred=lambda t: True) -> Task:
    for seed in range(200):
        t = generate(family, seed, "train")
        if pred(t):
            return t
    raise AssertionError("no task found")


def hat_task() -> Task:
    return first("four_on_floor_beat", lambda t: "closed hats" in t.intent)


@pytest.mark.parametrize("family", ["melody_in_key", "four_on_floor_beat", "reverb_send"])
def test_oracle_still_scores_one(family):
    t = first(family)
    s = run(t, oracle(t)).score
    assert s.reward == 1.0 and s.success and not s.damage


def test_one_note_melody_fails():
    t = first("melody_in_key", lambda t: "4-bar" in t.intent or "2-bar" in t.intent)
    steps = oracle(t)
    # pin every note to the first pitch, whatever the key
    for st in steps:
        notes = st["args"].get("notes")
        if notes:
            p = notes[0]["pitch"]
            for n in notes:
                n["pitch"] = p
    s = run(t, steps).score
    assert s.reward < 0.5 and not s.success
    assert "distinct_pitches_at_least" in s.critical_failed


def test_melody_not_reaching_last_bar_fails():
    t = first("melody_in_key", lambda t: "4-bar" in t.intent)
    steps = oracle(t)
    for st in steps:
        notes = st["args"].get("notes")
        if notes:
            st["args"]["notes"] = notes[:4]
    s = run(t, steps).score
    assert not s.success and "notes_span" in s.critical_failed and s.reward < 0.5


def test_drum_spam_fails():
    t = hat_task()
    steps = oracle(t)
    clip = next(s for s in steps if s["tool"] == "create_midi_clip")["args"]["length_beats"]
    spam = [
        {"pitch": p, "start_beat": i / 4, "duration_beats": 0.125, "velocity": 100}
        for i in range(int(clip * 4))
        for p in (KICK, CLOSED_HAT, SNARE, CLAP)
    ]
    for st in steps:
        if st["tool"] == "insert_midi_notes":
            st["args"]["notes"] = spam
    s = run(t, steps).score
    assert s.reward < 0.5 and not s.success
    assert {"only_pitches", "hits_only_on"} & set(s.critical_failed)


def test_junk_objects_are_penalized():
    t = first("melody_in_key")
    steps = oracle(t)[:-1] + [
        {"tool": "create_track", "args": {"name": "Junk", "kind": "midi"}},
        {"tool": "insert_device", "args": {"track": "Junk", "device": "simple_synth"}},
        {"tool": "insert_device", "args": {"track": "Junk", "device": "reverb"}},
        {"tool": "set_mixer_channel", "args": {"track": "Junk", "solo": True, "volume_db": 6}},
        oracle(t)[-1],
    ]
    s = run(t, steps).score
    # the shared damage cap (0.5) bounds the loss, so a perfect fraction still lands at 0.5
    assert s.reward <= 0.5 and not s.success
    assert len(s.damage) >= 4 and s.damage_penalty == 0.5


def test_each_extraneous_object_costs_015():
    t = first("melody_in_key")
    steps = oracle(t)[:-1] + [
        {"tool": "create_track", "args": {"name": "Extra", "kind": "midi"}},
        oracle(t)[-1],
    ]
    s = run(t, steps).score
    assert s.damage == ["extra track Extra"] and s.reward == pytest.approx(0.85)
    assert not s.success


def test_missing_submit_scores_zero():
    t = first("melody_in_key")
    s = run(t, oracle(t)[:-1]).score
    assert s.reward == 0.0 and not s.success and s.fraction == 1.0 and "no_submit" in s.flags


def test_always_blocked_fails_feasible_tasks():
    t = first("mix_balance")
    s = run(t, [{"tool": "submit", "args": {"status": "blocked"}}]).score
    assert s.reward == 0.0 and not s.success


def test_keyword_refusal_fails_near_miss_tasks():
    for seed in range(8):
        t = generate("near_miss", seed, "train")
        assert t.expected_status == "done"
        s = run(t, [{"tool": "submit", "args": {"status": "blocked"}}]).score
        assert s.reward == 0.0
        assert run(t, oracle(t)).score.success


def test_junk_clip_on_new_track_is_penalized():
    t = first("melody_in_key")
    steps = oracle(t)[:-1] + [
        {
            "tool": "create_midi_clip",
            "args": {"track": _track_of(t), "start_beat": 64, "length_beats": 4},
        },
        oracle(t)[-1],
    ]
    s = run(t, steps).score
    assert s.damage == [f"extra clip on {_track_of(t)}"] and not s.success and s.reward < 1.0


def _track_of(t: Task) -> str:
    return next(o.args["name"] for o in t.oracle if o.tool == "create_track")


def test_template_promoted_check_is_critical():
    t = first("melody_in_key", lambda t: "monophonic" in t.intent or "single-note" in t.intent)
    assert next(c for c in t.checks if c.kind == "notes_monophonic").critical


def test_shaped_reward_is_small_and_opt_in():
    t = first("melody_in_key")
    steps = oracle(t)
    for st in steps:
        notes = st["args"].get("notes")
        if notes:
            p = notes[0]["pitch"]
            for n in notes:
                n["pitch"] = p
    ep = Episode(t, "portable", 30, 3, reward_mode="shaped")
    for s in steps:
        ep.run(s["tool"], s["args"])
    assert 0 < ep.score.reward <= 0.1 and not ep.score.success
    assert run(t, steps).score.reward == 0.0  # strict stays zero
    # a wrong status is never rewarded, even when shaped
    ep2 = Episode(t, "portable", 30, 3, reward_mode="shaped")
    for s in oracle(t)[:-1] + [{"tool": "submit", "args": {"status": "blocked"}}]:
        ep2.run(s["tool"], s["args"])
    assert ep2.score.reward == 0.0


def test_diagnostics_are_reported_not_rewarded():
    t = first("melody_in_key")
    s = run(t, [{"tool": "get_session", "args": {}}, *oracle(t)]).score
    assert s.reward == 1.0 and s.read_first is True and s.calls == len(t.oracle)


@pytest.mark.parametrize(
    "family", ["compositional", "edit_notes", "device_tweak", "disambiguation", "chord_progression"]
)
def test_new_family_degenerate_policies_fail(family):
    for seed in range(10):
        t = generate(family, seed, "train")
        if t.expected_status == "done":
            assert not run(t, []).score.success
            assert run(t, [{"tool": "submit", "args": {"status": "blocked"}}]).score.reward == 0.0
        assert run(t, oracle(t)).score.success


def test_compositional_with_infeasible_part_is_blocked():
    blocked = [generate("compositional", s, "train") for s in range(30)]
    blocked = [t for t in blocked if t.expected_status == "blocked"]
    assert blocked and all(t.oracle[-1].args["status"] == "blocked" for t in blocked)
    assert any(
        t.expected_status == "done" for t in (generate("compositional", s) for s in range(30))
    )


def test_device_tweak_tolerance():
    t = generate("device_tweak", 0, "train")
    steps = oracle(t)
    assert run(t, steps).score.success
    for st in steps:
        if st["tool"] == "set_device_param":
            st["args"]["value"] = min(1.0, st["args"]["value"] + 0.2)
    assert not run(t, steps).score.success


def test_store_sample_by_tag():
    from openenv_daw.tasks.store import default_store

    t = default_store().sample("train", 3, tags=["L3"])
    assert "L3" in t.tags
