"""remove_device and clip_edit: the oracles score 1.0 and the shortcuts a model might take do not."""

from __future__ import annotations

import pytest
from openenv_daw.server.episode import Episode
from openenv_daw.server.tasks.generators import generate


def run(task, steps):
    ep = Episode(task, 60, 5)
    for tool, args in steps:
        ep.run(tool, args)
    ep.run("submit", {"status": task.expected_status})
    assert ep.score is not None
    return ep.score


def first(family, pred, split="train"):
    for seed in range(200):
        t = generate(family, seed, split)
        if pred(t):
            return t
    raise AssertionError("no task")


@pytest.mark.parametrize("split", ["train", "heldout"])
@pytest.mark.parametrize("family", ["remove_device", "clip_edit"])
def test_oracles_pass(family, split):
    for seed in range(40):
        t = generate(family, seed, split)
        steps = [(o.tool, o.args) for o in t.oracle if o.tool != "submit"]
        assert run(t, steps).reward == 1.0, t.task_id


def test_remove_device_blocked_share():
    tasks = [generate("remove_device", s, "train") for s in range(120)]
    blocked = sum(t.expected_status == "blocked" for t in tasks)
    assert 5 <= blocked <= 40


def test_wrong_device_removed_fails():
    t = first("remove_device", lambda t: t.expected_status == "done" and len(t.oracle) == 2)
    step = t.oracle[0]
    other = next(
        d["device"]
        for tr in t.initial_state["tracks"]
        if tr["name"] == step.args["track"]
        for d in tr["devices"]
        if d["device"] not in (step.args["device"], "simple_synth")
    )
    assert run(t, [("delete_device", {**step.args, "device": other})]).reward < 1.0


def test_clip_recreated_fails():
    t = first("clip_edit", lambda t: "name" in t.oracle[0].args)
    track = t.oracle[0].args
    name = track["name"]
    tr = t.initial_state["tracks"][0]["name"]
    steps = [
        ("delete_clip", {"clip": "Idea"}),
        ("create_midi_clip", {"track": tr, "start_beat": 0, "length_beats": 16, "name": name}),
    ]
    assert run(t, steps).reward < 1.0


def test_shorten_without_dropping_notes_is_an_error():
    t = first("clip_edit", lambda t: t.oracle[0].tool == "remove_midi_notes")
    ep = Episode(t, 60, 5)
    r = ep.run("update_midi_clip", {"clip": "Idea", "length_beats": 4})
    assert r["ok"] is False
