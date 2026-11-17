"""The rubric tree is the single source of the reward: server, score and components agree."""

from __future__ import annotations

import copy

import pytest
from openenv.core.env_server.mcp_types import CallToolAction

from openenv_daw.env import DawEnvironment
from openenv_daw.env.episode import Episode
from openenv_daw.tasks.generators import families, generate


def play(env, steps):
    last = None
    for st in steps:
        last = env.step(CallToolAction(tool_name=st.tool, arguments=st.args))
    return last


def collapse_pitches(task):
    steps = [copy.deepcopy(o) for o in task.oracle]
    for s in steps:
        notes = s.args.get("notes")
        if notes:
            for n in notes:
                n["pitch"] = notes[0]["pitch"]
    return steps


@pytest.mark.parametrize("mode", ["strict", "shaped"])
@pytest.mark.parametrize("family", list(families()))
def test_server_reward_matches_score(monkeypatch, mode, family):
    monkeypatch.setenv("OPENENV_DAW_REWARD_MODE", mode)
    env = DawEnvironment()
    for seed in range(4):
        t = generate(family, seed, "train")
        for steps in (t.oracle, collapse_pitches(t), t.oracle[-1:]):
            env.reset(task_id=t.task_id)
            last = play(env, steps)
            assert last.done
            assert last.reward == pytest.approx(last.metadata["score"]["reward"])


def test_shaped_mode_is_visible_to_the_server(monkeypatch):
    monkeypatch.setenv("OPENENV_DAW_REWARD_MODE", "shaped")
    t = generate("melody_in_key", 1, "train")
    env = DawEnvironment()
    env.reset(task_id=t.task_id)
    last = play(env, collapse_pitches(t))
    assert 0 < last.reward <= 0.1


def test_every_leaf_has_a_score_after_a_terminal_step():
    t = generate("melody_in_key", 0, "train")
    env = DawEnvironment()
    env.reset(task_id=t.task_id)
    play(env, t.oracle)
    named = dict(env.rubric.named_rubrics())
    assert {"status", "safety", "critical", "quality", "quality.correctness.optional"} <= set(named)
    assert all(r.last_score is not None for r in named.values())


def test_penalties_multiply_correctness(monkeypatch):
    monkeypatch.setenv("OPENENV_DAW_MUSIC_WEIGHT", "0")
    t = generate("melody_in_key", 0, "train")
    ep = Episode(t, "portable", 30, 3)
    for o in t.oracle[:-1]:
        ep.run(o.tool, o.args)
    ep.run("create_track", {"name": "Extra", "kind": "midi"})
    ep.run("submit", {"status": "done"})
    s = ep.score
    assert s.reward == pytest.approx(s.fraction * 0.85)
    assert s.components["quality.correctness.extraneous"] == pytest.approx(0.85)
