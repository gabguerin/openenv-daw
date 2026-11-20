"""The rubric tree is the single source of the reward: server, score and components agree."""

from __future__ import annotations

import copy

import pytest
from openenv.core.env_server.mcp_types import CallToolAction
from openenv_daw.server.daw_environment import DawEnvironment
from openenv_daw.server.episode import Episode
from openenv_daw.server.tasks.generators import families, generate


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


@pytest.mark.parametrize("family", list(families()))
def test_server_reward_matches_score(family):
    env = DawEnvironment()
    for seed in range(4):
        t = generate(family, seed, "train")
        for steps in (t.oracle, collapse_pitches(t), t.oracle[-1:]):
            env.reset(task_id=t.task_id)
            last = play(env, steps)
            assert last.done
            assert last.reward == pytest.approx(last.metadata["score"]["reward"])


def test_every_leaf_has_a_score_after_a_terminal_step():
    t = generate("melody_in_key", 0, "train")
    env = DawEnvironment()
    env.reset(task_id=t.task_id)
    play(env, t.oracle)
    named = dict(env.rubric.named_rubrics())
    assert {"status", "safety", "critical", "quality", "quality.correctness.optional"} <= set(named)
    assert all(r.last_score is not None for r in named.values())


def test_penalties_multiply_correctness(monkeypatch):
    monkeypatch.setattr("openenv_daw.server.rewards.score.MUSIC_WEIGHT", 0.0)
    t = generate("melody_in_key", 0, "train")
    ep = Episode(t, 30, 3)
    for o in t.oracle[:-1]:
        ep.run(o.tool, o.args)
    ep.run("create_track", {"name": "Extra", "kind": "midi"})
    ep.run("submit", {"status": "done"})
    s = ep.score
    assert s.reward == pytest.approx(s.fraction * 0.85)
    assert s.components["quality.correctness.extraneous"] == pytest.approx(0.85)


# --- the shaped training reward ---------------------------------------------------------------


def shaped_run(task, steps, status=None):
    from openenv_daw.server.rewards import DawRubric

    ep = Episode(task, 60, 5, rubric=DawRubric(shaped=True))
    for st in steps:
        ep.run(st.tool, st.args)
    if status:
        ep.run("submit", {"status": status})
    if not ep.done:
        ep.finalize(True, "no_submit")
    return ep.score


@pytest.mark.parametrize("family", list(families()))
def test_shaped_reward_gives_nothing_for_doing_nothing(family):
    for seed in range(8):
        t = generate(family, seed, "train")
        if t.expected_status != "done":
            continue
        for status in (None, "done", "blocked"):
            assert shaped_run(t, [], status).reward == 0.0, (t.task_id, status)


@pytest.mark.parametrize("family", list(families()))
def test_shaped_reward_pays_oracles_above_any_partial(family):
    for seed in range(6):
        t = generate(family, seed, "train")
        full = shaped_run(t, t.oracle[:-1], t.oracle[-1].args["status"])
        assert full.success and 0.8 <= full.reward <= 1.0, t.task_id
        if len(t.oracle) > 2:  # drop the last action: a partial solution
            part = shaped_run(t, t.oracle[:-2], t.oracle[-1].args["status"])
            if not part.success:  # (the last step can be an optional extra)
                assert part.reward < full.reward and part.reward <= 0.5, t.task_id


def test_shaped_reward_grades_a_chord_progression():
    t = generate("chord_progression", 0, "train")
    steps = [copy.deepcopy(o) for o in t.oracle[:-1]]
    notes = next(s for s in steps if s.tool == "insert_midi_notes").args["notes"]
    bars = max(n["start_beat"] for n in notes) // 4 + 1
    rewards = []
    for keep in range(int(bars) + 1):  # keep the notes of the first `keep` bars
        for s in steps:
            if s.tool == "insert_midi_notes":
                s.args["notes"] = [n for n in notes if n["start_beat"] < 4 * keep] or notes[:1]
        rewards.append(shaped_run(t, steps, "done").reward)
    assert rewards == sorted(rewards) and rewards[-1] >= 0.8 and rewards[-2] <= 0.5
    assert len(set(rewards[1:])) > 1  # graded, not a cliff
    assert rewards[0] <= 0.5


def test_strict_reward_is_unchanged_by_the_shaped_option():
    t = generate("chord_progression", 0, "train")
    steps = [copy.deepcopy(o) for o in t.oracle[:-1]]
    for s in steps:
        if s.tool == "insert_midi_notes":
            s.args["notes"] = s.args["notes"][:3]
    ep = Episode(t, 60, 5)
    for s in steps:
        ep.run(s.tool, s.args)
    ep.run("submit", {"status": "done"})
    assert ep.score.reward == 0.0
    assert shaped_run(t, steps, "done").reward > 0.0


def test_shaped_music_saturates_later():
    from openenv_daw.server.rewards import DawRubric

    facts = {"musicality": 0.85}
    from openenv_daw.server.rewards.score import Musicality, MusicBlend

    strict, shaped = Musicality(), Musicality(0.95)
    assert strict.score(facts) == 1.0 and shaped.score(facts) == pytest.approx(0.85 / 0.95)
    assert MusicBlend(0.95).validation_config()["music_full"] == 0.95
    assert DawRubric(shaped=True).quality.music.music_full == 0.95


def test_shaped_reward_has_no_cliff_on_a_single_optional_check():
    """A melody with too few notes fails one optional check and some critical ones: the shaped
    reward stays positive and never exceeds the all-critical-checks-pass reward."""
    t = next(
        t
        for t in (generate("melody_in_key", s, "train") for s in range(60))
        if sum(not c.critical for c in t.checks) == 1
    )
    full = shaped_run(t, t.oracle[:-1], "done").reward
    steps = [copy.deepcopy(o) for o in t.oracle[:-1]]
    for s in steps:
        if s.tool == "insert_midi_notes":
            s.args["notes"] = s.args["notes"][:5]
    part = shaped_run(t, steps, "done")
    assert 0.0 < part.reward <= 0.5 < full


def test_shaped_reward_passing_criticals_never_pays_less_than_failing_one():
    t = generate("melody_in_key", 0, "train")
    ok_steps = [copy.deepcopy(o) for o in t.oracle[:-1]]
    for s in ok_steps:  # all critical checks pass, the optional note count fails
        if s.tool == "insert_midi_notes":
            s.args["notes"] = s.args["notes"][:]
    scores = []
    for keep in (len(ok_steps[-1].args["notes"]), 5, 2):
        steps = copy.deepcopy(ok_steps)
        steps[-1].args["notes"] = steps[-1].args["notes"][:keep]
        scores.append(shaped_run(t, steps, "done").reward)
    assert scores == sorted(scores, reverse=True)


def test_shaped_progress_leaf_is_scored_even_when_a_gate_fails():
    from openenv_daw.server.rewards import DawRubric

    t = generate("mix_balance", 0, "train")
    ep = Episode(t, 60, 5, rubric=DawRubric(shaped=True))
    ep.run("submit", {"status": "blocked"})  # wrong status: the gate zeroes the reward
    assert ep.score.reward == 0.0
    assert ep.score.components["progress"] is not None
