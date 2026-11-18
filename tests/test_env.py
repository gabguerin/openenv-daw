import pytest
from openenv.core.env_server.mcp_types import CallToolAction
from openenv_daw.server.daw_environment import DawEnvironment
from openenv_daw.server.tasks.generators import families, generate
from openenv_daw.server.tasks.store import TaskStore, default_store


def call(env, tool, **kw):
    return env.step(CallToolAction(tool_name=tool, arguments=kw))


def run_oracle(env, task):
    obs = env.reset(task_id=task.task_id)
    assert obs.intent == task.intent
    last = None
    for st in task.oracle:
        last = call(env, st.tool, **st.args)
        assert not (st.tool != "submit" and last.done), (task.task_id, st, last.metadata)
    return last


@pytest.mark.parametrize("family", list(families()))
def test_oracle_scores_one_train(family):
    env = DawEnvironment()
    for seed in range(25):
        t = generate(family, seed, "train")
        last = run_oracle(env, t)
        assert last.done and last.reward == 1.0, (t.intent, last.metadata["score"])
        assert last.metadata["success"] is True


def test_oracle_scores_one_heldout():
    env = DawEnvironment()
    store = default_store()
    assert store.heldout
    for raw in store.heldout:
        t = store.get(raw.task_id)  # resolve by id, as the environment does
        last = run_oracle(env, t)
        assert last.reward == 1.0 and last.metadata["success"], (t.intent, last.metadata["score"])


def test_replay_identical_hash():
    t = generate("melody_in_key", 3)
    hashes, rewards = [], []
    for _ in range(2):
        env = DawEnvironment()
        last = run_oracle(env, t)
        hashes.append(last.metadata["state_hash"])
        rewards.append(last.reward)
    assert hashes[0] == hashes[1] and rewards[0] == rewards[1] == 1.0


def test_reset_hides_state_and_requires_get_session():
    env = DawEnvironment()
    obs = env.reset(task_id="reverb_send-train-0")
    assert obs.intent
    assert set(obs.model_dump()) >= {"intent", "task_id"}
    assert "tracks" not in obs.model_dump()


def test_limits_error_streak_and_step_cap():
    env = DawEnvironment()
    env.reset(task_id="session_setup-train-0")
    for _ in range(3):
        last = call(env, "delete_track", track="nope")
    assert last.done and last.metadata["termination"] == "error_limit"
    assert last.metadata["score"]["error_count"] == 3
    with pytest.raises(RuntimeError):
        call(env, "get_session")

    env.reset(task_id="session_setup-train-0")
    for _ in range(30):
        last = call(env, "get_session")
    assert last.done and last.metadata["termination"] == "step_limit" and last.metadata["truncated"]
    assert "no_submit" in last.metadata["score"]["flags"]


def test_blocked_task_and_wrong_status():
    t = generate("infeasible", 1)
    env = DawEnvironment()
    env.reset(task_id=t.task_id)
    last = call(env, "submit", status="done")
    assert last.reward == 0.0 and not last.metadata["success"]
    env.reset(task_id=t.task_id)
    call(env, "create_track", name="X", kind="midi")
    last = call(env, "submit", status="blocked")
    assert last.reward == 0.0  # critical state_unchanged failed


def test_damage_is_a_hard_gate():
    t = generate("reverb_send", 0)
    env = DawEnvironment()
    env.reset(task_id=t.task_id)
    for st in t.oracle[:-1]:
        call(env, st.tool, **st.args)
    other = next(
        x for x in env.backend.session.tracks if x.kind == "midi" and x.name not in t.intent
    )
    call(env, "set_mixer_channel", track=other.id, volume_db=-20)
    last = call(env, "submit", status="done")
    assert last.reward == 0.0 and not last.metadata["success"]
    assert last.metadata["score"]["damage"]


def test_store_ids_and_split():
    s = TaskStore()
    assert s.get("melody_in_key-train-7").split == "train"
    assert s.get_task("heldout", 0).split == "heldout"
    train_intents = {generate(f, i).intent for f in families() for i in range(30)}
    assert not train_intents & {t.intent for t in s.heldout}


async def test_step_async_matches_sync():
    t = generate("melody_in_key", 1)
    env = DawEnvironment()
    env.reset(task_id=t.task_id)
    last = None
    for st in t.oracle:
        last = await env.step_async(CallToolAction(tool_name=st.tool, arguments=st.args))
    assert last.done and last.reward == 1.0


def test_unseeded_reset_varies_and_reports_seed():
    env = DawEnvironment()
    obs = [env.reset() for _ in range(12)]
    assert len({o.task_id for o in obs}) > 1
    seed = obs[0].metadata["seed"]
    again = env.reset(seed=seed)
    assert again.task_id == obs[0].task_id


def test_stratified_heldout_prefix_covers_every_family():
    from openenv_daw.server.tasks.generators import families
    from openenv_daw.server.tasks.store import default_store
    from openenv_daw.trl_env import build_dataset

    rows = build_dataset("heldout", len(families()), stratified=True)
    got = {r["task_id"].split("-", 1)[0] for r in rows}
    assert got == set(families())
    assert len(default_store().heldout) >= 15 * len(families())


def test_heldout_has_at_least_15_tasks_per_family():
    from collections import Counter

    from openenv_daw.server.tasks.store import load_heldout

    counts = Counter(t.family for t in load_heldout())
    assert min(counts.values()) >= 15


def test_degenerate_policies_stay_under_the_reward_floor():
    """noop, submit-only and always-blocked score at most ``validation.reward.floor_margin``."""
    from openenv_daw.server.episode import Episode

    floor = 0.1  # openenv.yaml validation.reward.floor_margin
    feasible = [t for t in default_store().heldout if t.expected_status == "done"]
    assert feasible
    for steps in ([], [("submit", {"status": "done"})], [("submit", {"status": "blocked"})]):
        for t in feasible:
            ep = Episode(t)
            for tool, args in steps:
                ep.run(tool, args)
            if not ep.done:
                ep.finalize(True, "no_submit")
            assert ep.score is not None and ep.score.reward <= floor, (t.task_id, steps)
