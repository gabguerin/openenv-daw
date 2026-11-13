import pytest
from openenv.core.env_server.mcp_types import CallToolAction

from openenv_daw.env import DawEnvironment
from openenv_daw.tasks.generators import families, generate
from openenv_daw.tasks.store import TaskStore, default_store


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
    for t in store.heldout:
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


def test_damage_penalty():
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
    assert 0.8 < last.reward < 1.0 and not last.metadata["success"]
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
