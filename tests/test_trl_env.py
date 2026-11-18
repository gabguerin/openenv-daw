import json

import pytest

pytest.importorskip("transformers")
from openenv_daw.server.tools import build_server  # noqa: E402
from openenv_daw.trl_env import DawToolEnv, build_dataset  # noqa: E402
from transformers.utils import get_json_schema  # noqa: E402


def _tool_methods():
    return [
        n
        for n, f in vars(DawToolEnv).items()
        if callable(f) and not n.startswith("_") and n not in ("reset", "get_reward")
    ]


def test_methods_match_registry_and_have_schemas():
    from openenv.core.env_server.mcp_environment import get_server_tools
    from openenv_daw.server.backends import SimBackend

    reg = set(get_server_tools(build_server(lambda: SimBackend(), on_submit=lambda s, r: {})))
    assert set(_tool_methods()) == reg and "submit" in reg
    env = DawToolEnv()
    for n in reg:
        schema = get_json_schema(getattr(env, n))
        assert schema["function"]["name"] == n and schema["function"]["description"]


def test_oracle_through_wrapper_scores_one():
    rows = build_dataset("train", 12)
    for row in rows:
        env = DawToolEnv()
        env.reset(task_id=row["task_id"])
        for st in env.episode.task.oracle:
            out = json.loads(getattr(env, st.tool)(**st.args))
            assert out.get("ok") is True or st.tool == "submit", out
        assert env.get_reward() == 1.0
        with pytest.raises(ValueError):
            env.get_session()


def test_no_submit_is_truncated_and_scored():
    env = DawToolEnv()
    env.reset(task_id=build_dataset("train", 1)[0]["task_id"])
    assert env.get_reward() <= 0.0 + 1.0  # scored without raising
    assert env.episode.score.truncated
