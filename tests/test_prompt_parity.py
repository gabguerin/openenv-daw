"""SFT data, rollouts and GRPO render the same prompt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from openenv_daw.harness import DawSessionFactory, collect, scripted_model_step
from openenv_daw.runner.sft_export import export_sft
from openenv_daw.tasks.store import default_store
from openenv_daw.trl_env import DawToolEnv, tool_schemas


def test_session_tools_equal_trl_tools():
    session = DawSessionFactory(default_store()).create(default_store().get_task("train", 0))
    try:
        tools = session.list_tools()
    finally:
        session.close()
    expected = [s["function"] for s in tool_schemas()]
    assert [t.name for t in tools] == [f["name"] for f in expected]
    assert [t.input_schema for t in tools] == [f["parameters"] for f in expected]
    names = {n for n in dir(DawToolEnv) if not n.startswith("_")} - {"reset", "get_reward"}
    assert names == {f["name"] for f in expected}


def test_tool_results_serialize_like_trl(tmp_path: Path):
    task = default_store().get_task("train", 0)
    rec = collect(
        DawSessionFactory(default_store()), scripted_model_step("oracle", [task]), [task], tmp_path
    )[0]
    env = DawToolEnv()
    env.reset(task_id=task.task_id)
    tool_msgs = [m["content"] for m in rec["messages"] if m["role"] == "tool"]
    for o in task.oracle[:2]:
        assert getattr(env, o.tool)(**o.args) in tool_msgs


def test_export_tools_are_the_trl_schemas(tmp_path: Path):
    task = default_store().get_task("train", 0)
    collect(
        DawSessionFactory(default_store()),
        scripted_model_step("oracle", [task]),
        [task],
        tmp_path / "r",
    )
    export_sft(tmp_path / "r", tmp_path / "sft.jsonl")
    row = json.loads((tmp_path / "sft.jsonl").read_text().splitlines()[0])
    assert json.loads(row["tools"]) == list(tool_schemas())


@pytest.mark.parametrize("profile", ["portable", "portable_plugins", "mixer_only"])
def test_client_sends_the_trl_schemas(profile):
    from openenv_daw.harness import to_openai_tools
    from openenv_daw.trl_env import tool_schemas

    mcp = [
        {"name": s["function"]["name"], "description": s["function"]["description"],
         "inputSchema": s["function"]["parameters"]}
        for s in tool_schemas(profile)
    ]  # fmt: skip
    assert to_openai_tools(mcp) == list(tool_schemas(profile))
