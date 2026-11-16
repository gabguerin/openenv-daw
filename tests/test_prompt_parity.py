"""SFT data, rollouts and GRPO render the same prompt."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("transformers")

from openenv_daw.runner.report import ScriptedPolicy
from openenv_daw.runner.rollout import InProcessAdapter, prompt_tools, run_episode
from openenv_daw.runner.sft_export import export_sft
from openenv_daw.tasks.store import default_store
from openenv_daw.trl_env import DawToolEnv, tool_schemas


async def test_runner_tools_equal_trl_tools():
    adapter = InProcessAdapter(default_store())
    listed = await adapter.tools()
    assert [t["function"]["name"] for t in prompt_tools(listed)] == [
        t["function"]["name"] for t in tool_schemas()
    ]
    assert prompt_tools(listed) == list(tool_schemas())
    names = {n for n in dir(DawToolEnv) if not n.startswith("_")} - {"reset", "get_reward"}
    assert names == {t["function"]["name"] for t in tool_schemas()}
    await adapter.close()


async def test_tool_results_serialize_like_trl(tmp_path: Path):
    task = default_store().get_task("train", 0)
    rec = await run_episode(
        ScriptedPolicy("oracle", [task]), InProcessAdapter(default_store()), task.task_id
    )
    env = DawToolEnv()
    env.reset(task_id=task.task_id)
    for o in task.oracle[:2]:
        trl_out = getattr(env, o.tool)(**o.args)
        tool_msgs = [m["content"] for m in rec["messages"] if m["role"] == "tool"]
        assert trl_out in tool_msgs
    assert rec["nudges"] == 0


def test_export_drops_nudged_and_unmusical(tmp_path: Path):
    base = {"success": True, "actions": [], "messages": [], "tools": []}
    rows = [
        {**base, "task_id": "a", "nudges": 0, "musicality": 0.9},
        {**base, "task_id": "b", "nudges": 1, "musicality": 0.9},
        {**base, "task_id": "c", "nudges": 0, "musicality": 0.3},
        {**base, "task_id": "d", "nudges": 0, "musicality": None},
    ]
    src = tmp_path / "r.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows))
    stats = export_sft(src, tmp_path / "sft.jsonl")
    assert stats["kept"] == 2 and stats["nudged"] == 1 and stats["unmusical"] == 1
