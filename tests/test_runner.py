import json

from openenv.core.harness import ModelStepResult
from openenv.core.llm_client import LLMResponse, ToolCall

from openenv_daw.harness import (
    DawSessionFactory,
    collect,
    flat,
    read_results,
    scripted_model_step,
    select_tasks,
)
from openenv_daw.runner.sft_export import export_sft
from openenv_daw.tasks.store import default_store
from openenv_daw.tools import load_instructions


def broken(messages, tools, sampling):
    """First call is bogus, then the model stops."""
    if any(m["role"] == "tool" for m in messages):
        return ModelStepResult(LLMResponse("done"))
    return ModelStepResult(LLMResponse("", [ToolCall("c", "create_track", {"bogus": 1})]))


def test_collect_record_resume_and_export(tmp_path):
    tasks = select_tasks("train", 4)
    factory = DawSessionFactory()
    oracle = scripted_model_step("oracle", tasks)
    recs = collect(factory, oracle, tasks, tmp_path / "ok")
    assert len(recs) == 4
    for r in recs:
        assert r["reward"] == 1.0 and r["done"] and r["verify_metrics"]["success"]
        roles = [m["role"] for m in r["messages"]]
        assert roles[:2] == ["system", "user"] and "tool" in roles
        assert r["verify_metrics"]["step_log"] and r["task"] == flat(r)["task_id"]
    # resume skips collected episodes; a duplicate sample and a failing episode are added
    assert collect(factory, oracle, tasks, tmp_path / "ok") == []
    collect(factory, oracle, tasks[:1], tmp_path / "dup", prefix="dup")
    collect(factory, broken, tasks[:1], tmp_path / "bad", prefix="bad")
    assert not read_results(tmp_path / "bad")[0]["verify_metrics"]["success"]
    assert (tmp_path / "ok" / "metadata.json").is_file()
    stats = export_sft(
        [tmp_path / "ok", tmp_path / "dup", tmp_path / "bad"], tmp_path / "sft.jsonl"
    )
    assert stats == {"episodes": 6, "kept": 4, "dropped": 2, "unmusical": 0}
    rows = [json.loads(line) for line in (tmp_path / "sft.jsonl").read_text().splitlines()]
    assert set(rows[0]) == {"messages", "tools", "reward", "task_id"}
    assert rows[0]["reward"] == 1.0 and json.loads(rows[0]["tools"])[0]["type"] == "function"


def test_export_drops_unmusical(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    base = {"messages": [], "reward": 1.0, "extra": {}}
    rows = [
        {
            **base,
            "task": "a",
            "verify_metrics": {"success": True, "actions": [], "musicality": 0.9},
        },
        {
            **base,
            "task": "c",
            "verify_metrics": {"success": True, "actions": [], "musicality": 0.3},
        },
        {
            **base,
            "task": "d",
            "verify_metrics": {"success": True, "actions": [], "musicality": None},
        },
        {**base, "task": "e", "verify_metrics": {"success": False, "actions": []}},
    ]
    (d / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    stats = export_sft(d, tmp_path / "sft.jsonl")
    assert stats["kept"] == 2 and stats["unmusical"] == 1


def test_select_tasks_heldout_and_family():
    h = select_tasks("heldout", 5, family="melody_in_key")
    assert len(h) == 5 and all(t.family == "melody_in_key" and t.split == "heldout" for t in h)
    assert select_tasks("train", 3)[0].task_id == default_store().sample("train", 0).task_id


def test_instructions_follow_server_profile(tmp_path):
    tasks = select_tasks("train", 1)
    recs = collect(
        DawSessionFactory(profile="mixer_only"),
        scripted_model_step("oracle", tasks),
        tasks,
        tmp_path,
    )
    assert recs[0]["messages"][0]["content"] == load_instructions("mixer_only")
