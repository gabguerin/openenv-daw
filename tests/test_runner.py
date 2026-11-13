import json

from openenv_daw.runner.rollout import InProcessAdapter, run_episode, select_tasks
from openenv_daw.runner.sft_export import export_sft
from openenv_daw.tasks.generators import generate
from openenv_daw.tasks.model import Task


class OraclePolicy:
    """Fake LLM: replays a task's oracle as tool calls, one per assistant message."""

    name = "oracle"

    def __init__(self, tasks: dict[str, Task], break_at: int | None = None):
        self.tasks, self.break_at = tasks, break_at
        self.i = 0
        self.task: Task | None = None

    async def complete(self, messages, tools):
        intent = messages[1]["content"]
        if self.task is None or self.task.intent != intent or len(messages) == 2:
            self.task = next(t for t in self.tasks.values() if t.intent == intent)
            self.i = 0
        if self.i >= len(self.task.oracle):
            return {"role": "assistant", "content": "done"}
        st = self.task.oracle[self.i]
        args = dict(st.args)
        if self.break_at is not None and self.i == self.break_at:
            args = {"bogus": 1}
        self.i += 1
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{self.i}",
                    "type": "function",
                    "function": {"name": st.tool, "arguments": json.dumps(args)},
                }
            ],
        }


async def test_episode_record_and_export(tmp_path):
    tasks = {t.task_id: t for t in select_tasks("train", 4)}
    env = InProcessAdapter()
    recs = [await run_episode(OraclePolicy(tasks), env, tid) for tid in tasks]
    recs.append(await run_episode(OraclePolicy(tasks), env, next(iter(tasks))))  # duplicate
    # a failing episode: first oracle call replaced by a bogus one
    first = next(iter(tasks))
    recs.append(await run_episode(OraclePolicy(tasks, break_at=0), env, first))
    for r in recs[:4]:
        assert r["success"] and r["reward"] == 1.0 and r["done"]
        roles = [m["role"] for m in r["messages"]]
        assert roles[:2] == ["system", "user"] and "tool" in roles
        assert r["tools"][0]["type"] == "function" and r["step_log"]
    assert not recs[-1]["success"]
    src = tmp_path / "r.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    stats = export_sft(src, tmp_path / "sft.jsonl")
    assert stats == {"episodes": 6, "kept": 4, "dropped": 2}
    rows = [json.loads(line) for line in (tmp_path / "sft.jsonl").read_text().splitlines()]
    assert set(rows[0]) == {"messages", "tools"}


def test_select_tasks_heldout_and_family():
    h = select_tasks("heldout", 5, family="melody_in_key")
    assert len(h) == 5 and all(t.family == "melody_in_key" and t.split == "heldout" for t in h)
    assert select_tasks("train", 3)[0].task_id == generate("session_setup", 0).task_id
