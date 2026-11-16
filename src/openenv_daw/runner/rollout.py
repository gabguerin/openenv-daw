"""Run a policy against the environment and record chat trajectories (messages + tools)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from openenv_daw.backends import Backend
from openenv_daw.env.client import DawEnv
from openenv_daw.env.environment import DawEnvironment, result_data
from openenv_daw.sim.profiles import DEFAULT_PROFILE
from openenv_daw.tasks.model import Task
from openenv_daw.tasks.store import TaskStore, default_store
from openenv_daw.tools import load_instructions

from .client import Policy

MAX_NUDGES = 2
NUDGE = "Use a tool, or call submit when you are finished."


@dataclass
class StepOut:
    data: dict[str, Any] | None
    error: dict[str, Any] | None
    done: bool
    reward: float | None
    metadata: dict[str, Any]


class EnvAdapter(Protocol):
    async def start(self, task_id: str) -> str: ...

    @property
    def profile(self) -> str: ...

    async def tools(self) -> list[dict[str, Any]]: ...
    async def call(self, name: str, args: dict[str, Any]) -> StepOut: ...
    async def close(self) -> None: ...


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def prompt_tools(
    listed: list[dict[str, Any]], profile: str = DEFAULT_PROFILE
) -> list[dict[str, Any]]:
    """Tool schemas as TRL renders them for ``DawToolEnv`` (one prompt surface for SFT data,
    rollouts and GRPO); the MCP schemas when ``transformers`` is not installed."""
    try:
        from openenv_daw.trl_env import tool_schemas

        schemas = [dict(s) for s in tool_schemas(profile)]
    except ImportError:
        return to_openai_tools(listed)
    names = {t["name"] for t in listed}
    schemas = [s for s in schemas if s["function"]["name"] in names]
    if {s["function"]["name"] for s in schemas} != names:
        return to_openai_tools(listed)
    return schemas


class InProcessAdapter:
    def __init__(
        self,
        store: TaskStore | None = None,
        profile: str = DEFAULT_PROFILE,
        backend_factory: Callable[[str], Backend] | None = None,
        param_mask_prob: float = 0.0,
    ) -> None:
        self.env = DawEnvironment(
            store, profile, backend_factory=backend_factory, param_mask_prob=param_mask_prob
        )

    @classmethod
    def from_env(cls, env: DawEnvironment) -> InProcessAdapter:
        """Wrap an existing environment (for example a UI session's)."""
        self = cls.__new__(cls)
        self.env = env
        return self

    @property
    def profile(self) -> str:
        return self.env.profile

    async def start(self, task_id: str) -> str:
        return self.env.reset(task_id=task_id).intent

    async def tools(self) -> list[dict[str, Any]]:
        obs = await self.env.step_async(ListToolsAction())
        return [t.model_dump() for t in obs.tools]  # type: ignore[attr-defined]

    async def call(self, name: str, args: dict[str, Any]) -> StepOut:
        obs = await self.env.step_async(CallToolAction(tool_name=name, arguments=args))
        err = (
            {"code": obs.error.error_type.value, "message": obs.error.message}
            if obs.error
            else None
        )  # type: ignore[attr-defined]
        return StepOut(result_data(obs), err, obs.done, obs.reward, obs.metadata)  # type: ignore[arg-type]

    async def close(self) -> None:
        self.env.close()


class WebSocketAdapter:
    def __init__(self, url: str) -> None:
        self.client = DawEnv(base_url=url)
        self._open = False
        self.profile = DEFAULT_PROFILE

    async def _ensure(self) -> None:
        if not self._open:
            await self.client.__aenter__()
            self._open = True

    async def start(self, task_id: str) -> str:
        await self._ensure()
        obs = (await self.client.reset(task_id=task_id)).observation
        self.profile = obs.profile or DEFAULT_PROFILE
        return obs.intent

    async def tools(self) -> list[dict[str, Any]]:
        await self._ensure()
        return await self.client.list_tools()

    async def call(self, name: str, args: dict[str, Any]) -> StepOut:
        r = await self.client.call(name, **args)
        return StepOut(r.observation.data, r.observation.error, r.done, r.reward, r.metadata or {})

    async def close(self) -> None:
        if self._open:
            await self.client.__aexit__(None, None, None)


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        v = json.loads(raw or "{}")
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        return {}


async def iter_episode(
    policy: Policy, env: EnvAdapter, task_id: str, max_turns: int = 40
) -> AsyncIterator[dict[str, Any]]:
    """Run one episode, yielding events: ``start`` (intent, tools), ``assistant`` (message),
    ``call`` (name, args, out, step) after every tool call and finally ``end`` (the record
    ``run_episode`` returns: chat ``messages``, ``tools`` and the verified outcome)."""
    intent = await env.start(task_id)
    tools = prompt_tools(await env.tools(), getattr(env, "profile", DEFAULT_PROFILE))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": load_instructions(getattr(env, "profile", DEFAULT_PROFILE))},
        {"role": "user", "content": intent},
    ]
    yield {"type": "start", "intent": intent, "tools": tools}
    final = StepOut(None, None, False, None, {})
    actions: list[dict[str, Any]] = []
    nudges = 0
    for _ in range(max_turns):
        msg = await policy.complete(messages, tools)
        messages.append(msg)
        yield {"type": "assistant", "message": msg}
        calls = msg.get("tool_calls") or []
        if not calls:
            if nudges >= MAX_NUDGES:
                break
            nudges += 1
            messages.append({"role": "user", "content": NUDGE})
            continue
        for tc in calls:
            if final.done:
                out = {"ok": False, "error": {"code": "invalid_args", "message": "episode ended"}}
            else:
                name, args = tc["function"]["name"], _parse_args(tc["function"]["arguments"])
                final = await env.call(name, args)
                actions.append({"tool": name, "args": args})
                out = final.data or {"ok": False, "error": final.error}
                yield {"type": "call", "name": name, "args": args, "out": out, "step": final}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(out),  # as DawToolEnv serializes results
                }
            )
        if final.done:
            break
    meta = final.metadata
    score = meta.get("score") or {}
    yield {
        "type": "end",
        "record": {
            "task_id": task_id,
            "model": getattr(policy, "name", ""),
            "intent": intent,
            "messages": messages,
            "tools": tools,
            "reward": final.reward if final.done else 0.0,
            "success": bool(meta.get("success")) if final.done else False,
            "done": final.done,
            "termination": meta.get("termination", "no_submit"),
            "failed_checks": meta.get("failed_checks", []),
            "nudges": nudges,
            "error_count": score.get("error_count"),
            "musicality": score.get("musicality"),
            "step_log": meta.get("step_log", []),
            "actions": actions,
            "state_hash": meta.get("state_hash"),
            "profile": meta.get("profile", getattr(env, "profile", DEFAULT_PROFILE)),
            "adapter_masked": meta.get("adapter_masked", []),
        },
    }


async def run_episode(
    policy: Policy, env: EnvAdapter, task_id: str, max_turns: int = 40
) -> dict[str, Any]:
    """Run one episode and return its record (see ``iter_episode``)."""
    record: dict[str, Any] = {}
    async for ev in iter_episode(policy, env, task_id, max_turns):
        if ev["type"] == "end":
            record = ev["record"]
    return record


def select_tasks(split: str, n: int, start: int = 0, family: str | None = None) -> list[Task]:
    store = default_store()
    if split == "heldout":
        return [t for t in store.heldout if family in (None, t.family)][start : start + n]
    seen: dict[str, Task] = {}
    i = start
    while len(seen) < n and i < start + 50 * n:
        t = store.sample("train", i, family)
        seen.setdefault(t.task_id, t)
        i += 1
    return list(seen.values())


async def run_many(
    make_policy: Any, make_env: Any, tasks: list[Task], samples: int, concurrency: int, out: Path
) -> list[dict[str, Any]]:
    queue: asyncio.Queue[str] = asyncio.Queue()
    for t in tasks:
        for _ in range(samples):
            queue.put_nowait(t.task_id)
    records: list[dict[str, Any]] = []
    lock = asyncio.Lock()

    async def worker() -> None:
        env, policy = make_env(), make_policy()
        try:
            while True:
                try:
                    tid = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                rec = await run_episode(policy, env, tid)
                async with lock:
                    records.append(rec)
                    with out.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
        finally:
            await env.close()

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return records


def main(argv: list[str] | None = None) -> None:
    from .client import OpenAIPolicy
    from .sft_export import export_sft

    ap = argparse.ArgumentParser(prog="openenv-daw-runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rollout", help="run an OpenAI-compatible model on tasks")
    r.add_argument("--split", default="train", choices=["train", "heldout"])
    r.add_argument("--family")
    r.add_argument("--start", type=int, default=0)
    r.add_argument("-n", type=int, default=20, help="number of tasks")
    r.add_argument("--samples", type=int, default=1, help="episodes per task")
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--url", help="WebSocket env server; default is in-process")
    r.add_argument("--out", default="outputs/rollouts.jsonl")
    r.add_argument("--temperature", type=float, default=0.7)
    v = sub.add_parser("eval", help="evaluate a policy and write report.json / report.md")
    v.add_argument("--tasks", default="heldout", help="'heldout' or a JSONL path")
    v.add_argument(
        "--policy",
        default="openai",
        choices=["openai", "oracle", "noop", "submit_only", "blocked", "random"],
    )
    v.add_argument("--samples", type=int, default=1)
    v.add_argument("--concurrency", type=int, default=4)
    v.add_argument("--url")
    v.add_argument(
        "--profile", help="score under one profile (default: portable, plus portable_plugins)"
    )
    v.add_argument("--out", default="outputs/eval")
    e = sub.add_parser("export", help="filter successful rollouts into SFT JSONL")
    e.add_argument("rollouts")
    e.add_argument("out")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        s = export_sft(Path(a.rollouts), Path(a.out))
        print(json.dumps(s))
        return
    if a.cmd == "eval":
        from .report import evaluate, evaluate_dual, load_task_set, report_markdown

        args = (a.policy, load_task_set(a.tasks), Path(a.out), a.samples, a.concurrency, a.url)
        if a.profile or a.url:  # a server's profile is fixed
            rep = asyncio.run(evaluate(*args, profile=a.profile or DEFAULT_PROFILE))
        else:
            rep = asyncio.run(evaluate_dual(*args))
        print(report_markdown(rep))
        for p, info in rep.get("informational", {}).items():
            print(
                f"\ninformational ({p}): {info['success']}/{info['episodes']} = {info['success_rate']:.1%}"
            )
        return
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks = select_tasks(a.split, a.n, a.start, a.family)
    recs = asyncio.run(
        run_many(
            lambda: OpenAIPolicy(temperature=a.temperature),
            (lambda: WebSocketAdapter(a.url)) if a.url else InProcessAdapter,
            tasks,
            a.samples,
            a.concurrency,
            out,
        )
    )
    ok = sum(r["success"] for r in recs)
    print(f"{ok}/{len(recs)} successful episodes -> {out}", file=sys.stderr)
