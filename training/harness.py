"""Rollout collection on OpenEnv's harness (``openenv.core.harness``), the same loop
``openenv collect`` runs: a ``ResourceSession`` per episode, ``MCPHarnessAdapter`` for the
tool-calling loop, ``RolloutSerializer`` for ``results.jsonl`` (+ ``metadata.json``) that can be
pushed to the Hub and read by TRL (``messages`` column).

``openenv collect`` only knows two environments, so this module is the env-owned session factory
the OpenEnv docs prescribe, plus a threaded ``collect`` over it. Tool schemas and tool-result
strings are the ones ``DawToolEnv`` (GRPO) uses, so SFT data, teacher rollouts and RL share one
prompt surface.
"""

from __future__ import annotations

import json
import os
import random
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction, Tool
from openenv.core.harness import (
    HarnessRunLimits,
    MCPHarnessAdapter,
    ModelStep,
    ModelStepResult,
    ResourceSession,
    ResourceSessionFactory,
    ToolResult,
    VerifyResult,
)
from openenv.core.harness.collect import EpisodeRecord, RolloutSerializer, build_model_step
from openenv.core.llm_client import (
    LLMClient,
    LLMResponse,
    OpenAIClient,
    ToolCall,
    _mcp_tools_to_openai,
    create_llm_client,
)
from openenv_daw.client import DawEnv
from openenv_daw.server.backends import Backend
from openenv_daw.server.daw_environment import DawEnvironment, result_data
from openenv_daw.server.tasks.model import Task
from openenv_daw.server.tasks.store import TaskStore, default_store
from openenv_daw.server.tools import load_instructions

SCRIPTED = ("oracle", "noop", "submit_only", "blocked", "random")


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


def prompt_tools(listed: list[dict[str, Any]]) -> list[Tool]:
    """Tools as TRL renders them for ``DawToolEnv`` (the MCP listing without ``transformers``)."""
    names = {t["name"] for t in listed}
    try:
        from openenv_daw.training.trl_env import tool_schemas

        fns = [s["function"] for s in tool_schemas()]
        if {f["name"] for f in fns} == names:
            return [
                Tool(
                    name=f["name"],
                    description=f.get("description", ""),
                    input_schema=f.get("parameters") or {"type": "object", "properties": {}},
                )
                for f in fns
            ]
    except ImportError:
        pass
    return [
        Tool(**{k: t[k] for k in ("name", "description", "input_schema") if k in t}) for t in listed
    ]


# --- one episode's link to the environment: in-process or a deployed server --------------------


class _Link:
    """``start`` / ``tools`` / ``call`` over a ``DawEnvironment`` or a ``DawEnv`` client."""

    def __init__(self, env: DawEnvironment | None, client: Any) -> None:
        self.env, self.client = env, client

    def _step(self, action: Any) -> Any:
        if self.env is not None:
            return self.env.step(action)
        return self.client.step(action).observation

    def start(self, task_id: str, seed: int | None, episode_id: str | None) -> str:
        if self.env is not None:
            return self.env.reset(task_id=task_id, seed=seed, episode_id=episode_id).intent
        obs = self.client.reset(task_id=task_id, seed=seed, episode_id=episode_id).observation
        return str(obs.intent)

    def tools(self) -> list[dict[str, Any]]:
        obs = self._step(ListToolsAction())
        return [t.model_dump() if hasattr(t, "model_dump") else dict(t) for t in obs.tools]

    def call(self, name: str, args: dict[str, Any]) -> tuple[Any, Any, bool, float | None, dict]:
        obs = self._step(CallToolAction(tool_name=name, arguments=args))
        if self.env is not None:
            data = result_data(obs)
            err = (
                {"code": obs.error.error_type.value, "message": obs.error.message}
                if obs.error
                else None
            )
        else:
            data, err = obs.data, obs.error
        return data, err, bool(obs.done), obs.reward, dict(obs.metadata or {})

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
        elif hasattr(self.client, "close"):
            self.client.close()


class DawSession(ResourceSession):
    def __init__(self, link: _Link, task_id: str, seed: int | None, episode_id: str | None):
        self.link, self.task_id = link, task_id
        self.intent = link.start(task_id, seed, episode_id)
        self._tools = prompt_tools(link.tools())
        self.done, self.reward = False, 0.0
        self.meta: dict[str, Any] = {}
        self.actions: list[dict[str, Any]] = []

    def initial_messages(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": load_instructions()},
            {"role": "user", "content": self.intent},
        ]

    def list_tools(self) -> list[Tool]:
        return list(self._tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if self.done:  # more calls in the same assistant turn after the episode ended
            out: Any = {"ok": False, "error": {"code": "invalid_args", "message": "episode ended"}}
            return ToolResult(data=json.dumps(out), done=True, metadata={"reward": self.reward})
        data, err, done, reward, meta = self.link.call(name, arguments)
        self.actions.append({"tool": name, "args": arguments})
        out = data or {"ok": False, "error": err}
        if done:
            self.done, self.meta = True, meta
        self.reward = float(reward or 0.0) if done else 0.0
        # the string DawToolEnv returns: tool messages match GRPO rollouts
        return ToolResult(data=json.dumps(out), done=done, metadata={"reward": self.reward})

    def verify(self, transcript: list[Any], final_state: Any | None = None) -> VerifyResult:
        meta = self.meta if self.done else {}
        score = meta.get("score") or {}
        return VerifyResult(
            env_reward=self.reward if self.done else 0.0,
            done=self.done,
            metrics={
                "task_id": self.task_id,
                "success": bool(meta.get("success")) if self.done else False,
                "termination": meta.get("termination", "no_submit"),
                "failed_checks": meta.get("failed_checks", []),
                "musicality": score.get("musicality"),
                "error_count": score.get("error_count"),
                "state_hash": meta.get("state_hash"),
                "step_log": meta.get("step_log", []),
                "actions": self.actions,
            },
        )

    def close(self) -> None:
        self.link.close()


class DawSessionFactory(ResourceSessionFactory[DawSession]):
    """In-process by default; ``base_url`` drives a deployed server ."""

    def __init__(
        self,
        store: TaskStore | None = None,
        backend_factory: Callable[[], Backend] | None = None,
        base_url: str | None = None,
    ) -> None:
        self.store, self.backend_factory, self.base_url = store, backend_factory, base_url

    def create(
        self, task: Any, seed: int | None = None, episode_id: str | None = None
    ) -> DawSession:
        task_id = task.task_id if isinstance(task, Task) else str(task)
        if self.base_url:
            link = _Link(None, DawEnv(base_url=self.base_url).sync())
        else:
            env = DawEnvironment(self.store, backend_factory=self.backend_factory)
            link = _Link(env, None)
        try:
            return DawSession(link, task_id, seed, episode_id)
        except Exception:
            link.close()
            raise


# --- model steps -----------------------------------------------------------------------------


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """MCP tool dicts as OpenAI tools, with the ``return`` key ``get_json_schema`` renders (and
    every TRL/SFT prompt contains) that OpenEnv's converter drops."""
    out = _mcp_tools_to_openai(tools)
    for t in out:
        t["function"].setdefault("return", {"type": "string"})
    return out


def parse_args(raw: str | None) -> dict[str, Any]:
    """Tool-call arguments as a dict; malformed JSON becomes ``{}`` so the environment scores the
    call as an error instead of the episode being lost."""
    try:
        v = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return v if isinstance(v, dict) else {}


class DawOpenAIClient(OpenAIClient):
    """``OpenAIClient`` that shows the model the TRL tool schemas and tolerates bad arguments."""

    async def complete_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kwargs: Any
    ) -> LLMResponse:
        kw = self._chat_completion_kwargs(messages, **kwargs)
        if tools:
            kw["tools"] = to_openai_tools(tools)
        msg = (await self._client.chat.completions.create(**kw)).choices[0].message
        calls = [
            ToolCall(tc.id, tc.function.name, parse_args(tc.function.arguments))
            for tc in msg.tool_calls or []
        ]
        return LLMResponse(msg.content or "", calls)


def teacher_client(
    provider: str = "openai",
    model: str | None = None,
    llm_endpoint: str | None = None,
    llm_port: int | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> LLMClient:
    """A hosted provider (``openai`` / ``anthropic``) or any OpenAI-compatible server
    (``llm_endpoint``; also ``OPENAI_BASE_URL``, model from ``OPENENV_DAW_MODEL``)."""
    model = model or os.environ.get("OPENENV_DAW_MODEL")
    if not model:
        raise ValueError("pass a model (or set OPENENV_DAW_MODEL)")
    if provider == "openai":
        llm_endpoint = llm_endpoint or os.environ.get("OPENAI_BASE_URL")
    if llm_endpoint:
        return DawOpenAIClient(
            llm_endpoint,
            llm_port,
            model,
            api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
    key = os.environ.get(f"{provider.upper()}_API_KEY") or os.environ.get("API_KEY")
    if not key:
        raise ValueError(f"set {provider.upper()}_API_KEY")
    return create_llm_client(provider, model, key, temperature=temperature, max_tokens=max_tokens)


def teacher_model_step(client: LLMClient | None = None, **kwargs: Any) -> ModelStep:
    return build_model_step(client or teacher_client(**kwargs))


def scripted_model_step(kind: str, tasks: list[Task], seed: int = 0) -> ModelStep:
    """Oracle ceiling and degenerate floors, as model steps ."""
    if kind not in SCRIPTED:
        raise ValueError(f"unknown policy {kind!r}")
    by_intent = {t.intent: t for t in tasks}
    rng = random.Random(seed)
    lock = threading.Lock()

    def call(name: str, args: dict[str, Any], i: int) -> ModelStepResult:
        return ModelStepResult(LLMResponse("", [ToolCall(f"call_{i}", name, args)]))

    def step(messages: list[Any], tools: list[Tool], sampling: dict[str, Any]) -> ModelStepResult:
        done = sum(m["role"] == "tool" for m in messages)
        if kind == "noop":
            return ModelStepResult(LLMResponse("I will not do anything."))
        if kind == "submit_only":
            return call("submit", {"status": "done"}, done) if not done else _text("done")
        if kind == "blocked":
            return call("submit", {"status": "blocked", "reason": "not possible"}, done)
        if kind == "oracle":
            steps = by_intent[messages[1]["content"]].oracle
            if done >= len(steps):
                return _text("done")
            return call(steps[done].tool, steps[done].args, done)
        if done >= 3:
            with lock:
                return call("submit", {"status": rng.choice(["done", "blocked"])}, done)
        with lock:
            pool: list[tuple[str, dict[str, Any]]] = [
                ("get_session", {}),
                ("list_devices", {}),
                ("create_track", {"name": f"T{rng.randrange(99)}", "kind": "midi"}),
                ("set_transport", {"tempo_bpm": float(rng.randrange(60, 180))}),
            ]
            name, args = rng.choice(pool)
        return call(name, args, done)

    return step


def _text(s: str) -> ModelStepResult:
    return ModelStepResult(LLMResponse(s))


# --- collection ------------------------------------------------------------------------------


def collect(
    factory: DawSessionFactory,
    model_step: ModelStep,
    tasks: list[Task],
    out_dir: Path,
    *,
    samples: int = 1,
    workers: int = 4,
    max_turns: int = 40,
    resume: bool = True,
    prefix: str = "ep",
    metadata: dict[str, Any] | None = None,
    failed: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run ``model_step`` on every task ``samples`` times and append each ``EpisodeRecord`` to
    ``out_dir/results.jsonl`` (resumable by episode id). Returns the records written; the task ids
    of episodes that crashed are appended to ``failed`` (and not recorded, so a resume retries)."""
    serializer = RolloutSerializer(out_dir)
    if not resume and serializer.results_path.exists():
        serializer.results_path.unlink()
    serializer.write_metadata(
        {
            "env": "openenv-daw",
            "num_episodes_requested": len(tasks) * samples,
        }
        | (metadata or {})
    )
    plan = [
        (f"{prefix}-{i:06d}", t) for i, t in enumerate(t for t in tasks for _ in range(samples))
    ]
    done_ids = serializer.collected_episode_ids()
    limits = HarnessRunLimits(max_turns=max_turns)
    adapter, lock = MCPHarnessAdapter(), threading.Lock()
    written: list[dict[str, Any]] = []

    def run(item: tuple[str, Task]) -> None:
        episode_id, task = item
        if episode_id in done_ids:
            return
        try:
            session = factory.create(task, episode_id=episode_id)
        except Exception as exc:
            if failed is not None:
                failed.append(task.task_id)
            return print(f"failed {episode_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
        try:
            rollout = adapter.run_white_box(model_step, session, limits)
            verify = session.verify(rollout.messages)
            record = EpisodeRecord.from_rollout(
                episode_id,
                rollout,
                verify,
                task=task.task_id,
            )
        except Exception as exc:  # a malformed teacher reply must not kill the whole job
            if failed is not None:
                failed.append(task.task_id)
            return print(f"failed {episode_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            session.close()
        with lock:
            serializer.write_episode(record)
            written.append(record.to_dict())

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(run, plan))
    return written


def read_results(path: Path) -> list[dict[str, Any]]:
    """Records of a collect dir (or its ``results.jsonl``)."""
    p = path / "results.jsonl" if path.is_dir() else path
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def flat(rec: dict[str, Any]) -> dict[str, Any]:
    """An ``EpisodeRecord`` dict as the flat row the report reads (rows written by the old runner
    are already flat and returned as is)."""
    if "verify_metrics" not in rec:
        return rec
    return {
        "task_id": rec["task"],
        "reward": rec["reward"],
        **rec["verify_metrics"],
    }
