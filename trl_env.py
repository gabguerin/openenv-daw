"""TRL ``environment_factory`` class generated from the FastMCP tool registry.

Tool names, signatures and docstrings come from ``tools.registry`` (no duplication), and calls
go straight to the in-process simulator (no MCP client round trip) for training throughput.

    trainer = GRPOTrainer(model=..., train_dataset=build_dataset("train", 512),
                          environment_factory=DawToolEnv, args=GRPOConfig(...))
"""

from __future__ import annotations

import functools
import inspect
import itertools
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, get_args, get_origin

from openenv.core.env_server.mcp_environment import get_server_tools
from openenv_daw.server.backends import Backend, SimBackend
from openenv_daw.server.episode import MAX_CONSECUTIVE_ERRORS, MAX_STEPS, Episode
from openenv_daw.server.rewards import DawRubric
from openenv_daw.server.tasks.store import TaskStore, default_store
from openenv_daw.server.tools import NoteIn, build_server, load_instructions

_SEEDS = itertools.count()


def _plain(annotation: Any) -> Any:
    """Swap pydantic models for ``dict`` so transformers' JSON-schema derivation accepts them."""
    if annotation is NoteIn:
        return dict
    args = get_args(annotation)
    if args and get_origin(annotation) is not None:
        new = tuple(_plain(a) for a in args)
        if new != args:
            origin = get_origin(annotation)
            try:
                return annotation.copy_with(new)  # typing generics, incl. Optional/Union
            except AttributeError:
                return (
                    origin[new]
                    if origin is not None and origin.__module__ == "builtins"
                    else annotation
                )
    return annotation


def _flatten(annotation: Any) -> list[Any]:
    return [annotation, *[x for a in get_args(annotation) for x in _flatten(a)]]


def _make_method(name: str, fn: Callable[..., Any]) -> Callable[..., str]:
    sig = inspect.signature(fn)
    hints = {
        k: _plain(v) for k, v in inspect.get_annotations(fn, eval_str=True).items() if k != "return"
    }

    def method(self: Any, *args: Any, **kwargs: Any) -> str:
        names = list(sig.parameters)
        arguments = {**dict(zip(names, args, strict=False)), **kwargs}
        return self._call(name, arguments)

    method.__name__ = name
    method.__qualname__ = f"DawToolEnv.{name}"
    doc = fn.__doc__ or ""
    if "notes" in sig.parameters and NoteIn in _flatten(
        inspect.get_annotations(fn, eval_str=True)["notes"]
    ):
        fields = ", ".join(f"{k} ({v.description})" for k, v in NoteIn.model_fields.items())
        doc = re.sub(
            r"^(\s*notes:.*)$",
            lambda m: f"{m.group(1)} Each note is an object with: {fields}.",
            doc,
            count=1,
            flags=re.M,
        )
    method.__doc__ = doc
    method.__annotations__ = {**hints, "return": str}
    method.__signature__ = sig.replace(  # type: ignore[attr-defined]
        parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            *[
                p.replace(annotation=hints.get(p.name, p.annotation))
                for p in sig.parameters.values()
            ],
        ],
        return_annotation=str,
    )
    return method


def _log_episode(path: Path, ep: Episode) -> None:
    sc = ep.score
    assert sc is not None
    row = {
        "task_id": ep.task.task_id,
        "family": ep.task.family,
        "reward": sc.reward,
        "success": sc.success,
        "gate_passed": sc.gate_passed,
        "musicality": sc.musicality,
        "calls": sc.calls,
        "error_count": sc.error_count,
        "termination": ep.termination,
        "failed_checks": sc.failed_checks,
    }
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def make_environment_class(
    store: TaskStore | None = None,
    max_steps: int = MAX_STEPS,
    max_consecutive_errors: int = MAX_CONSECUTIVE_ERRORS,
    backend_factory: Callable[[], Backend] | None = None,
    shaped: bool = False,
    episode_log: Path | None = None,
) -> type:
    """Build an ``environment_factory`` class exposing every tool (plus submit). ``shaped`` scores
    with the dense training reward (``DawRubric(shaped=True)``) instead of the strict one. With
    ``episode_log`` every finished episode appends a JSON line (reward, success, termination...)."""
    template = build_server(lambda: SimBackend(), on_submit=lambda s, r: {})
    tools = get_server_tools(template)

    class _Env:
        def __init__(self) -> None:
            self.store = store or default_store()
            self.episode: Episode | None = None

        def reset(self, task_id: str | None = None, seed: int | None = None, **kwargs: Any) -> None:
            if task_id:
                task = self.store.get(task_id)
            else:  # no task given: sample by seed (a process-wide counter when none is passed)
                task = self.store.sample("train", next(_SEEDS) if seed is None else seed)
            self.episode = Episode(
                task,
                max_steps,
                max_consecutive_errors,
                backend_factory,
                DawRubric(shaped=True) if shaped else None,
            )

        def get_reward(self) -> float:
            ep = self.episode
            if ep is None:
                return 0.0
            if not ep.done:  # stopped without submit or hit the generation limit: truncated
                ep.finalize(True, "no_submit")
            assert ep.score is not None
            if episode_log is not None:
                _log_episode(episode_log, ep)
            return float(ep.score.reward)

        def _call(self, name: str, arguments: dict[str, Any]) -> str:
            if self.episode is None:
                raise RuntimeError("environment not reset")
            if self.episode.done:
                raise ValueError("The episode is over; no further tool calls are accepted.")
            return json.dumps(self.episode.run(name, arguments))

    for name, tool in tools.items():
        setattr(_Env, name, _make_method(name, tool.fn))
    _Env.__name__ = _Env.__qualname__ = "DawToolEnv"
    return _Env


DawToolEnv = make_environment_class()


@functools.cache
def tool_schemas() -> tuple[dict[str, Any], ...]:
    """The tool schemas TRL shows the model for ``DawToolEnv`` (``transformers.get_json_schema``
    over the generated methods, in TRL's order). The runner and the SFT export use these, so SFT
    data, rollouts and GRPO all render the same prompt."""
    from transformers.utils import get_json_schema

    cls = DawToolEnv
    inst = cls()
    return tuple(
        get_json_schema(getattr(inst, name))
        for name, _ in inspect.getmembers(cls, predicate=inspect.isfunction)
        if name not in ("reset", "get_reward") and not name.startswith("_")
    )


def _stratified_order(tasks: list[Any]) -> list[int]:
    """Indices of ``tasks`` interleaved round-robin across families (file order within one)."""
    by_family: dict[str, list[int]] = {}
    for i, t in enumerate(tasks):
        by_family.setdefault(t.family, []).append(i)
    order: list[int] = []
    for rank in range(max(map(len, by_family.values()), default=0)):
        order += [v[rank] for v in by_family.values() if rank < len(v)]
    return order


def build_dataset(
    split: str = "train",
    n: int = 256,
    store: TaskStore | None = None,
    stratified: bool = False,
    task_ids: list[str] | None = None,
) -> list[dict]:
    """Rows for ``train_dataset``: a conversational ``prompt`` plus the ``task_id`` that ``reset``
    receives. Works with ``datasets.Dataset.from_list``. With ``stratified`` the heldout rows are
    interleaved across families, so any prefix of ``n`` rows covers every family. ``task_ids``
    selects specific tasks instead (for example those a model solves only some of the time)."""
    store = store or default_store()
    instructions = load_instructions()
    if task_ids is not None:
        tasks = [store.get(i) for i in task_ids]
    else:
        order = _stratified_order(store.heldout) if stratified and split == "heldout" else None
        tasks = [store.get_task(split, order[i % len(order)] if order else i) for i in range(n)]
    return [
        {
            "prompt": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": t.intent},
            ],
            "task_id": t.task_id,
        }
        for t in tasks
    ]
