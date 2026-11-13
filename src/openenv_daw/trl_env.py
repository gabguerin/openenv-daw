"""TRL ``environment_factory`` class generated from the FastMCP tool registry.

Tool names, signatures and docstrings come from ``tools.registry`` (no duplication), and calls
go straight to the in-process simulator (no MCP client round trip) for training throughput.

    trainer = GRPOTrainer(model=..., train_dataset=build_dataset("train", 512),
                          environment_factory=DawToolEnv, args=GRPOConfig(...))
"""

from __future__ import annotations

import inspect
import itertools
import json
import re
from collections.abc import Callable
from typing import Any, get_args, get_origin

from openenv.core.env_server.mcp_environment import get_server_tools

from openenv_daw.backends import Backend, SimBackend
from openenv_daw.env.episode import Episode
from openenv_daw.sim.profiles import DEFAULT_PROFILE
from openenv_daw.tasks.resolve import apply_param_mask
from openenv_daw.tasks.store import TaskStore, default_store
from openenv_daw.tools.registry import NoteIn, build_server, load_instructions

_SEEDS = itertools.count()
MAX_STEPS = 30
MAX_CONSECUTIVE_ERRORS = 3


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


def make_environment_class(
    profile: str = DEFAULT_PROFILE,
    store: TaskStore | None = None,
    max_steps: int = MAX_STEPS,
    max_consecutive_errors: int = MAX_CONSECUTIVE_ERRORS,
    backend_factory: Callable[[str], Backend] | None = None,
    param_mask_prob: float = 0.1,
) -> type:
    """Build an ``environment_factory`` class exposing every tool of ``profile`` (plus submit)."""
    template = build_server(lambda: SimBackend(profile=profile), profile, on_submit=lambda s, r: {})
    tools = get_server_tools(template)

    class _Env:
        def __init__(self) -> None:
            self.store = store or default_store()
            self.episode: Episode | None = None

        def reset(self, task_id: str | None = None, seed: int | None = None, **kwargs: Any) -> None:
            if task_id:
                task = self.store.get(task_id, profile)
            else:  # no task given: sample by seed (a process-wide counter when none is passed)
                task = self.store.sample(
                    "train", next(_SEEDS) if seed is None else seed, profile=profile
                )
            task, mask = apply_param_mask(
                task, profile, param_mask_prob, seed if seed is not None else task.task_id
            )
            self.episode = Episode(
                task, profile, max_steps, max_consecutive_errors, backend_factory, mask
            )

        def get_reward(self) -> float:
            ep = self.episode
            if ep is None:
                return 0.0
            if not ep.done:  # stopped without submit or hit the generation limit: truncated
                ep.finalize(True, "no_submit")
            assert ep.score is not None
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


def build_dataset(
    split: str = "train",
    n: int = 256,
    store: TaskStore | None = None,
    profile: str = DEFAULT_PROFILE,
) -> list[dict]:
    """Rows for ``train_dataset``: a conversational ``prompt`` plus the ``task_id`` that ``reset``
    receives. Works with ``datasets.Dataset.from_list``."""
    store = store or default_store()
    instructions = load_instructions(profile)
    rows = []
    for i in range(n):
        task = store.get_task(split, i, profile)
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": task.intent},
                ],
                "task_id": task.task_id,
            }
        )
    return rows
