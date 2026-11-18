"""DawEnvironment: an OpenEnv MCPEnvironment over the JSON DAW simulator.

One task per episode. The agent acts only through MCP tools (``CallToolAction``); the terminal
reward is computed by ``DawRubric`` when the agent calls ``submit``, hits the step limit, or
makes too many consecutive errors."""

from __future__ import annotations

import importlib.metadata
import random
import uuid
from collections.abc import Callable
from typing import Any

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
)
from openenv.core.env_server.types import Action, EnvironmentMetadata, Observation
from openenv_daw.models import DawResetObservation, DawState
from openenv_daw.server.backends import Backend, SimBackend
from openenv_daw.server.episode import MAX_CONSECUTIVE_ERRORS, MAX_STEPS, Episode
from openenv_daw.server.rewards import DawRubric
from openenv_daw.server.tasks.store import TaskStore, default_store
from openenv_daw.server.tools import build_server

SPLITS = ("train", "heldout")
TRAIN_EPOCH = 4096  # advertised size of the generated train split (``reset(index=i)`` for i < 4096)
TRAIN_PREVIEW = 64


def result_data(obs: CallToolObservation) -> dict[str, Any] | None:
    """The tool's JSON dict result, if it returned one."""
    res = obs.result
    if res is None:
        return None
    data = getattr(res, "data", None)
    if isinstance(data, dict):
        return data
    sc = getattr(res, "structured_content", None)
    if isinstance(sc, dict):
        return sc.get("result", sc) if isinstance(sc.get("result", None), dict) else sc
    return None


class DawEnvironment(MCPEnvironment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        store: TaskStore | None = None,
        max_steps: int = MAX_STEPS,
        max_consecutive_errors: int = MAX_CONSECUTIVE_ERRORS,
        backend_factory: Callable[[], Backend] | None = None,
    ) -> None:
        self.backend_factory = backend_factory
        self.store = store or default_store()
        self.max_steps = max_steps
        self.max_consecutive_errors = max_consecutive_errors
        self.episode: Episode | None = None
        self._idle_backend = SimBackend()
        self._state = DawState(episode_id=str(uuid.uuid4()))
        super().__init__(
            build_server(
                lambda: self.backend,
                on_submit=lambda status, reason: self._require_episode().on_submit(status, reason),
            )
        )
        self.rubric = DawRubric()

    @property
    def backend(self) -> Backend:
        return self.episode.backend if self.episode else self._idle_backend

    def _require_episode(self) -> Episode:
        if self.episode is None:
            raise RuntimeError("call reset before step")
        return self.episode

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        *,
        task_id: str | None = None,
        split: str = "train",
        family: str | None = None,
        index: int | None = None,
        **kwargs: Any,
    ) -> DawResetObservation:
        if seed is None:  # unseeded resets must not all land on the same task
            seed = random.SystemRandom().randrange(2**31)
        if task_id is not None:
            task = self.store.get(task_id)
        elif index is not None:
            task = self.store.get_task(split, index)
        else:
            task = self.store.sample(split, seed, family)
        self.episode = Episode(
            task, self.max_steps, self.max_consecutive_errors, self.backend_factory
        )
        self._reset_rubric()
        self._state = DawState(
            episode_id=episode_id or str(uuid.uuid4()),
            task_id=task.task_id,
            state_hash=self.backend.session.state_hash(),
        )
        return DawResetObservation(
            task_id=task.task_id,
            intent=task.intent,
            metadata={"family": task.family, "split": task.split, "seed": seed},
        )

    def _post_step(self, action: Action, obs: Observation) -> Observation:
        if isinstance(action, CallToolAction) and isinstance(obs, CallToolObservation):
            ep = self._require_episode()
            transport = (
                {"code": obs.error.error_type.value, "message": obs.error.message}
                if obs.error is not None
                else None
            )
            ep.record(action.tool_name, action.arguments, result_data(obs), transport)
            why = ep.after_step()
            s = self._state
            s.step_count, s.consecutive_errors = ep.step_count, ep.consecutive_errors
            s.error_count, s.step_log = ep.error_count, ep.step_log
            s.state_hash = self.backend.session.state_hash()
            if why is not None:
                assert ep.score is not None
                s.done, s.score, s.success = True, ep.score.as_dict(), ep.score.success
                s.status = ep.score.status
                obs.done = True
                obs.metadata.update(ep.result_metadata())
        return obs

    def _check_open(self, action: Action) -> None:
        if not isinstance(action, CallToolAction):
            return  # listing tools is allowed at any time (the web UI does it before reset)
        if self.episode is None:
            raise RuntimeError("call reset before step")
        if self.episode.done:
            raise RuntimeError("episode is over; call reset")

    def step(self, action: Action, timeout_s: float | None = None, **kwargs: Any) -> Observation:
        self._check_open(action)
        obs = self._post_step(action, super().step(action, timeout_s=timeout_s, **kwargs))
        obs.reward = self._apply_rubric(action, obs)
        return obs

    async def step_async(
        self, action: Action, timeout_s: float | None = None, **kwargs: Any
    ) -> Observation:
        self._check_open(action)
        obs = self._post_step(
            action, await super().step_async(action, timeout_s=timeout_s, **kwargs)
        )
        obs.reward = await self._apply_rubric_async(action, obs)
        return obs

    def _step_impl(
        self, action: Action, timeout_s: float | None = None, **kwargs: Any
    ) -> Observation:
        return Observation(
            metadata={"error": f"unsupported action {type(action).__name__}; use CallToolAction"}
        )

    # --- Task API: discovery only, no side effects -------------------------------------------
    # ``train`` tasks are generated from seeds, so its size is the epoch size we advertise.

    def list_splits(self) -> list[str]:
        return list(SPLITS)

    def num_tasks(self, split: str) -> int:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected {list(SPLITS)}")
        return TRAIN_EPOCH if split == "train" else len(self.store.heldout)

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        n = self.num_tasks(split)
        if not 0 <= index < n:
            raise IndexError(f"index {index} out of range for split {split} (n={n})")
        task = self.store.get_task(split, index)
        return {
            "id": task.task_id,
            "index": index,
            "split": split,
            "family": task.family,
            "intent": task.intent,
        }

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, Any]]:
        n = self.num_tasks(split)
        lo = 0 if start is None else max(start, 0)
        hi = n if stop is None else min(stop, n)
        return [self.get_task(split, i) for i in range(lo, hi)]

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        """Every heldout task; a bounded preview of ``train`` (``num_tasks`` is the real size)."""
        n = self.num_tasks(split)
        return self.get_task_range(split, 0, n if split == "heldout" else min(n, TRAIN_PREVIEW))

    @property
    def state(self) -> DawState:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="openenv_daw",
            description="DAW tool-use environment: a JSON DAW simulator behind portable MCP tools",
            version=importlib.metadata.version("openenv-daw"),
            documentation_url="https://huggingface.co/spaces/gabguerin/openenv-daw",
        )


__all__ = ["CallToolAction", "DawEnvironment", "ListToolsAction", "result_data"]
