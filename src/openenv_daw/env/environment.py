"""DawEnvironment: an OpenEnv MCPEnvironment over the JSON DAW simulator.

One task per episode. The agent acts only through MCP tools (``CallToolAction``); the terminal
reward is computed by ``DawRubric`` when the agent calls ``submit``, hits the step limit, or
makes too many consecutive errors."""

from __future__ import annotations

import os
import uuid
from typing import Any

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.mcp_types import (
    CallToolAction,
    CallToolObservation,
    ListToolsAction,
)
from openenv.core.env_server.types import Action, Observation

from openenv_daw.backends import SimBackend
from openenv_daw.rewards import DawRubric, score_episode
from openenv_daw.schema import Session
from openenv_daw.sim.profiles import DEFAULT_PROFILE
from openenv_daw.tasks.model import Task
from openenv_daw.tasks.store import TaskStore, default_store
from openenv_daw.tools import build_server

from .models import DawResetObservation, DawState

MAX_STEPS = int(os.getenv("OPENENV_DAW_MAX_STEPS", "30"))
MAX_CONSECUTIVE_ERRORS = int(os.getenv("OPENENV_DAW_MAX_CONSECUTIVE_ERRORS", "3"))


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
        profile: str = DEFAULT_PROFILE,
        max_steps: int = MAX_STEPS,
        max_consecutive_errors: int = MAX_CONSECUTIVE_ERRORS,
    ) -> None:
        self.store = store or default_store()
        self.profile = profile
        self.max_steps = max_steps
        self.max_consecutive_errors = max_consecutive_errors
        self.backend = SimBackend(profile=profile)
        self._task: Task | None = None
        self._initial = Session()
        self._submitted: tuple[str, str] | None = None
        self._done = True
        self._state = DawState(episode_id=str(uuid.uuid4()), profile=profile)
        super().__init__(build_server(lambda: self.backend, profile, on_submit=self._on_submit))
        self.rubric = DawRubric()

    # --- submit tool hook ---------------------------------------------------------------
    def _on_submit(self, status: str, reason: str) -> dict[str, Any]:
        self._submitted = (status, reason)
        return {"ok": True, "status": status, "message": "Task submitted."}

    # --- episode ------------------------------------------------------------------------
    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        *,
        task_id: str | None = None,
        split: str = "train",
        family: str | None = None,
        index: int | None = None,
        profile: str | None = None,
        **kwargs: Any,
    ) -> DawResetObservation:
        if profile is not None and profile != self.profile:
            raise ValueError("the profile is fixed per environment instance; start a new server")
        if task_id is not None:
            task = self.store.get(task_id)
        elif index is not None:
            task = self.store.get_task(split, index)
        else:
            task = self.store.sample(split, seed or 0, family)
        self._task = task
        self._initial = Session.restore(task.initial_state) if task.initial_state else Session()
        self.backend = SimBackend(Session.restore(self._initial.snapshot()), self.profile)
        self._submitted = None
        self._done = False
        self._reset_rubric()
        self._state = DawState(
            episode_id=episode_id or str(uuid.uuid4()),
            task_id=task.task_id,
            profile=self.profile,
            state_hash=self.backend.session.state_hash(),
        )
        return DawResetObservation(
            task_id=task.task_id,
            intent=task.intent,
            profile=self.profile,
            metadata={"family": task.family, "split": task.split},
        )

    def _record(self, action: CallToolAction, obs: CallToolObservation) -> bool:
        """Log the verified step; return whether it was an error."""
        data = result_data(obs)
        if obs.error is not None:
            err: dict[str, Any] | None = {
                "code": obs.error.error_type.value,
                "message": obs.error.message,
            }
        elif data is not None and data.get("ok") is False:
            err = data.get("error")
        elif data is None:
            err = {"code": "execution_error", "message": "tool returned no structured result"}
        else:
            err = None
        s = self._state
        s.step_count += 1
        s.step_log.append(
            {
                "step": s.step_count,
                "tool": action.tool_name,
                "arguments": action.arguments,
                "ok": err is None,
                "error": err,
                "result": data,
            }
        )
        if err is None:
            s.consecutive_errors = 0
        else:
            s.consecutive_errors += 1
            s.error_count += 1
        return err is not None

    def _finalize(self, action: Action, obs: Observation, truncated: bool, why: str) -> None:
        assert self._task is not None
        status = self._submitted[0] if self._submitted else None
        score = score_episode(
            self._task,
            self._initial,
            self.backend.session,
            status,
            self._state.error_count,
            truncated,
        )
        s = self._state
        s.done, s.status, s.success, s.score = True, status, score.success, score.as_dict()
        s.state_hash = self.backend.session.state_hash()
        self._done = True
        obs.done = True
        obs.metadata.update(
            {
                "score": s.score,
                "success": score.success,
                "termination": why,
                "truncated": truncated,
                "failed_checks": score.failed_checks,
                "step_log": s.step_log,
                "state_hash": s.state_hash,
            }
        )

    def _post_step(self, action: Action, obs: Observation) -> Observation:
        if isinstance(action, CallToolAction) and isinstance(obs, CallToolObservation):
            self._record(action, obs)
            self._state.state_hash = self.backend.session.state_hash()
            s = self._state
            if self._submitted is not None:
                self._finalize(action, obs, False, "submit")
            elif s.consecutive_errors >= self.max_consecutive_errors:
                self._finalize(action, obs, True, "error_limit")
            elif s.step_count >= self.max_steps:
                self._finalize(action, obs, True, "step_limit")
        obs.reward = self._apply_rubric(action, obs)
        return obs

    def _check_open(self, action: Action) -> None:
        if self._task is None:
            raise RuntimeError("call reset before step")
        if self._done and isinstance(action, CallToolAction):
            raise RuntimeError("episode is over; call reset")

    def step(self, action: Action, timeout_s: float | None = None, **kwargs: Any) -> Observation:
        self._check_open(action)
        obs = super().step(action, timeout_s=timeout_s, **kwargs)
        return self._post_step(action, obs)

    async def step_async(
        self, action: Action, timeout_s: float | None = None, **kwargs: Any
    ) -> Observation:
        self._check_open(action)
        obs = await super().step_async(action, timeout_s=timeout_s, **kwargs)
        return self._post_step(action, obs)

    def _step_impl(
        self, action: Action, timeout_s: float | None = None, **kwargs: Any
    ) -> Observation:
        return Observation(
            metadata={"error": f"unsupported action {type(action).__name__}; use CallToolAction"}
        )

    @property
    def state(self) -> DawState:
        return self._state


__all__ = ["CallToolAction", "DawEnvironment", "ListToolsAction", "result_data"]
