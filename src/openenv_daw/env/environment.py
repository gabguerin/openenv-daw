"""DawEnvironment: an OpenEnv MCPEnvironment over the JSON DAW simulator.

One task per episode. The agent acts only through MCP tools (``CallToolAction``); the terminal
reward is computed by ``DawRubric`` when the agent calls ``submit``, hits the step limit, or
makes too many consecutive errors."""

from __future__ import annotations

import os
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

from openenv_daw.backends import Backend, SimBackend
from openenv_daw.rewards import DawRubric
from openenv_daw.sim.profiles import DEFAULT_PROFILE
from openenv_daw.tasks.model import Task
from openenv_daw.tasks.resolve import apply_param_mask
from openenv_daw.tasks.store import TaskStore, default_store
from openenv_daw.tools import build_server
from openenv_daw.versions import SCHEMA_VERSION, TOOLSET_VERSION

from .episode import Episode
from .models import DawResetObservation, DawState

PARAM_MASK_PROB = float(
    os.getenv("OPENENV_DAW_PARAM_MASK_PROB", "0")
)  # train split only; DawToolEnv defaults to 0.1
MAX_STEPS = int(os.getenv("OPENENV_DAW_MAX_STEPS", "30"))
MAX_CONSECUTIVE_ERRORS = int(os.getenv("OPENENV_DAW_MAX_CONSECUTIVE_ERRORS", "3"))


def package_metadata() -> EnvironmentMetadata:
    from importlib import metadata

    try:
        md = metadata.metadata("openenv-daw")
        version = md["Version"]
        description = md["Summary"] or "DAW tool-use environment"
        readme = metadata.metadata("openenv-daw").json.get("description") or None  # long description
    except metadata.PackageNotFoundError:  # running from a source tree
        version, description, readme = "0.0.0", "DAW tool-use environment", None
    return EnvironmentMetadata(
        name="openenv_daw",
        description=description,
        readme_content=readme,
        version=version,
        documentation_url="https://huggingface.co/spaces/gabguerin/openenv-daw",
    )


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
        backend_factory: Callable[[str], Backend] | None = None,
        param_mask_prob: float = PARAM_MASK_PROB,
    ) -> None:
        self.param_mask_prob = param_mask_prob
        self.backend_factory = backend_factory
        self.store = store or default_store()
        self.profile = profile
        self.max_steps = max_steps
        self.max_consecutive_errors = max_consecutive_errors
        self.episode: Episode | None = None
        self._idle_backend = SimBackend(profile=profile)
        self._state = DawState(episode_id=str(uuid.uuid4()), profile=profile)
        super().__init__(
            build_server(
                lambda: self.backend,
                profile,
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
        if seed is None:  # unseeded resets must not all land on the same task
            seed = random.SystemRandom().randrange(2**31)
        if task_id is not None:
            task = self.store.get(task_id, self.profile)
        elif index is not None:
            task = self.store.get_task(split, index, self.profile)
        else:
            task = self.store.sample(split, seed, family, self.profile)
        task, mask = apply_param_mask(task, self.profile, self.param_mask_prob, seed)
        self.episode = Episode(
            task,
            self.profile,
            self.max_steps,
            self.max_consecutive_errors,
            self.backend_factory,
            mask,
        )
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
            metadata={
                "family": task.family,
                "split": task.split,
                "seed": seed,
                "toolset_version": TOOLSET_VERSION,
                "schema_version": SCHEMA_VERSION,
            },
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

    # --- web interface (stock Playground tab) -----------------------------------------
    def render_web(self, observation: dict[str, Any]) -> str | None:
        ep = self.episode
        if ep is None:
            return None
        import html

        from openenv_daw.server.ui import render

        final = ep.score.as_dict() if ep.score else None
        return (
            '<p class="dw-mut">This Playground is one environment shared by every visitor; '
            "use the DAW Studio tab for your own session.</p>"
            f"<h3>{html.escape(ep.task.intent)}</h3>"
            + render.session_html(ep.backend.session)
            + render.checks_html(ep.task, ep.initial, ep.backend.session, ep.error_count, final)
        )

    def web_actions(self, observation: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        def call(tool: str, **args: Any) -> dict[str, Any]:
            return {"type": "call_tool", "tool_name": tool, "arguments": args}

        return [
            ("get_session", call("get_session")),
            ("list_devices", call("list_devices")),
            ("submit done", call("submit", status="done")),
            ("submit blocked", call("submit", status="blocked", reason="not possible")),
        ]

    @property
    def state(self) -> DawState:
        return self._state

    # --- metadata and Task API (OpenEnv standard) ---------------------------------------
    def get_metadata(self) -> EnvironmentMetadata:
        return package_metadata()

    TRAIN_PREVIEW = 20  # train is generated from seeds; list_tasks returns a bounded preview

    @staticmethod
    def _task_spec(task: Task) -> dict[str, Any]:
        """Prompt-level view of a task: no checks, no oracle."""
        return {
            "task_id": task.task_id,
            "family": task.family,
            "split": task.split,
            "intent": task.intent,
            "tags": task.tags,
            "expected_status": task.expected_status,
        }

    def _check_split(self, split: str) -> None:
        if split not in self.store.list_splits():
            raise IndexError(f"unknown split {split!r}")

    def list_splits(self) -> list[dict[str, str]]:
        return [
            {"name": s, "type": "train" if s == "train" else "test"}
            for s in self.store.list_splits()
        ]

    def num_tasks(self, split: str) -> int:
        self._check_split(split)
        return self.store.num_tasks(split)

    def get_task(self, split: str, index: int) -> dict[str, Any]:
        self._check_split(split)
        if index < 0 or index >= self.store.num_tasks(split):
            raise IndexError(index)
        return self._task_spec(self.store.get_task(split, index, self.profile))

    def get_task_range(
        self, split: str, start: int | None = None, stop: int | None = None
    ) -> list[dict[str, Any]]:
        n = self.num_tasks(split)
        lo, hi, _ = slice(start, stop).indices(n)
        if hi - lo > 1000:
            raise IndexError("range too large (max 1000 tasks)")
        return [self.get_task(split, i) for i in range(lo, hi)]

    def list_tasks(self, split: str) -> list[dict[str, Any]]:
        """All heldout tasks; a preview of the first seeds for the unbounded train split."""
        n = self.num_tasks(split)
        return self.get_task_range(split, 0, min(n, self.TRAIN_PREVIEW) if split == "train" else n)


__all__ = ["CallToolAction", "DawEnvironment", "ListToolsAction", "result_data"]
