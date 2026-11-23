"""Episode bookkeeping shared by DawEnvironment (MCP path) and the TRL wrapper (direct path):
limits, step log and terminal scoring."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from openenv_daw.server.backends import Backend, SimBackend
from openenv_daw.server.rewards.score import DawRubric, EpisodeScore, score_episode
from openenv_daw.server.sim.session import Session
from openenv_daw.server.tasks.model import Task

MAX_STEPS = 30
MAX_CONSECUTIVE_ERRORS = 3

READ_TOOLS = frozenset({"get_session", "list_devices", "get_device_params"})


class Episode:
    def __init__(
        self,
        task: Task,
        max_steps: int = MAX_STEPS,
        max_consecutive_errors: int = MAX_CONSECUTIVE_ERRORS,
        backend_factory: Callable[[], Backend] | None = None,
        rubric: DawRubric | None = None,
    ) -> None:
        self.task = task
        self.rubric = rubric
        self.max_steps, self.max_consecutive_errors = max_steps, max_consecutive_errors
        self.initial = Session.restore(task.initial_state) if task.initial_state else Session()
        self.backend: Backend = (backend_factory or SimBackend)()
        self.backend.load(self.initial)
        self.step_count = 0
        self.consecutive_errors = 0
        self.error_count = 0
        self.step_log: list[dict[str, Any]] = []
        self.submitted: tuple[str, str] | None = None
        self.done = False
        self.termination: str | None = None
        self.score: EpisodeScore | None = None

    def on_submit(self, status: str, reason: str) -> dict[str, Any]:
        self.submitted = (status, reason)
        return {
            "ok": True,
            "status": status,
            "message": "Task submitted. Reply briefly with no further tool calls.",
        }

    def record(
        self,
        tool: str,
        arguments: dict[str, Any],
        data: dict[str, Any] | None,
        transport_error: dict[str, Any] | None = None,
    ) -> bool:
        """Log a verified step; returns whether it was an error."""
        if transport_error is not None:
            err: dict[str, Any] | None = transport_error
        elif data is None:
            err = {"code": "execution_error", "message": "tool returned no structured result"}
        elif data.get("ok") is False:
            err = data.get("error")
        else:
            err = None
        self.step_count += 1
        self.step_log.append(
            {
                "step": self.step_count,
                "tool": tool,
                "arguments": arguments,
                "ok": err is None,
                "error": err,
                "result": data,
            }
        )
        if err is None:
            self.consecutive_errors = 0
        else:
            self.consecutive_errors += 1
            self.error_count += 1
        return err is not None

    def after_step(self) -> str | None:
        """Finalize if the episode just ended; returns the termination reason."""
        if self.submitted is not None:
            return self.finalize(False, "submit")
        if self.consecutive_errors >= self.max_consecutive_errors:
            return self.finalize(True, "error_limit")
        if self.step_count >= self.max_steps:
            return self.finalize(True, "step_limit")
        return None

    def finalize(self, truncated: bool, why: str) -> str:
        status = self.submitted[0] if self.submitted else None
        steps = [e["tool"] for e in self.step_log if e["tool"] != "submit"]
        mutating = next((i for i, t in enumerate(steps) if t not in READ_TOOLS), None)
        read_first = (
            None
            if mutating is None and not steps
            else any(
                t in READ_TOOLS for t in steps[: mutating if mutating is not None else len(steps)]
            )
        )
        self.score = score_episode(
            self.task,
            self.initial,
            self.backend.session,
            status,
            self.error_count,
            truncated,
            calls=len(steps),
            read_first=read_first,
            rubric=self.rubric,
        )
        self.done, self.termination = True, why
        return why

    def result_metadata(self) -> dict[str, Any]:
        assert self.score is not None
        return {
            "score": self.score.as_dict(),
            "facts": self.score.facts.as_dict() if self.score.facts else None,
            "success": self.score.success,
            "termination": self.termination,
            "truncated": self.score.truncated,
            "failed_checks": self.score.failed_checks,
            "step_log": self.step_log,
            "state_hash": self.backend.session.state_hash(),
        }

    def run(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Direct (non-MCP) path: execute a tool call and do the bookkeeping."""
        if self.done:
            raise RuntimeError("episode is over")
        if tool == "submit":
            data = self.on_submit(
                str(arguments.get("status", "done")), str(arguments.get("reason", ""))
            )
        else:
            data = self.backend.call(tool, arguments)
        self.record(tool, arguments, data)
        self.after_step()
        return data
