"""Episode bookkeeping shared by DawEnvironment (MCP path) and the TRL wrapper (direct path):
limits, verified step log and terminal scoring."""

from __future__ import annotations

from typing import Any

from openenv_daw.backends import SimBackend
from openenv_daw.rewards.rubrics import EpisodeScore, score_episode
from openenv_daw.schema import Session
from openenv_daw.tasks.model import Task


class Episode:
    def __init__(
        self, task: Task, profile: str, max_steps: int, max_consecutive_errors: int
    ) -> None:
        self.task = task
        self.max_steps, self.max_consecutive_errors = max_steps, max_consecutive_errors
        self.initial = Session.restore(task.initial_state) if task.initial_state else Session()
        self.backend = SimBackend(Session.restore(self.initial.snapshot()), profile)
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
        return {"ok": True, "status": status, "message": "Task submitted."}

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
        self.score = score_episode(
            self.task, self.initial, self.backend.session, status, self.error_count, truncated
        )
        self.done, self.termination = True, why
        return why

    def result_metadata(self) -> dict[str, Any]:
        assert self.score is not None
        return {
            "score": self.score.as_dict(),
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
