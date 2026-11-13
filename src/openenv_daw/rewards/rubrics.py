"""Terminal reward: critical gate, then fraction of checks passed minus penalties.

``score_episode`` is a pure function; the rubric classes compose it with openenv's
``Sequential`` so each component is inspectable (``named_rubrics``) and replaceable."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from openenv.core.rubrics import Rubric, Sequential

from openenv_daw.schema import Session
from openenv_daw.tasks.model import Task

from .checks import run_check
from .damage import damage_violations, extraneous_violations

ERROR_PENALTY = 0.02
ERROR_PENALTY_CAP = 0.2
DAMAGE_PENALTY = 0.15
DAMAGE_PENALTY_CAP = 0.5


@dataclass
class EpisodeScore:
    reward: float
    success: bool
    gate_passed: bool
    fraction: float
    check_results: list[bool]
    failed_checks: list[str]
    critical_failed: list[str]
    damage: list[str]
    damage_penalty: float
    error_count: int
    error_penalty: float
    status: str | None
    truncated: bool
    flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_episode(
    task: Task,
    initial: Session,
    final: Session,
    status: str | None,
    error_count: int,
    truncated: bool,
) -> EpisodeScore:
    results = [run_check(c.kind, final, initial, c.args) for c in task.checks]
    failed = [c.kind for c, ok in zip(task.checks, results, strict=True) if not ok]
    critical_failed = [
        c.kind for c, ok in zip(task.checks, results, strict=True) if c.critical and not ok
    ]
    rest = [ok for c, ok in zip(task.checks, results, strict=True) if not c.critical]
    fraction = sum(rest) / len(rest) if rest else 1.0
    violations = damage_violations(initial, final, task.touchable) + extraneous_violations(
        initial, final, task
    )
    dmg_pen = min(DAMAGE_PENALTY_CAP, DAMAGE_PENALTY * len(violations))
    err_pen = min(ERROR_PENALTY_CAP, ERROR_PENALTY * error_count)
    # ``submit`` is required: its status is the answer to feasibility, so finishing the work
    # without submitting, refusing a feasible task, or attempting an infeasible one scores 0.
    # ``fraction`` is still reported for diagnostics.
    status_wrong = status != task.expected_status
    gate = not critical_failed and not status_wrong
    reward = max(0.0, fraction - dmg_pen - err_pen) if gate else 0.0
    success = gate and all(results) and not violations and status == task.expected_status
    flags = []
    if truncated:
        flags.append("truncated")
    if status is None:
        flags.append("no_submit")
    return EpisodeScore(
        reward=round(reward, 6),
        success=success,
        gate_passed=gate,
        fraction=fraction,
        check_results=results,
        failed_checks=failed,
        critical_failed=critical_failed,
        damage=violations,
        damage_penalty=dmg_pen,
        error_count=error_count,
        error_penalty=err_pen,
        status=status,
        truncated=truncated,
        flags=flags,
    )


def _score(observation: Any) -> dict[str, Any] | None:
    return observation.metadata.get("score") if getattr(observation, "done", False) else None


class CriticalGate(Rubric):
    """1.0 when no critical check failed and the submit status is acceptable, else 0.0."""

    def forward(self, action: Any, observation: Any) -> float:
        s = _score(observation)
        if s is None:
            return 0.0
        return 1.0 if s["gate_passed"] else 0.0


class NetScore(Rubric):
    """Fraction of non-critical checks passed, minus damage and error penalties."""

    def forward(self, action: Any, observation: Any) -> float:
        s = _score(observation)
        if s is None:
            return 0.0
        return max(0.0, s["fraction"] - s["damage_penalty"] - s["error_penalty"])


class DawRubric(Sequential):
    def __init__(self) -> None:
        super().__init__(CriticalGate(), NetScore())
