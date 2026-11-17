"""Terminal reward: critical gate, then fraction of checks passed minus penalties, then a soft
musicality blend. ``score_episode`` is a pure function; ``DawRubric`` exposes it to OpenEnv."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from openenv.core.rubrics import Rubric
from openenv_daw.server.sim.session import Session
from openenv_daw.server.tasks.model import Task

from .checks import run_check
from .damage import damage_violations, extraneous_violations
from .musicality import musicality

ERROR_PENALTY = 0.02
ERROR_PENALTY_CAP = 0.2
DAMAGE_PENALTY = 0.15
DAMAGE_PENALTY_CAP = 0.5
MUSIC_WEIGHT = 0.2  # share of the reward that musicality can earn (once the gate passed)
MUSIC_FULL = 0.8  # musicality at which the bonus saturates (the oracle reaches it)


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
    musicality: float | None = None  # raw heuristic score; ``None`` for non-musical tasks
    music_detail: dict[str, Any] = field(default_factory=dict)
    # diagnostics, never part of the reward
    calls: int = 0
    oracle_len: int = 0
    read_first: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_episode(
    task: Task,
    initial: Session,
    final: Session,
    status: str | None,
    error_count: int,
    truncated: bool,
    *,
    calls: int = 0,
    read_first: bool | None = None,
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
    net = max(0.0, fraction - dmg_pen - err_pen)
    music, music_detail = (None, {}) if status_wrong else musicality(task.music, final)
    w = MUSIC_WEIGHT
    if gate and music is not None:
        # soft bonus: musicality earns up to ``w`` of the reward, saturating at MUSIC_FULL
        reward = (1 - w) * net + w * min(1.0, music / MUSIC_FULL)
    else:
        reward = net if gate else 0.0
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
        musicality=None if music is None else round(music, 4),
        music_detail=music_detail,
        calls=calls,
        oracle_len=max(len(task.oracle) - 1, 0),
        read_first=read_first,
    )


class DawRubric(Rubric):
    """The episode reward, read from the score the environment attaches to the final step."""

    def forward(self, action: Any, observation: Any) -> float:
        if not getattr(observation, "done", False):
            return 0.0
        score = observation.metadata.get("score")
        return float(score["reward"]) if score else 0.0

    def validation_config(self) -> dict[str, Any]:
        return {"music_weight": MUSIC_WEIGHT, "music_full": MUSIC_FULL}
