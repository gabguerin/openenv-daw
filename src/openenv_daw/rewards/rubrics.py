"""Terminal reward as an OpenEnv rubric tree.

``measure_episode`` is a pure function that records facts about a finished episode (check
results, damage, extra objects, errors, musicality) and does no reward arithmetic. ``DawRubric``
turns those facts into the reward, and every consumer (the MCP server, the TRL wrapper, the web
UI) reads that one number. Every component is inspectable through ``named_rubrics()``.

Tree (all leaves run on the terminal step)::

    DawRubric
    |- status    Gate(SubmitStatus)       wrong or missing submit status -> 0
    |- safety    Gate(NoDamage)           changing an untouchable object -> 0
    |- critical  Gate(CriticalChecks)     a failed critical check -> 0 (shaped mode: small credit)
    `- quality   MusicBlend
        |- correctness  Correctness = OptionalChecks x NoExtraneous x ErrorFree
        `- music        Musicality
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from openenv.core.env_server.types import Observation
from openenv.core.rubrics import Gate, Rubric

from openenv_daw.schema import Session
from openenv_daw.tasks.model import Task

from .checks import run_check
from .damage import damage_violations, extraneous_violations
from .musicality import musicality

ERROR_PENALTY = 0.02
ERROR_PENALTY_CAP = 0.2
EXTRANEOUS_PENALTY = 0.15
EXTRANEOUS_PENALTY_CAP = 0.5
SHAPED_SCALE = 0.1  # reward cap of the "shaped" mode
MUSIC_WEIGHT = 0.2  # share of the reward that musicality can earn (once the gates passed)
MUSIC_FULL = 0.8  # musicality at which the bonus saturates (the oracle reaches it)


def music_weight() -> float:
    """Share of the reward that is musicality: ``OPENENV_DAW_MUSIC_WEIGHT`` (0 disables it)."""
    return float(os.getenv("OPENENV_DAW_MUSIC_WEIGHT", MUSIC_WEIGHT))


def default_reward_mode() -> str:
    """``strict`` (default) or ``shaped``: ``OPENENV_DAW_REWARD_MODE``."""
    return os.getenv("OPENENV_DAW_REWARD_MODE", "strict").lower()


def extraneous_penalty(n: int) -> float:
    return min(EXTRANEOUS_PENALTY_CAP, EXTRANEOUS_PENALTY * n)


def error_penalty(n: int) -> float:
    return min(ERROR_PENALTY_CAP, ERROR_PENALTY * n)


# --- measurement -----------------------------------------------------------------------------


@dataclass
class EpisodeFacts:
    """What happened in an episode. Rewards are derived from these by ``DawRubric``."""

    check_kinds: list[str]
    check_results: list[bool]
    check_critical: list[bool]
    damage: list[str]  # untouchable pre-existing objects that changed
    extraneous: list[str]  # objects the task did not ask for
    error_count: int
    status: str | None
    expected_status: str
    musicality: float | None  # raw heuristic score; ``None`` for non-musical tasks
    music_detail: dict[str, Any] = field(default_factory=dict)

    @property
    def status_ok(self) -> bool:
        return self.status == self.expected_status

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_episode(
    task: Task, initial: Session, final: Session, status: str | None, error_count: int
) -> EpisodeFacts:
    results = [run_check(c.kind, final, initial, c.args) for c in task.checks]
    status_wrong = status != task.expected_status
    music, detail = (None, {}) if status_wrong else musicality(task.music, final)
    return EpisodeFacts(
        check_kinds=[c.kind for c in task.checks],
        check_results=results,
        check_critical=[c.critical for c in task.checks],
        damage=damage_violations(initial, final, task.touchable),
        extraneous=extraneous_violations(initial, final, task),
        error_count=error_count,
        status=status,
        expected_status=task.expected_status,
        musicality=music,
        music_detail=detail,
    )


# --- rubric tree -----------------------------------------------------------------------------


def _facts(observation: Any) -> dict[str, Any] | None:
    if not getattr(observation, "done", False):
        return None
    return (getattr(observation, "metadata", None) or {}).get("facts")


def _fraction(facts: dict[str, Any], critical: bool) -> float:
    picked = [
        ok
        for ok, c in zip(facts["check_results"], facts["check_critical"], strict=True)
        if c == critical
    ]
    return sum(picked) / len(picked) if picked else 1.0


class _Leaf(Rubric):
    """A rubric on the terminal facts; 0.0 until the episode is done. Its configuration is
    public, so ``openenv validate`` can introspect it."""

    def validation_config(self) -> dict[str, Any]:
        return {}

    def score(self, facts: dict[str, Any]) -> float:
        raise NotImplementedError

    def forward(self, action: Any, observation: Any) -> float:
        facts = _facts(observation)
        return 0.0 if facts is None else float(self.score(facts))


class SubmitStatus(_Leaf):
    """1.0 when the submitted status is the expected one (no submit scores 0)."""

    def score(self, facts: dict[str, Any]) -> float:
        return 1.0 if facts["status"] == facts["expected_status"] else 0.0


class NoDamage(_Leaf):
    """1.0 when no pre-existing object the task did not declare touchable was changed."""

    def score(self, facts: dict[str, Any]) -> float:
        return 0.0 if facts["damage"] else 1.0


class CriticalChecks(_Leaf):
    """Fraction of critical checks passed (1.0 when there are none)."""

    def score(self, facts: dict[str, Any]) -> float:
        return _fraction(facts, True)


class OptionalChecks(_Leaf):
    """Fraction of non-critical checks passed (1.0 when there are none)."""

    def score(self, facts: dict[str, Any]) -> float:
        return _fraction(facts, False)


class NoExtraneous(_Leaf):
    """1 - 0.15 per extra object (new track, device, solo, loud fader), capped at 0.5 off."""

    def score(self, facts: dict[str, Any]) -> float:
        return 1.0 - extraneous_penalty(len(facts["extraneous"]))


class ErrorFree(_Leaf):
    """1 - 0.02 per tool error, capped at 0.2 off."""

    def score(self, facts: dict[str, Any]) -> float:
        return 1.0 - error_penalty(facts["error_count"])


class Musicality(_Leaf):
    """Musicality over ``MUSIC_FULL``, capped at 1. Tasks that are not about music score 1.0."""

    def score(self, facts: dict[str, Any]) -> float:
        m = facts["musicality"]
        return 1.0 if m is None else min(1.0, m / MUSIC_FULL)


class Correctness(Rubric):
    """Optional checks, scaled down by extra objects and tool errors."""

    def __init__(self) -> None:
        super().__init__()
        self.optional = OptionalChecks()
        self.extraneous = NoExtraneous()
        self.errors = ErrorFree()

    def validation_config(self) -> dict[str, Any]:
        return {}

    def forward(self, action: Any, observation: Any) -> float:
        return (
            self.optional(action, observation)
            * self.extraneous(action, observation)
            * self.errors(action, observation)
        )


class MusicBlend(Rubric):
    """``(1 - W) * correctness + W * musicality`` for musical tasks, else correctness."""

    def __init__(self) -> None:
        super().__init__()
        self.correctness = Correctness()
        self.music = Musicality()

    def validation_config(self) -> dict[str, Any]:
        return {"music_weight": music_weight(), "music_full": MUSIC_FULL}

    def forward(self, action: Any, observation: Any) -> float:
        correct = self.correctness(action, observation)
        music = self.music(action, observation)
        facts = _facts(observation)
        w = music_weight()
        if facts is None or facts["musicality"] is None or w <= 0:
            return correct
        return (1 - w) * correct + w * music


class DawRubric(Rubric):
    """The episode reward. Hard constraints are ``Gate``s; extras and errors are multipliers.

    ``reward_mode`` is ``strict`` (default) or ``shaped``; ``None`` reads it at every call from
    ``OPENENV_DAW_REWARD_MODE``. In shaped mode a right submit status and no damage with failed
    critical checks earns ``SHAPED_SCALE`` times the fraction of critical checks passed."""

    def __init__(self, reward_mode: str | None = None) -> None:
        super().__init__()
        self.reward_mode = reward_mode
        self.status = Gate(SubmitStatus())
        self.safety = Gate(NoDamage())
        self.critical = Gate(CriticalChecks())
        self.quality = MusicBlend()

    def validation_config(self) -> dict[str, Any]:
        return {
            "reward_mode": self.reward_mode or default_reward_mode(),
            "shaped_scale": SHAPED_SCALE,
        }

    def forward(self, action: Any, observation: Any) -> float:
        if _facts(observation) is None:
            return 0.0
        # every child runs, so ``named_rubrics()`` shows each component's last score
        status = self.status(action, observation)
        safety = self.safety(action, observation)
        critical = self.critical(action, observation)
        quality = self.quality(action, observation)
        if status == 0.0 or safety == 0.0:
            return 0.0
        if critical == 0.0:
            if (self.reward_mode or default_reward_mode()) == "shaped":
                raw = self.critical.rubric.last_score or 0.0
                return round(SHAPED_SCALE * raw, 6)
            return 0.0
        return round(quality, 6)

    def components(self) -> dict[str, float | None]:
        return {name: r.last_score for name, r in self.named_rubrics()}


# --- episode score ---------------------------------------------------------------------------


@dataclass
class EpisodeScore:
    reward: float
    success: bool
    gate_passed: bool  # right submit status, no damage and every critical check passed
    fraction: float  # share of non-critical checks passed
    check_results: list[bool]
    failed_checks: list[str]
    critical_failed: list[str]
    damage: list[str]  # changed untouchable objects: reward 0
    extraneous: list[str]  # extra objects: each costs 15% of the correctness score (cap 50%)
    extraneous_penalty: float
    error_count: int
    error_penalty: float
    status: str | None
    truncated: bool
    flags: list[str] = field(default_factory=list)
    musicality: float | None = None  # raw heuristic score; ``None`` for non-musical tasks
    music_detail: dict[str, Any] = field(default_factory=dict)
    components: dict[str, float | None] = field(default_factory=dict)  # rubric leaf scores
    # diagnostics, never part of the reward
    calls: int = 0
    oracle_len: int = 0
    read_first: bool | None = None
    facts: EpisodeFacts | None = field(default=None, repr=False)  # not in ``as_dict``

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("facts")
        return d


def score_episode(
    task: Task,
    initial: Session,
    final: Session,
    status: str | None,
    error_count: int,
    truncated: bool,
    *,
    reward_mode: str | None = None,
    calls: int = 0,
    read_first: bool | None = None,
    rubric: DawRubric | None = None,
) -> EpisodeScore:
    facts = measure_episode(task, initial, final, status, error_count)
    rubric = rubric or DawRubric(reward_mode)
    reward = rubric(None, Observation(done=True, metadata={"facts": facts.as_dict()}))
    assert isinstance(reward, float)
    results = facts.check_results
    failed = [k for k, ok in zip(facts.check_kinds, results, strict=True) if not ok]
    critical_failed = [
        k
        for k, ok, c in zip(facts.check_kinds, results, facts.check_critical, strict=True)
        if c and not ok
    ]
    gate = facts.status_ok and not facts.damage and not critical_failed
    success = gate and all(results) and not facts.extraneous
    flags = []
    if truncated:
        flags.append("truncated")
    if status is None:
        flags.append("no_submit")
    return EpisodeScore(
        reward=reward,
        success=success,
        gate_passed=gate,
        fraction=_fraction(facts.as_dict(), False),
        check_results=results,
        failed_checks=failed,
        critical_failed=critical_failed,
        damage=facts.damage,
        extraneous=facts.extraneous,
        extraneous_penalty=extraneous_penalty(len(facts.extraneous)),
        error_count=error_count,
        error_penalty=error_penalty(error_count),
        status=status,
        truncated=truncated,
        flags=flags,
        musicality=None if facts.musicality is None else round(facts.musicality, 4),
        music_detail=facts.music_detail,
        components=rubric.components(),
        facts=facts,
        calls=calls,
        oracle_len=max(len(task.oracle) - 1, 0),
        read_first=read_first,
    )
