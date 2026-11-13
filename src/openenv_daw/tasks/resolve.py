"""Profile-dependent feasibility. Stored tasks (and the heldout JSONL) are profile-neutral:
they declare ``requires`` (tool names and capability tags) and this helper decides, per profile,
whether the task is feasible. An uncovered requirement turns it into a "blocked" task: nothing may
change and the correct answer is ``submit("blocked")``."""

from __future__ import annotations

from openenv_daw.sim.profiles import DEFAULT_PROFILE, covers

from .model import Check, OracleStep, Task

SUBMIT_BLOCKED_STEP = OracleStep(
    tool="submit", args={"status": "blocked", "reason": "not possible with the available tools"}
)


def blocked_for_profile(task: Task, profile: str = DEFAULT_PROFILE) -> bool:
    return not covers(profile, task.requires)


def resolve_for_profile(task: Task, profile: str = DEFAULT_PROFILE) -> Task:
    if not blocked_for_profile(task, profile):
        return task
    tags = task.tags if "blocked" in task.tags else [*task.tags, "blocked"]
    return task.model_copy(
        update={
            "expected_status": "blocked",
            "checks": [Check(kind="state_unchanged", critical=True)],
            "oracle": [SUBMIT_BLOCKED_STEP],
            "touchable": [],
            "allowed_new": [],
            "tags": tags,
        }
    )
