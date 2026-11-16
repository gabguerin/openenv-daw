"""Profile-dependent feasibility. Stored tasks (and the heldout JSONL) are profile-neutral:
they declare ``requires`` (tool names, capability tags and ``param:<device>.<param>`` entries) and
this helper decides, per profile and per-episode parameter mask, whether the task is feasible. An
uncovered requirement turns it into a "blocked" task: nothing may change and the correct answer is
``submit("blocked")``."""

from __future__ import annotations

import random

from openenv_daw.sim.catalog import load_catalog, load_library, unavailable_items
from openenv_daw.sim.profiles import DEFAULT_PROFILE, capabilities_for, covers, tools_for

from .model import Check, OracleStep, Task

SUBMIT_BLOCKED_STEP = OracleStep(
    tool="submit", args={"status": "blocked", "reason": "not possible with the available tools"}
)


def required_params(task: Task) -> list[str]:
    """``param:<device>.<param>`` requirements: declared ones plus those the oracle uses."""
    out = [r for r in task.requires if r.startswith("param:")]
    cat = load_catalog()
    for step in task.oracle:
        if step.tool == "set_device_param" and step.args.get("device") in cat:
            key = f"param:{step.args['device']}.{str(step.args.get('param', '')).lower()}"
            if key not in out:
                out.append(key)
    return out


def _alternative(task: Task, wanted: str, unavailable: frozenset[str]) -> str | None:
    """A loadable library item for an oracle load that the DAW cannot do, when the request only
    described the sound (a ``has_device`` check with ``tags_any``): any item carrying one of the
    tags, preferring the same kind of device. None when the request named the item."""
    tags: set[str] = set()
    for c in task.checks:
        if c.kind == "has_device":
            tags |= set(c.args.get("tags_any", ()))
    if not tags:
        return None
    lib = load_library()
    orig = lib[wanted]
    pool = [
        i
        for i in lib.values()
        if i.id not in unavailable and i.role == orig.role and tags & set(i.tags)
    ]
    pool.sort(key=lambda i: (i.base != orig.base, i.kind != orig.kind, i.id))
    return pool[0].id if pool else None


def _reroute_items(task: Task, profile: str, mask: frozenset[str]) -> Task | None:
    """The task with oracle loads of unavailable library items replaced by available ones, the
    task itself when nothing changes, None when an item is needed but cannot be loaded."""
    unavailable = unavailable_items(tools_for(profile) | capabilities_for(profile), mask)
    lib = load_library()
    steps, changed = [], False
    for st in task.oracle:
        dev = st.args.get("device")
        if st.tool == "insert_device" and dev in lib and dev in unavailable:
            alt = _alternative(task, dev, unavailable)
            if alt is None:
                return None
            st = OracleStep(tool=st.tool, args={**st.args, "device": alt})
            changed = True
        steps.append(st)
    return task.model_copy(update={"oracle": steps}) if changed else task


def blocked_for_profile(
    task: Task, profile: str = DEFAULT_PROFILE, mask: frozenset[str] = frozenset()
) -> bool:
    if not covers(profile, [*task.requires, *required_params(task)], mask):
        return True
    return _reroute_items(task, profile, mask) is None


def resolve_for_profile(
    task: Task, profile: str = DEFAULT_PROFILE, mask: frozenset[str] = frozenset()
) -> Task:
    if covers(profile, [*task.requires, *required_params(task)], mask):
        routed = _reroute_items(task, profile, mask)
        if routed is not None:
            return routed
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


def sample_param_mask(task: Task, seed: int | str, prob: float) -> frozenset[str]:
    """Training augmentation: with probability ``prob`` one parameter is unavailable. It is drawn
    from the parameters the task's oracle uses when there are any (so the episode really tests
    the 'check supported, then submit blocked' behaviour), else from the whole catalog."""
    rng = random.Random(f"mask:{seed}:{task.task_id}")
    if prob <= 0 or rng.random() >= prob:
        return frozenset()
    used = [r[6:] for r in required_params(task)]
    pool = used or sorted(f"{d.id}.{p}" for d in load_catalog().values() for p in d.params)
    return frozenset({rng.choice(pool)})


def apply_param_mask(
    task: Task, profile: str, prob: float, seed: int | str
) -> tuple[Task, frozenset[str]]:
    """Mask (train tasks only) and re-resolve the task against the mask."""
    if task.split != "train" or prob <= 0:
        return task, frozenset()
    mask = sample_param_mask(task, seed, prob)
    return (resolve_for_profile(task, profile, mask) if mask else task), mask
