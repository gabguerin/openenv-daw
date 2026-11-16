import json

import pytest

from openenv_daw.runner.report import (
    build_report,
    evaluate,
    load_task_set,
    pass_at_k,
    pass_pow_k,
    wilson,
)
from openenv_daw.tasks.resolve import resolve_for_profile
from openenv_daw.tasks.store import default_store


def test_wilson_and_passk():
    lo, hi = wilson(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.61
    assert wilson(0, 0) == (0.0, 0.0)
    assert pass_at_k(4, 1, 1) == pytest.approx(0.25)
    assert pass_pow_k(4, 3, 3) == pytest.approx(0.25)
    assert pass_at_k(4, 4, 2) == 1.0


async def test_oracle_ceiling_and_floors(tmp_path):
    tasks = load_task_set("heldout")
    oracle = await evaluate("oracle", tasks, tmp_path / "o")
    assert oracle["success"] == len(tasks) and oracle["mean_reward"] == 1.0
    assert (tmp_path / "o" / "report.md").read_text().startswith("# Evaluation")
    floor = 0.1  # openenv.yaml validation.reward.floor_margin
    for name in ("noop", "submit_only", "blocked"):
        rep = await evaluate(name, tasks, tmp_path / name)
        feasible = [
            t.task_id for t in tasks if resolve_for_profile(t, "portable").expected_status == "done"
        ]
        recs = [
            json.loads(x) for x in (tmp_path / name / "rollouts.jsonl").read_text().splitlines()
        ]
        got = [r["reward"] for r in recs if r["task_id"] in feasible]
        assert max(got) <= floor, name
        assert rep["success_rate"] <= 0.2
    assert default_store().heldout


def test_build_report_confusion():
    t = load_task_set("heldout")[:2]
    recs = [
        {"task_id": x.task_id, "success": False, "reward": 0.0, "actions": [], "step_log": []}
        for x in t
    ]
    rep = build_report(recs, t, "x")
    assert rep["status_confusion"] and rep["episodes"] == 2
