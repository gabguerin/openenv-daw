import pytest
from openenv_daw.server.tasks.store import default_store
from openenv_daw.training.harness import flat, read_results
from openenv_daw.training.report import (
    build_report,
    evaluate,
    load_task_set,
    pass_at_k,
    pass_pow_k,
    wilson,
)


def test_wilson_and_passk():
    lo, hi = wilson(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.61
    assert wilson(0, 0) == (0.0, 0.0)
    assert pass_at_k(4, 1, 1) == pytest.approx(0.25)
    assert pass_pow_k(4, 3, 3) == pytest.approx(0.25)
    assert pass_at_k(4, 4, 2) == 1.0


def test_oracle_ceiling_and_floors(tmp_path):
    tasks = load_task_set("heldout")
    oracle = evaluate("oracle", tasks, tmp_path / "o")
    assert oracle["success"] == len(tasks) and oracle["mean_reward"] == 1.0
    assert (tmp_path / "o" / "report.md").read_text().startswith("# Evaluation")
    floor = 0.1  # openenv.yaml validation.reward.floor_margin
    for name in ("noop", "submit_only", "blocked"):
        rep = evaluate(name, tasks, tmp_path / name)
        feasible = [t.task_id for t in tasks if t.expected_status == "done"]
        recs = [flat(r) for r in read_results(tmp_path / name)]
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
