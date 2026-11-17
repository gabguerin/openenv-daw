"""Evaluation: run a policy over tasks, aggregate into a report (success with a Wilson interval,
breakdowns, status confusion matrix, pass@k) and gate on a pass fraction.

Built-in policies (``harness.scripted_model_step``) give the floors and the ceiling: ``oracle``,
``noop``, ``submit_only``, ``blocked`` and ``random``."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openenv.core.harness import ModelStep

from openenv_daw.backends import Backend
from openenv_daw.sim.profiles import DEFAULT_PROFILE
from openenv_daw.tasks.model import Task

GATE_FRACTION = 0.85  # a student must solve at least 85% of the benchmark


def gate_threshold(n: int) -> int:
    return math.ceil(GATE_FRACTION * n)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        by_family[r["task_id"].split("-", 1)[0]].append(bool(r["success"]))
    n = len(records)
    return {
        "episodes": n,
        "success": sum(bool(r["success"]) for r in records),
        "mean_reward": sum(r["reward"] for r in records) / n if n else 0.0,
        "by_family": {f: f"{sum(v)}/{len(v)}" for f, v in sorted(by_family.items())},
    }


def gate(records: list[dict[str, Any]], n_tasks: int) -> tuple[bool, str]:
    """Pass when the successes of one pass over ``n_tasks`` tasks reach ``ceil(0.85 * n)``."""
    solved = len({r["task_id"] for r in records if r["success"]})
    need = gate_threshold(n_tasks)
    return solved >= need, f"{solved}/{n_tasks} solved, need {need}"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate that at least one of ``k`` samples succeeds (``n`` drawn, ``c`` right)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_pow_k(n: int, c: int, k: int) -> float:
    """Estimate that all ``k`` samples succeed."""
    return math.comb(c, k) / math.comb(n, k) if c >= k else 0.0


def submitted_status(rec: dict[str, Any]) -> str:
    for e in reversed(rec.get("step_log") or []):
        if e["tool"] == "submit" and e["ok"]:
            return str(e["arguments"].get("status", "done"))
    return "none"


def _group(rows: list[tuple[dict[str, Any], Task]], key: Callable[[Task], list[str]]) -> dict:
    g: dict[str, list[bool]] = defaultdict(list)
    m: dict[str, list[float]] = defaultdict(list)
    for r, t in rows:
        for k in key(t):
            g[k].append(bool(r["success"]))
            if r.get("musicality") is not None:
                m[k].append(float(r["musicality"]))
    return {
        k: {"success": f"{sum(v)}/{len(v)}", "rate": round(sum(v) / len(v), 4)}
        | ({"musicality": round(sum(m[k]) / len(m[k]), 4)} if m[k] else {})
        for k, v in sorted(g.items())
    }


def build_report(
    records: list[dict[str, Any]], tasks: list[Task], policy: str = ""
) -> dict[str, Any]:
    by_id = {t.task_id: t for t in tasks}
    rows = [(r, by_id[r["task_id"]]) for r in records if r["task_id"] in by_id]
    n = len(rows)
    ok = sum(bool(r["success"]) for r, _ in rows)
    lo, hi = wilson(ok, n)
    confusion: Counter[str] = Counter(
        f"{t.expected_status}->{submitted_status(r)}" for r, t in rows
    )
    per_task: dict[str, list[bool]] = defaultdict(list)
    for r, _ in rows:
        per_task[r["task_id"]].append(bool(r["success"]))
    k = min((len(v) for v in per_task.values()), default=1)
    ks = {
        f"pass@{k}": round(
            sum(pass_at_k(len(v), sum(v), k) for v in per_task.values()) / max(len(per_task), 1), 4
        ),
        f"pass^{k}": round(
            sum(pass_pow_k(len(v), sum(v), k) for v in per_task.values()) / max(len(per_task), 1), 4
        ),
    }
    calls = [len(r.get("actions") or []) for r, _ in rows]
    oracle_len = [max(len(t.oracle), 1) for _, t in rows]
    return {
        "policy": policy,
        "episodes": n,
        "tasks": len(per_task),
        "success": ok,
        "success_rate": round(ok / n, 4) if n else 0.0,
        "ci95": [round(lo, 4), round(hi, 4)],
        "mean_reward": round(sum(r["reward"] for r, _ in rows) / n, 4) if n else 0.0,
        "mean_musicality": (
            round(sum(ms) / len(ms), 4)
            if (ms := [r["musicality"] for r, _ in rows if r.get("musicality") is not None])
            else None
        ),
        **ks,
        "by_family": _group(rows, lambda t: [t.family]),
        "by_tag": _group(rows, lambda t: t.tags or ["untagged"]),
        "by_expected_status": _group(rows, lambda t: [t.expected_status]),
        "status_confusion": dict(sorted(confusion.items())),
        "terminations": dict(Counter(r.get("termination", "?") for r, _ in rows)),
        "mean_calls": round(sum(calls) / n, 2) if n else 0.0,
        "mean_calls_vs_oracle": round(
            sum(c / o for c, o in zip(calls, oracle_len, strict=True)) / n, 2
        )
        if n
        else 0.0,
        "mean_errors": round(sum(r.get("error_count") or 0 for r, _ in rows) / n, 3) if n else 0.0,
    }


def report_markdown(rep: dict[str, Any]) -> str:
    lo, hi = rep["ci95"]
    lines = [
        f"# Evaluation: {rep['policy'] or 'policy'}",
        "",
        f"- Episodes: {rep['episodes']} over {rep['tasks']} tasks",
        f"- Success: {rep['success']}/{rep['episodes']} = {rep['success_rate']:.1%} (95% CI {lo:.1%}-{hi:.1%})",
        f"- Mean reward: {rep['mean_reward']}",
    ]
    lines += [f"- {k}: {v}" for k, v in rep.items() if k.startswith("pass")]
    lines += [
        f"- Calls per episode: {rep['mean_calls']} ({rep['mean_calls_vs_oracle']}x oracle length), errors {rep['mean_errors']}",
        f"- Terminations: {rep['terminations']}",
        f"- Status (expected->submitted): {rep['status_confusion']}",
    ]
    for title, key in (
        ("Family", "by_family"),
        ("Tag", "by_tag"),
        ("Expected status", "by_expected_status"),
    ):
        lines += ["", f"| {title} | success | rate |", "|---|---|---|"]
        lines += [f"| {k} | {v['success']} | {v['rate']:.1%} |" for k, v in rep[key].items()]
    return "\n".join(lines) + "\n"


def load_task_set(spec: str) -> list[Task]:
    """``heldout`` (the default heldout), or the path of a JSONL file."""
    from openenv_daw.tasks.store import default_store, load_tasks

    return list(default_store().heldout) if spec == "heldout" else load_tasks(spec)


def evaluate(
    policy_name: str,
    tasks: list[Task],
    out_dir: Path,
    samples: int = 1,
    workers: int = 4,
    url: str | None = None,
    model_step: ModelStep | None = None,
    profile: str = DEFAULT_PROFILE,
    model: str | None = None,
    backend_factory: Callable[[str], Backend] | None = None,
) -> dict[str, Any]:
    """Run ``policy_name`` on ``tasks`` (stored in their profile-neutral form) as served by
    ``profile`` and write ``results.jsonl``, ``report.json`` and ``report.md`` into ``out_dir``.
    Returns the report."""
    from openenv_daw.harness import (
        DawSessionFactory,
        collect,
        flat,
        scripted_model_step,
        teacher_model_step,
    )
    from openenv_daw.tasks.resolve import resolve_for_profile
    from openenv_daw.tasks.store import TaskStore

    resolved = [resolve_for_profile(t, profile) for t in tasks]
    if model_step is None:
        if policy_name == "openai":
            model_step = teacher_model_step(model=model, temperature=0.0)
        else:
            model_step = scripted_model_step(policy_name, resolved)
    factory = DawSessionFactory(
        TaskStore(tasks), profile, backend_factory=backend_factory, base_url=url
    )
    records = collect(
        factory,
        model_step,
        tasks,
        out_dir,
        samples=samples,
        workers=workers,
        resume=False,
        metadata={"policy": policy_name},
    )
    rep = build_report([flat(r) for r in records], resolved, policy_name)
    rep["profile"] = profile
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2))
    (out_dir / "report.md").write_text(report_markdown(rep))
    return rep


DUAL_PROFILES = ("portable", "portable_plugins")


def evaluate_dual(
    policy_name: str,
    tasks: list[Task],
    out_dir: Path,
    samples: int = 1,
    workers: int = 4,
    url: str | None = None,
    model_step: ModelStep | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """The same stored tasks scored under ``portable`` (the headline: every DAW can do it) and
    ``portable_plugins`` (informational: a DAW that also loads third-party plugins). Each profile
    writes its own report into ``out_dir/<profile>``; the combined ``report.json`` is the headline
    report plus an ``informational`` entry."""
    reps = {
        p: evaluate(
            policy_name,
            tasks,
            out_dir / p,
            samples,
            workers,
            url,
            model_step,
            profile=p,
            model=model,
        )
        for p in DUAL_PROFILES
    }
    head = {**reps[DUAL_PROFILES[0]], "informational": {p: reps[p] for p in DUAL_PROFILES[1:]}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(head, indent=2))
    md = report_markdown(reps[DUAL_PROFILES[0]]) + "".join(
        f"\n## Informational: {p}\n\n" + report_markdown(reps[p]) for p in DUAL_PROFILES[1:]
    )
    (out_dir / "report.md").write_text(md)
    return head
