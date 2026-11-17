"""``python -m openenv_daw.runner``: ``collect`` (flags follow ``openenv collect``), ``export``
(collected episodes -> SFT rows) and ``eval`` (report against the heldout split)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openenv_daw.harness import (
    DawSessionFactory,
    collect,
    scripted_model_step,
    select_tasks,
    teacher_model_step,
)
from openenv_daw.sim.profiles import DEFAULT_PROFILE

from .sft_export import export_sft


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="openenv-daw-runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="run a teacher model on tasks and record the episodes")
    c.add_argument("--provider", default="openai", help="openai | anthropic | scripted")
    c.add_argument("--model")
    c.add_argument("--llm-endpoint", help="OpenAI-compatible server (vLLM, TGI, Ollama)")
    c.add_argument("--llm-port", type=int)
    c.add_argument("--base-url", help="deployed env server; default is in-process")
    c.add_argument("--split", default="train", choices=["train", "heldout"])
    c.add_argument("--family")
    c.add_argument("--start", type=int, default=0)
    c.add_argument("-n", "--num-episodes", type=int, default=20, help="number of tasks")
    c.add_argument("--samples", type=int, default=1, help="episodes per task")
    c.add_argument("--max-turns", type=int, default=40)
    c.add_argument("--max-tokens", type=int, default=1024)
    c.add_argument("--temperature", type=float, default=0.7)
    c.add_argument("--workers", "--concurrency", type=int, default=4)
    c.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    c.add_argument("--push-to-hub", help="dataset repo id 'user/name'")
    c.add_argument("--private", action="store_true")
    c.add_argument("-o", "--output-dir", default="outputs/rollouts")
    e = sub.add_parser("export", help="filter collected episodes into SFT JSONL")
    e.add_argument("rollouts", nargs="+", help="collect output dir(s)")
    e.add_argument("-o", "--out", default="sft.jsonl")
    e.add_argument("--min-musicality", type=float, default=0.6)
    v = sub.add_parser("eval", help="evaluate a policy and write report.json / report.md")
    v.add_argument("--tasks", default="heldout", help="'heldout' or a JSONL path")
    v.add_argument(
        "--policy",
        default="openai",
        choices=["openai", "oracle", "noop", "submit_only", "blocked", "random"],
    )
    v.add_argument("--model")
    v.add_argument("--samples", type=int, default=1)
    v.add_argument("--workers", "--concurrency", type=int, default=4)
    v.add_argument("--url")
    v.add_argument(
        "--profile", help="score under one profile (default: portable, plus portable_plugins)"
    )
    v.add_argument("--out", default="outputs/eval")
    a = ap.parse_args(argv)

    if a.cmd == "export":
        print(json.dumps(export_sft([Path(r) for r in a.rollouts], Path(a.out), a.min_musicality)))
        return
    if a.cmd == "eval":
        from .report import evaluate, evaluate_dual, load_task_set, report_markdown

        args = (a.policy, load_task_set(a.tasks), Path(a.out), a.samples, a.workers, a.url)
        kw = {"model": a.model}
        if a.profile or a.url:  # a server's profile is fixed
            rep = evaluate(*args, profile=a.profile or DEFAULT_PROFILE, **kw)
        else:
            rep = evaluate_dual(*args, **kw)
        print(report_markdown(rep))
        for p, info in rep.get("informational", {}).items():
            print(
                f"\ninformational ({p}): {info['success']}/{info['episodes']} = {info['success_rate']:.1%}"
            )
        return

    tasks = select_tasks(a.split, a.num_episodes, a.start, a.family)
    factory = DawSessionFactory(base_url=a.base_url)
    if a.provider == "scripted":
        step = scripted_model_step("oracle", tasks)
    else:
        step = teacher_model_step(
            provider=a.provider,
            model=a.model,
            llm_endpoint=a.llm_endpoint,
            llm_port=a.llm_port,
            temperature=a.temperature,
            max_tokens=a.max_tokens,
        )
    out = Path(a.output_dir)
    recs = collect(
        factory,
        step,
        tasks,
        out,
        samples=a.samples,
        workers=a.workers,
        max_turns=a.max_turns,
        resume=a.resume,
        metadata={"provider": a.provider, "model": a.model, "split": a.split},
    )
    ok = sum(r["verify_metrics"]["success"] for r in recs)
    print(f"{ok}/{len(recs)} successful episodes -> {out}", file=sys.stderr)
    if a.push_to_hub:
        from openenv.core.harness.collect import push_to_hf_hub

        print(push_to_hf_hub(output_dir=out, repo_id=a.push_to_hub, private=a.private))
