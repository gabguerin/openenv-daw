"""Milestone 1 check: replay every task's oracle solution against a running server over
WebSocket. Every episode must score 1.0, and a second replay must give an identical state hash.

    uv run python scripts/oracle_rollout.py --url http://localhost:8000 [--train 20]
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from openenv_daw.client import DawEnv
from openenv_daw.server.tasks.generators import families, generate
from openenv_daw.server.tasks.store import load_heldout


async def replay(env: DawEnv, task) -> tuple[float | None, str, str]:
    await env.reset(task_id=task.task_id)
    res = None
    for st in task.oracle:
        res = await env.call(st.tool, **st.args)
        if st.tool != "submit" and res.done:
            raise RuntimeError(f"{task.task_id}: ended early at {st.tool}: {res.metadata}")
    assert res is not None and res.done
    return res.reward, res.metadata["state_hash"], task.intent


async def main(url: str, n_train: int) -> int:
    tasks = [generate(f, s, "train") for f in families() for s in range(n_train)]
    tasks += load_heldout()
    bad = 0
    async with DawEnv(base_url=url) as env:
        for t in tasks:
            r1, h1, _ = await replay(env, t)
            r2, h2, _ = await replay(env, t)
            ok = r1 == 1.0 and r2 == 1.0 and h1 == h2
            bad += not ok
            if not ok:
                print(f"FAIL {t.task_id}: {t.intent!r} rewards={r1},{r2} hashes_equal={h1 == h2}")
    print(f"{len(tasks) - bad}/{len(tasks)} tasks: oracle reward 1.0 with identical replay hash")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--train", type=int, default=20, help="train seeds per family")
    a = ap.parse_args()
    sys.exit(asyncio.run(main(a.url, a.train)))
