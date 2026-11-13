"""Steps/sec benchmark: in-process (simulator only and full environment) and over WebSocket.

    uv run python scripts/benchmark.py [--url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import asyncio
import time

from openenv.core.env_server.mcp_types import CallToolAction

from openenv_daw.backends import SimBackend
from openenv_daw.env.client import DawEnv
from openenv_daw.env.environment import DawEnvironment
from openenv_daw.tasks.generators import families, generate


def tasks(n: int):
    fams = list(families())
    return [generate(fams[i % len(fams)], i) for i in range(n)]


def bench_backend(n: int = 300) -> float:
    steps = 0
    t0 = time.perf_counter()
    for t in tasks(n):
        b = SimBackend()
        for st in t.oracle[:-1]:
            b.call(st.tool, st.args)
            steps += 1
    return steps / (time.perf_counter() - t0)


async def bench_env_async(n: int = 100) -> float:
    env = DawEnvironment()
    steps = 0
    t0 = time.perf_counter()
    for t in tasks(n):
        env.reset(task_id=t.task_id)
        for st in t.oracle:
            await env.step_async(CallToolAction(tool_name=st.tool, arguments=st.args))
            steps += 1
    return steps / (time.perf_counter() - t0)


async def bench_ws(url: str, n: int = 100) -> float:
    steps = 0
    async with DawEnv(base_url=url) as env:
        t0 = time.perf_counter()
        for t in tasks(n):
            await env.reset(task_id=t.task_id)
            for st in t.oracle:
                await env.call(st.tool, **st.args)
                steps += 1
        return steps / (time.perf_counter() - t0)


async def main(url: str | None) -> None:
    print(f"simulator only (ops):          {bench_backend():10.0f} steps/s")
    print(f"environment, in-process async: {await bench_env_async():10.0f} steps/s")
    if url:
        print(f"environment over WebSocket:    {await bench_ws(url):10.0f} steps/s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    asyncio.run(main(ap.parse_args().url))
