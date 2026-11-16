"""Throughput and latency benchmark: simulator only, full environment in-process, and over
WebSocket with N concurrent clients.

uv run python scripts/benchmark.py [--url http://localhost:8000] [--clients 8] [--json out.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from typing import Any

from openenv.core.env_server.mcp_types import CallToolAction

from openenv_daw.backends import SimBackend
from openenv_daw.env.client import DawEnv
from openenv_daw.env.environment import DawEnvironment
from openenv_daw.schema import Session
from openenv_daw.tasks.generators import families, generate


def tasks(n: int):
    fams = list(families())
    return [generate(fams[i % len(fams)], i) for i in range(n)]


def pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))] * 1000 if s else 0.0  # ms


def summary(
    steps: int, wall: float, step_lat: list[float], reset_lat: list[float]
) -> dict[str, Any]:
    return {
        "steps": steps,
        "steps_per_s": round(steps / wall, 1) if wall else 0.0,
        "step_ms_p50": round(pct(step_lat, 0.5), 3),
        "step_ms_p95": round(pct(step_lat, 0.95), 3),
        "reset_ms_p50": round(pct(reset_lat, 0.5), 3),
        "reset_ms_p95": round(pct(reset_lat, 0.95), 3),
    }


def bench_backend(n: int = 300) -> dict[str, Any]:
    ts = tasks(n)  # generation is not part of the measurement
    step_lat: list[float] = []
    reset_lat: list[float] = []
    t0 = time.perf_counter()
    for t in ts:
        r0 = time.perf_counter()
        b = SimBackend()
        if t.initial_state:
            b.load(Session.restore(t.initial_state))
        reset_lat.append(time.perf_counter() - r0)
        for st in t.oracle[:-1]:
            s0 = time.perf_counter()
            b.call(st.tool, st.args)
            step_lat.append(time.perf_counter() - s0)
    return summary(len(step_lat), time.perf_counter() - t0, step_lat, reset_lat)


async def bench_env_async(n: int = 100) -> dict[str, Any]:
    env = DawEnvironment()
    ts = tasks(n)
    step_lat: list[float] = []
    reset_lat: list[float] = []
    for t in ts[:5]:  # warmup
        env.reset(task_id=t.task_id)
    t0 = time.perf_counter()
    for t in ts:
        r0 = time.perf_counter()
        env.reset(task_id=t.task_id)
        reset_lat.append(time.perf_counter() - r0)
        for st in t.oracle:
            s0 = time.perf_counter()
            await env.step_async(CallToolAction(tool_name=st.tool, arguments=st.args))
            step_lat.append(time.perf_counter() - s0)
    return summary(len(step_lat), time.perf_counter() - t0, step_lat, reset_lat)


async def _ws_client(url: str, ts: list, step_lat: list[float], reset_lat: list[float]) -> None:
    async with DawEnv(base_url=url) as env:
        for t in ts:
            r0 = time.perf_counter()
            await env.reset(task_id=t.task_id)
            reset_lat.append(time.perf_counter() - r0)
            for st in t.oracle:
                s0 = time.perf_counter()
                await env.call(st.tool, **st.args)
                step_lat.append(time.perf_counter() - s0)


async def bench_ws(url: str, n: int = 100, clients: int = 1) -> dict[str, Any]:
    """``n`` episodes per client, ``clients`` concurrent WebSocket sessions."""
    ts = tasks(n)
    step_lat: list[float] = []
    reset_lat: list[float] = []
    t0 = time.perf_counter()
    await asyncio.gather(*(_ws_client(url, ts, step_lat, reset_lat) for _ in range(clients)))
    out = summary(len(step_lat), time.perf_counter() - t0, step_lat, reset_lat)
    out["clients"] = clients
    out["mean_step_ms"] = round(statistics.fmean(step_lat) * 1000, 3) if step_lat else 0.0
    return out


async def main(url: str | None, clients: int, out: str | None) -> None:
    results: dict[str, Any] = {"simulator_ops": bench_backend()}
    results["environment_in_process"] = await bench_env_async()
    if url:
        results["websocket_1_client"] = await bench_ws(url)
        if clients > 1:
            results[f"websocket_{clients}_clients"] = await bench_ws(url, clients=clients)
    for name, r in results.items():
        print(
            f"{name:28s} {r['steps_per_s']:>9.0f} steps/s  step p50/p95 "
            f"{r['step_ms_p50']:.2f}/{r['step_ms_p95']:.2f} ms  reset p50/p95 "
            f"{r['reset_ms_p50']:.2f}/{r['reset_ms_p95']:.2f} ms"
        )
    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None)
    ap.add_argument("--clients", type=int, default=1, help="concurrent WebSocket sessions")
    ap.add_argument("--json", default=None, help="write results to this file")
    a = ap.parse_args()
    asyncio.run(main(a.url, a.clients, a.json))
