# openenv-daw

An [OpenEnv](https://github.com/openenv/OpenEnv) environment for DAW tool use ("train abstract,
deploy specific"). A lightweight JSON DAW state simulator sits behind a portable set of MCP tools
that map onto the scripting APIs of Ableton Live 12, REAPER 7, Bitwig and Ardour. No audio engine
is needed: the simulator is pure Python and deterministic.

## Design

- **Tools**: 15 portable tools plus `submit`, declared once as a FastMCP server
  (`openenv_daw.tools.registry`). Tool results are JSON; mutating tools return the affected
  object, errors are typed (`not_found`, `invalid_args`, `conflict`, `unsupported`).
- **Simulator**: linear timeline in beats, dB mixer, return tracks with implicit sends,
  1/16-beat grid snapping, stable ids, a canonical 7-device catalog
  (`src/openenv_daw/catalog/devices.yaml`) and capability profiles (`portable` default).
- **Environment** (`DawEnvironment`, an OpenEnv `MCPEnvironment`): one request per episode.
  The reset observation is the request text only; the agent must call `get_session`. Episodes end
  on `submit`, after 30 steps, or after 3 consecutive errors. A verified step log is kept.
- **Reward**: any failed critical check (or a wrong submit status) gives 0; otherwise the
  fraction of remaining checks passed, minus a penalty for changing untouchable objects
  (0.15 each, cap 0.5) and per tool error (0.02 each, cap 0.2). `metadata["success"]` is binary.
- **Tasks**: 5 seeded families (`session_setup`, `melody_in_key`, `four_on_floor_beat`,
  `reverb_send`, `mix_balance`) plus about 10% infeasible requests (`submit("blocked")`).
  Train tasks are generated from seeds; `heldout.jsonl` uses unseen templates and slot values.
  Every task carries an oracle solution.

## Run

```bash
uv sync --all-extras
uv run pytest
docker build -t openenv-daw . && docker run -p 8000:8000 openenv-daw
uv run python scripts/oracle_rollout.py --url http://localhost:8000
uv run python scripts/benchmark.py --url http://localhost:8000
```

```python
from openenv_daw.env.client import DawEnv

async with DawEnv(base_url="http://localhost:8000") as env:
    r = await env.reset(split="train", seed=3)
    print(r.observation.intent)
    r = await env.call("create_track", name="Lead", kind="midi")
    r = await env.call("submit", status="done")  # r.reward, r.observation.metadata["score"]
```

## Benchmarks (Apple M-series, arm64)

| Path | steps/s |
|---|---|
| Simulator ops only | ~23,000 |
| Environment, in-process (FastMCP client overhead) | ~200 |
| Environment over WebSocket (Docker) | ~270 |

For high-throughput RL use the in-process TRL wrapper, which calls the simulator directly.

## Milestone 1

`scripts/oracle_rollout.py` replays every oracle against the Docker server over WebSocket:
120/120 tasks (60 train, 60 heldout) score 1.0 and replay to an identical state hash.

## License

Apache-2.0. A future headless audio backend (DawDreamer/Pedalboard, GPL-3) would be shipped as a
separate image under GPL terms.
