---
title: openenv-daw
emoji: 🎹
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
base_path: /web
tags:
  - openenv
---

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
- **Reward**: `submit` is required (no submit scores 0). Any failed critical check (or a wrong
  submit status) gives 0; otherwise the fraction of remaining checks passed, minus a penalty for
  changing untouchable objects or adding extraneous ones (new tracks or devices the task did not
  list in `allowed_new`, solo, a new fader above 0 dB: 0.15 each, shared cap 0.5) and per tool
  error (0.02 each, cap 0.2). `metadata["success"]` is binary.
- **Profiles**: `portable` and `mixer_only` map to tools and capability tags. Tasks declare
  `requires` (tool names plus tags such as `group_tracks`, `automation`, `third_party_plugin`,
  `render`); `resolve_for_profile(task, profile)` turns a task whose requirements are not covered
  into a "blocked" one, so the JSONL stays profile-neutral. `TaskStore.sample/get` take a profile.
- **Sounds and plugins**: the agent never browses a DAW. `list_devices(query="sub bass")` searches
  a canonical library (`catalog/library.yaml`: synth presets, drum kits, effect presets, racks,
  plugins) and `insert_device(device=<id>)` loads a hit. Presets load as their catalog device with
  editable parameters; racks and plugins load read-only. Plugins need the `third_party_plugin`
  capability (profile `portable_plugins`). Adapters map library ids to the DAW's items in their
  `DeviceMap` (`library:`); unmapped ids are hidden from the agent.
- **Foreign objects**: about 20% of tasks start with a read-only group track holding a foreign
  plugin and audio/session clips; they are readable, any mutation returns `unsupported`, and
  changing them is scored as damage.
- **Tasks**: seeded families `session_setup`, `melody_in_key`, `four_on_floor_beat`, `reverb_send`,
  `mix_balance`, `chord_progression`, `edit_notes`, `device_tweak` (display-value tolerance of 5%),
  `disambiguation` (duplicate clip names), `load_sound` (named or described presets, racks, plugins), `compositional` (chains of 2-3 families; an infeasible part
  blocks the whole task), feasible `near_miss` requests that mention infeasible-sounding words, and
  about 10% infeasible requests (`submit("blocked")`). Tasks carry difficulty tags `L1`-`L3`.
  Train tasks are generated from seeds; `heldout_v5.jsonl` (157 tasks, default; `OPENENV_DAW_HELDOUT`
  selects another file; v2, v3 and v4 are frozen) uses unseen templates, slot values and, for
  `compositional`, unseen family combinations.
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
| Simulator ops only | ~22,000 |
| Environment, in-process (FastMCP client overhead) | ~240 |
| Environment over WebSocket (Docker) | ~270 |

For high-throughput RL use the in-process TRL wrapper, which calls the simulator directly.

`scripts/benchmark.py --url ... --clients 8 --json out.json` also reports step/reset latency
(p50/p95) and throughput with N concurrent WebSocket sessions.

## Evaluation

```bash
uv run python -m openenv_daw.runner eval --tasks heldout --policy oracle   # ceiling
uv run python -m openenv_daw.runner eval --tasks heldout --policy noop     # floors: noop, blocked, random
uv run python -m openenv_daw.runner eval --tasks heldout --policy openai --samples 4 --out outputs/m
```

Writes `rollouts.jsonl`, `report.json` and `report.md`: success with a Wilson 95% interval, mean
reward, pass@k / pass^k, breakdowns by family, tag and expected status, the status confusion matrix
(expected -> submitted), terminations and calls against oracle length.

## Web UI (Hugging Face Space)

With `ENABLE_WEB_INTERFACE=true` the server mounts a UI at `/web`: a **DAW Studio** tab (one private
environment per visitor, arrangement view with piano rolls, live checks with the reward of submitting
`done` or `blocked` now, manual tool calls, oracle stepping, trajectory download) and the
stock shared Playground.
Deploy with the `space-openenv-daw` workflow (manual or `space-v*` tags) or `openenv push`.
Install the extras with `uv sync --extra server --extra ui`.

## Reward modes

`OPENENV_DAW_REWARD_MODE=shaped` (also `make_environment_class(reward_mode=...)`) gives a failed
critical gate with the right submit status 0.1 x the fraction of critical checks passed. The default
`strict` mode scores it 0. Efficiency (`calls`, `oracle_len`, `read_first`) is reported on the score,
never rewarded.

## Milestone 1

`scripts/oracle_rollout.py` replays every oracle against the Docker server over WebSocket:
120/120 tasks (60 train, 60 heldout) score 1.0 and replay to an identical state hash.

## License

Apache-2.0. A future headless audio backend (DawDreamer/Pedalboard, GPL-3) would be shipped as a
separate image under GPL terms.

## Rollouts and SFT data

The runner drives any OpenAI-compatible chat endpoint against the environment, in-process or over
WebSocket against a running server.

```bash
uv sync --extra runner
export OPENAI_BASE_URL=... OPENAI_API_KEY=... OPENENV_DAW_MODEL=...
uv run python -m openenv_daw.runner rollout --split train --n 100 --out rollouts.jsonl
uv run python -m openenv_daw.runner export rollouts.jsonl --out sft.jsonl
```

`export` keeps only successful episodes, deduplicated by (task, action sequence), and writes
`{"messages", "tools"}` rows ready for chat-format SFT.

## Training with TRL

`openenv_daw.trl_env.DawToolEnv` is a TRL `environment_factory` class generated from the same
FastMCP registry (tool names, signatures and docstrings are not duplicated). It calls the simulator
in-process, so it avoids the MCP client overhead and runs at simulator speed. `get_reward()` returns
the episode score; stopping without `submit` is scored as truncated. See `examples/grpo_train.py`.

> Mirror of `daw-agent/packages/openenv-daw` (private monorepo). Send changes there.


## Bringing your own DAW (adapter SDK)

`openenv_daw.adapters` lets any DAW be trained against and driven by the same tools:

1. Implement the `Driver` protocol (`read_session`, `apply`, optional native `undo`, `close`) for your DAW.
2. Describe its devices in a `DeviceMap` YAML; parameters it cannot set go in `unsupported_params`.
3. Run `assert_driver_conformance(driver_factory)` from `openenv_daw.adapters.testing` against the task suite
   (a `FakeDriver` shows how).
4. Register a backend factory under the `openenv_daw.backends` entry point group and serve it with
   `OPENENV_DAW_BACKEND=<name> openenv-daw-server` (or `openenv-daw serve --backend <name>`).

`MirrorBackend` re-reads the DAW on every call, dry-runs on the simulator, applies one atomic edit, verifies it
and rolls back on mismatch. Adapters never add tools. Parameter gaps are exposed to the agent as
`supported: false`; tasks that need an unsupported parameter expect `submit("blocked")`
(see the `portable_gap` profile). Destructive loading requires the `__openenv_scratch__` marker track.
The adapter API is provisional until a second driver exists.
