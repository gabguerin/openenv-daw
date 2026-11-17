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

**Teach an AI agent to use a digital audio workstation.** An [OpenEnv](https://github.com/openenv/OpenEnv)
environment where an agent gets a plain-language request ("add a 4-bar melody in D minor on a new
track called Lead") and has to carry it out with DAW tools, then gets scored on the result.

![DAW Studio: reset a task, call tools, watch the reward](assets/demo.gif)

*The DAW Studio web UI: start a task, call tools by hand or step the oracle, and watch the checks
and reward update.*

## Why use it

- **Train abstract, deploy specific.** The agent learns one portable tool set. Adapters map it to
  Ableton Live 12, REAPER 7, Bitwig and Ardour.
- **Fast and deterministic.** A pure-Python simulator, no audio engine, no DAW needed. The same
  seed gives the same task and the same state hash.
- **Verifiable rewards.** Every task has checks and an oracle solution that scores 1.0, so a
  reward bug shows up as a failing test.
- **Ready for RL and SFT.** A TRL environment, a rollout runner, SFT export and a 212-task
  held-out benchmark are included.

## Try it in 60 seconds

```bash
docker build -t openenv-daw . && docker run -p 8000:8000 -e ENABLE_WEB_INTERFACE=true openenv-daw
```

Open <http://localhost:8000/web> and use the **DAW Studio** tab:

1. Press **Reset** (or **Random task**) to get a request.
2. Pick a tool, edit its arguments and press **Run**, or press **Step oracle** to watch the
   reference solution.
3. Watch the **Checks** and the **reward if you submit now**, then call `submit`.

Without Docker: `uv sync --extra server --extra ui`, then
`ENABLE_WEB_INTERFACE=true uv run uvicorn openenv_daw.server.app:app --port 8000`.

## Use it from Python

```bash
pip install openenv-daw
```

```python
from openenv_daw.env.client import DawEnv

with DawEnv(base_url="http://localhost:8000").sync() as env:
    r = env.reset(split="train", seed=3)
    print(r.observation.intent)              # the request, as plain text
    env.call("get_session")                  # the agent must look at the project first
    env.call("create_track", name="Lead", kind="midi")
    r = env.call("submit", status="done")
    print(r.reward, r.observation.metadata["success"])
```

Use `async with DawEnv(...) as env:` and `await env.reset(...)` / `await env.call(...)` in async
code. To start a deployed Space image locally: `DawEnv.from_env("<user>/openenv-daw")`.

## What the agent can do

15 portable tools plus `submit`. Every result is JSON, and errors are typed (`not_found`,
`invalid_args`, `conflict`, `unsupported`).

| Area | Tools |
|---|---|
| Project | `get_session`, `set_transport` |
| Tracks and clips | `create_track`, `delete_track`, `rename_track`, `create_midi_clip`, `delete_clip` |
| Notes | `insert_midi_notes`, `remove_midi_notes` |
| Mixing | `set_mixer_channel`, `set_send` |
| Sounds and effects | `list_devices`, `insert_device`, `get_device_params`, `set_device_param` |
| Finish | `submit` (`done`, or `blocked` when the request is impossible) |

Tasks come in families, from simple to compositional:

| Family | Example request |
|---|---|
| `session_setup` | Switch the time signature to 7/8 and set 135 beats per minute. |
| `mix_balance` | Set Lead to -9 dB, pan Pad hard right, and mute Drums. |
| `melody_in_key`, `chord_progression`, `four_on_floor_beat` | Musical writing, checked for key, harmony and rhythm. |
| `edit_notes`, `device_tweak`, `reverb_send`, `load_sound` | Change existing material, tweak a device, add a send, load a described sound. |
| `disambiguation`, `near_miss` | Duplicate names; requests that only sound impossible. |
| `compositional` | Chains of 2 to 3 of the above. |
| `infeasible` | About 10% of tasks: the right answer is `submit("blocked")`. |

Difficulty tags `L1` to `L3`. Train tasks are generated from seeds; the held-out set
`heldout_v6.jsonl` has 212 tasks with unseen templates and values.

## How scoring works

1. **Gate.** Any failed critical check, a wrong submit status, or no `submit` at all scores 0.
2. **Net score.** The fraction of remaining checks passed, minus penalties for damaging things the
   task did not touch and for tool errors.
3. **Musicality.** For musical tasks, 20% of the reward is a soft musicality score (melody shape,
   voice leading, rhythm). It counts only after the gate passes and never changes `success`.
   Set `OPENENV_DAW_MUSIC_WEIGHT=0` to turn it off.

`metadata["success"]` is binary and about correctness only. For training you can use
`OPENENV_DAW_REWARD_MODE=shaped`, which gives partial credit for passing critical checks.
Details: [docs/design.md](https://github.com/gabguerin/openenv-daw/blob/main/docs/design.md#scoring).

## Evaluate and train

```bash
uv sync --all-extras
uv run pytest
uv run python -m openenv_daw.runner eval --tasks heldout --policy oracle   # ceiling
uv run python -m openenv_daw.runner eval --tasks heldout --policy noop     # floors: noop, submit_only, blocked, random
uv run python -m openenv_daw.runner eval --tasks heldout --policy openai --samples 4 --out outputs/m
```

`eval` writes `rollouts.jsonl`, `report.json` and `report.md`: success with a 95% interval, mean
reward and musicality, pass@k, breakdowns by family and tag, and calls against oracle length.

**Collect rollouts and SFT data** from any OpenAI-compatible endpoint:

```bash
uv sync --extra runner
export OPENAI_BASE_URL=... OPENAI_API_KEY=... OPENENV_DAW_MODEL=...
uv run python -m openenv_daw.runner rollout --split train --n 100 --out rollouts.jsonl
uv run python -m openenv_daw.runner export rollouts.jsonl --out sft.jsonl
```

`export` keeps successful, musical episodes without nudges, deduplicated by task and action
sequence, as `{"messages", "tools"}` rows for chat-format SFT.

**Train with TRL.** `openenv_daw.trl_env.DawToolEnv` is a TRL `environment_factory` class built
from the same tool registry. It calls the simulator in-process, so it runs at simulator speed. See
`examples/grpo_train.py`.

| Path | steps/s (Apple M-series) |
|---|---|
| Simulator ops only | ~22,000 |
| Environment, in-process | ~240 |
| Environment over WebSocket (Docker) | ~270 |

## Deploy

- **Hugging Face Space** (the web UI runs no model, so `cpu-basic` is enough):
  `openenv push --repo-id <you>/openenv-daw --hardware cpu-basic --exclude .hfexclude`, or the
  `space-openenv-daw` workflow.
- **Docker**: `docker build -t openenv-daw .` (built on `openenv-base`, dependencies from `uv.lock`).
- Check it against the OpenEnv standard: `openenv validate --level runtime`.

## Go further

- [Design](https://github.com/gabguerin/openenv-daw/blob/main/docs/design.md): simulator, profiles,
  sounds and plugins, foreign objects, task families, scoring details.
- [Writing an adapter](https://github.com/gabguerin/openenv-daw/blob/main/docs/writing-an-adapter.md):
  drive your own DAW with the same tools (implement `Driver`, describe devices in a `DeviceMap`,
  pass `assert_driver_conformance`). The adapter API is provisional until a second driver exists.
- [CHANGELOG](https://github.com/gabguerin/openenv-daw/blob/main/CHANGELOG.md).

## License

Apache-2.0. A future headless audio backend (DawDreamer/Pedalboard, GPL-3) would be shipped as a
separate image under GPL terms. This repository mirrors `daw-agent/packages/openenv-daw`; send
changes to the monorepo.
