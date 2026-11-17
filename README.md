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

Install the [OpenEnv](https://github.com/openenv/OpenEnv) CLI (`pip install openenv`), then from this
folder:

```bash
uv sync --extra server --extra ui     # once
ENABLE_WEB_INTERFACE=true uv run server
```

Open <http://localhost:8000/web> and use the **DAW Studio** tab:

1. Press **Reset** (or **Random task**) to get a request.
2. Pick a tool, edit its arguments and press **Run**, or press **Step oracle** to watch the
   reference solution.
3. Watch the **Checks** and the **reward if you submit now**, then call `submit`.

`uv run server` runs on the host. To run in a container instead:

```bash
docker build -f server/Dockerfile -t openenv-daw .
docker run -p 8000:8000 -e ENABLE_WEB_INTERFACE=true openenv-daw
```

## Use it from Python

```bash
pip install openenv-daw
```

```python
from openenv_daw.client import DawEnv

with DawEnv(base_url="http://localhost:8000").sync() as env:
    r = env.reset(split="train", seed=3)
    print(r.observation.intent)  # the request, as plain text
    env.call("get_session")  # the agent must look at the project first
    env.call("create_track", name="Lead", kind="midi")
    r = env.call("submit", status="done")
    print(r.reward, r.observation.metadata["success"])
```

Use `async with DawEnv(...) as env:` and `await env.reset(...)` / `await env.call(...)` in async
code. To start a deployed Space image locally: `DawEnv.from_env("<user>/openenv-daw")`.

## What the agent can do

15 portable tools plus `submit`. Every result is JSON, and errors are typed (`not_found`,
`invalid_args`, `conflict`, `unsupported`, `execution_error`).

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
`heldout.jsonl` has 212 tasks with unseen templates and values.

## How scoring works

1. **Gate.** Any failed critical check, a wrong submit status, or no `submit` at all scores 0.
2. **Net score.** The fraction of remaining checks passed, minus penalties for damaging things the
   task did not touch and for tool errors.
3. **Musicality.** For musical tasks, 20% of the reward is a soft musicality score (melody shape,
   voice leading, rhythm). It counts only after the gate passes and never changes `success`.

`metadata["success"]` is binary and about correctness only.
Details: [docs/design.md](https://github.com/gabguerin/openenv-daw/blob/main/docs/design.md#scoring).

## Evaluate and train

```bash
uv sync --all-extras
uv run pytest
uv run python -m openenv_daw.training eval --tasks heldout --policy oracle   # ceiling
uv run python -m openenv_daw.training eval --tasks heldout --policy noop     # floors: noop, submit_only, blocked, random
uv run python -m openenv_daw.training eval --tasks heldout --policy openai --samples 4 --out outputs/m
```

`eval` writes `results.jsonl`, `report.json` and `report.md`: success with a 95% interval, mean
reward and musicality, pass@k, breakdowns by family and tag, and calls against oracle length.

**Collect rollouts, SFT, then GRPO** (the OpenEnv
[Collect Rollouts for SFT](https://huggingface.co/docs/openenv/tutorials/sft-warmup) pipeline):

```bash
uv sync --extra runner --extra trl
export OPENAI_API_KEY=...
# 1. teacher rollouts: results.jsonl + metadata.json, resumable, optionally pushed to the Hub
uv run python -m openenv_daw.training collect --provider openai --model gpt-5-mini -n 300 \
    -o rollouts [--push-to-hub <you>/daw-rollouts]
#    self-hosted teacher: --llm-endpoint http://localhost:8000 --model Qwen/Qwen3-8B
# 2. filter by the environment's verdict into chat rows {messages, tools, reward, task_id}
uv run python -m openenv_daw.training export rollouts -o sft.jsonl
# 3. SFT warm-up (format compliance), then GRPO from the checkpoint
uv run python examples/sft_warmup.py --data sft.jsonl --model Qwen/Qwen3-1.7B
uv run python examples/grpo_train.py --model runs/daw-sft
```

`openenv collect` itself only knows `openspiel` and `reasoning_gym`; `openenv_daw.training.harness` is this
environment's session factory (`DawSessionFactory`) on the same OpenEnv harness. `export` keeps
successful, musical episodes (musicality of at least 0.6), deduplicated by task and action sequence.
`DawToolEnv` is a TRL `environment_factory` class built from the same tool registry (reward from
`get_reward`), and tool schemas and tool results match the collected data. It runs the simulator
in-process, so it trains at simulator speed.

| Path | steps/s (Apple M-series) |
|---|---|
| Simulator ops only | ~22,000 |
| Environment, in-process | ~240 |
| Environment over WebSocket (Docker) | ~270 |

## Deploy and share

All with the `openenv` CLI:

```bash
openenv validate --level static --skip-build        # fast manifest check
openenv validate --level runtime                    # builds the image and runs the contract checks
openenv validate http://localhost:8000              # probe a running server
openenv push --repo-id <you>/openenv-daw --hardware cpu-basic --exclude .hfexclude
openenv fork <owner>/openenv-daw                    # copy a deployed Space to your account
```

The web UI runs no model, so a `cpu-basic` Space is enough (Docker Spaces on `cpu-basic` need a
Hugging Face PRO account). The `space-openenv-daw` workflow runs the same `push`. The image is built
on `openenv-base` with dependencies from `uv.lock`. To use a deployed Space from Python,
`pip install git+https://huggingface.co/spaces/<owner>/openenv-daw` and
`DawEnv.from_env("<owner>/openenv-daw")`.

## Go further

- [Design](https://github.com/gabguerin/openenv-daw/blob/main/docs/design.md): simulator, sounds and
  plugins, foreign objects, task families, scoring details.
- [Writing an adapter](https://github.com/gabguerin/openenv-daw/blob/main/docs/writing-an-adapter.md):
  drive your own DAW with the same tools (implement `Driver`, describe devices in a `DeviceMap`,
  pass `assert_driver_conformance`). The adapter API is provisional until a second driver exists.
- [CHANGELOG](https://github.com/gabguerin/openenv-daw/blob/main/CHANGELOG.md).

## License

Apache-2.0. A future headless audio backend (DawDreamer/Pedalboard, GPL-3) would be shipped as a
separate image under GPL terms. This repository mirrors `daw-agent/packages/openenv-daw`; send
changes to the monorepo.
