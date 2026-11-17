# Changelog

The `openenv_daw.adapters` API is provisional until a second driver exists.

## 0.7.0

Two tools and display-unit parameters. Rewards of existing tasks are unchanged.

- New tools `delete_device` (catalog devices only; racks and plugins are `unsupported`) and
  `update_midi_clip` (rename, move, resize; snaps to the grid, overlaps are `conflict`, shrinking
  below the last note is `invalid_args`, notes are never trimmed). The simulator emits the existing
  `RemoveDevice` and `UpdateClip` changes, so drivers need no new code.
- `set_device_param` takes `display_value` (a number in the parameter's unit, or a step name for
  stepped parameters) or `value`; exactly one of the two. Out-of-range values are `invalid_args`.
  `get_device_params` also returns `min`, `max`, `scale` and `steps`.
- Adapters: `MirrorBackend` reads the notes of a clip before `update_midi_clip`, and `IdMap.bind`
  drops the old native key when an id is rebound. `adapters.testing` adds `TOOL_SCENARIOS` and
  `assert_tool_scenarios`, which cover tools no task oracle uses. The Ableton driver warns that a
  move rebuilds the clip.
- Follow-up: no task template uses the new tools yet, so trained policies will not learn them
  until tasks do. In-place note edits and a track index argument are left out on purpose.

## 0.6.0

Breaking: rewards change when a task damages existing objects.

- `DawRubric` is a rubric tree (`status`, `safety`, `critical` gates and a `quality` blend) that
  computes the reward. The server (`obs.reward`), the TRL wrapper and the web UI all use it, so
  they agree; `named_rubrics()` shows every component's `last_score`.
- Changing an object the task did not declare touchable is a hard gate (reward 0), not a 0.15
  subtraction. Extra objects and tool errors are multipliers on the correctness score
  (`1 - min(0.5, 0.15 n)` and `1 - min(0.2, 0.02 n)`), no longer subtracted.
- `EpisodeScore`: `damage` is now untouchable-object changes only; new `extraneous`,
  `extraneous_penalty` and `components`; `damage_penalty` removed. `metadata["facts"]` carries the
  measurements the rubric scores.
- Discovery: `discovery.json` (tags, representative queries, MCP tools, license) for
  `openenv catalog build`, with a CI job that builds the catalog and checks the env is found.

## 0.5.0

Breaking: rebuilt on the `openenv init` layout, with the guardrails that did not pay for their
complexity removed.

- Layout: `models.py`, `client.py`, `openenv.yaml` at the root; the environment is under `server/`
  (`app.py`, `daw_environment.py`, `Dockerfile`, `sim/`, `tasks/`, `rewards/`, `ui/`). Import paths
  are now `openenv_daw.server.*`.
- Removed: capability profiles, `requires`, parameter masks and `supported: false`, shaped reward
  mode, contract versions, `inconsistent_state`, mirror verification and rollback (`Tolerances`),
  `Driver.capabilities`/`native_undo`/`undo_last`, `render`, the Task API and foreign-object
  injection in generated tasks.
- Held-out set is a single `server/tasks/heldout.jsonl` (212 tasks); older `heldout_v*` are gone.
- `Episode(task, max_steps, max_consecutive_errors, backend_factory)`, `resolve_backend(name)()`
  and `make_environment_class(...)` take no profile.
- `scripts/typecheck.sh` runs mypy for the new layout.

## 0.4.0

Breaking: the bespoke runner is replaced by OpenEnv's harness.

- `openenv_daw.harness`: `DawSessionFactory` / `DawSession` (a `ResourceSession`), `collect`
  (threaded `MCPHarnessAdapter` rollouts into `RolloutSerializer`'s `results.jsonl`, resumable,
  Hub-pushable), `teacher_client` (OpenAI, Anthropic or any OpenAI-compatible endpoint) and
  `scripted_model_step` (oracle, noop, submit_only, blocked, random).
- Removed `runner.client` (`Policy`, `OpenAIPolicy`) and `runner.rollout` (adapters, nudging loop,
  `run_many`). CLI: `runner collect|export|eval`, flags follow `openenv collect`.
- `export_sft` reads collect dirs and writes `{messages, tools (JSON string), reward, task_id}`.
- `build_dataset_for` is `build_dataset(task_ids=...)`; `examples/sft_warmup.py` added;
  `examples/grpo_train.py` disables Qwen3 thinking.

## 0.3.0

Breaking: rewards change for train and heldout tasks (stricter checks, musicality blend).

- Scoring: soft musicality blend `reward = (1 - W) * net + W * min(1, musicality / 0.8)` with W = 0.2
  (`OPENENV_DAW_MUSIC_WEIGHT`, 0 disables), applied only when the correctness gate passes; `success`
  stays correctness-only. Heuristics (melody contour, cadence, rhythm, motifs; chord voice leading;
  drum accents) are calibrated on 50 Nottingham folk tunes (`tests/fixtures/reference_melodies.json`).
- Reward holes closed: chords need at least 3 notes per bar, spread within a quarter beat, held for
  half a bar; new checks `notes_every_bar`, `notes_transposed`, `existing_notes_kept_except`; hi-hat
  and range checks; a submit-only policy scores 0 on every feasible task. `music.roman_chord` reads
  numeral case (quality) and `voice_lead` builds smooth voicings.
- Generators: motif-based melodies (best of 40 by melody score), voice-led chord oracles, `prog_major`
  / `prog_minor` pools (train and heldout disjoint), drum velocity accents, device_tweak avoids
  default values, session_setup avoids default 4/4 at 120.
- `heldout_v6.jsonl` (212 tasks, at least 15 per family) is the default. `heldout_v5` stays frozen but
  is no longer oracle-valid for chord tasks.
- Runner: `submit_only` policy, per-group musicality, tool schemas identical to TRL's, plain
  `json.dumps` tool results; SFT export drops episodes with nudges or musicality below 0.6.
- `trl_env`: `reward_mode` may be a callable, `tool_schemas`, stratified `build_dataset`.
- TOOLSET 1.3: one instructions line about musicality.
- OpenEnv standard: `/metadata` carries name, description, version and README; Task API
  (`splits`, `tasks`, `num_tasks`, `task`, `task_range`, prompt-level fields only, train
  `tasks` is a 20-task preview); `openenv.yaml` declares `rubric_tree`, `task_api` and task counts;
  `server` script.
- Image: built on `ghcr.io/huggingface/openenv-base` with `uv sync` from `uv.lock`
  (`scripts/lock_standalone.sh`, pinned to the monorepo lock); `requirements.lock` is gone. The Space
  runs on `cpu-basic`.

## 0.2.0

- New families: `chord_progression`, `edit_notes`, `device_tweak` (check `device_param_display_in`),
  `disambiguation`, `compositional`; difficulty tags `L1`-`L3`; `TaskStore.sample(tags=)`.
  Train family sampling changes because the family weights changed.
- Scoring: extra clips on new tracks are extraneous damage when a task declares a `clip:<track>[:n]`
  budget; templates can promote checks to critical (`critical:`); `EpisodeScore` reports `calls`,
  `oracle_len`, `read_first`; opt-in `OPENENV_DAW_REWARD_MODE=shaped`. Rewards of train tasks change.
- Fixes: `reset()` without a seed picks a random one (`metadata["seed"]`); the runner uses the
  server profile's instructions; listing tools works before `reset`; the throughput benchmark loads
  each task's initial state.
- `python -m openenv_daw.runner eval` (reports, floor/ceiling policies); `summarize`/`gate` moved
  here from sidekick; benchmark latency percentiles and `--clients`.
- Web UI (`ui` extra): DAW Studio tab and a Space deploy workflow.

- TOOLSET 1.2: `list_devices(query=)` searches a canonical library of presets, drum kits, racks and
  plugins (`catalog/library.yaml`); `insert_device` takes a library id or exact name and answers a
  miss with suggestions. No new tools; one instructions line. Plugins need the `third_party_plugin`
  capability (new `portable_plugins` profile); in `portable` they are unavailable.
- SCHEMA 1.1: `Device.preset`, `ForeignDevice.item` and `ForeignDevice.role` (unset values are left
  out of `state_hash`, so old hashes stay valid). Presets load as their catalog device with
  parameter overrides; racks and plugins load as read-only foreign devices.
- Adapters: `AddDevice.preset`, new `AddForeignDevice` change, `DeviceMap` gains `library:` and
  per-entry `locator` hints, unmapped library items are hidden (`DeviceMap.masked()`).
- New `load_sound` family; `has_device` accepts `preset`, `item` and `tags_any`; oracle loads of
  items a DAW cannot load are rerouted to an available item with the same tag, or become blocked.
  `generate(..., profile=)`. `heldout_v5.jsonl` (157 tasks) is the default; v4 stays frozen.
  `infeasible` plugin pools now only hold plugins outside the library. Train sampling changes.
- `heldout_v5.jsonl` is stored profile-neutral (`generate(..., profile=None)`): plugin tasks are
  loadable and `resolve_for_profile` blocks them where the profile cannot load plugins. Consumers of
  raw `store.heldout` must resolve (`store.get`, `resolve_for_profile`). Infeasible plugin requests
  require the new never-offered tag `uncatalogued_plugin` (they stay blocked under `portable_plugins`).
- Eval: `evaluate(..., profile=)` and `evaluate_dual` score the same tasks under `portable`
  (headline) and `portable_plugins` (informational); the runner `eval` does both unless `--profile`.
- Adapters: `MirrorBackend` and `SimBackend` take `profile=None` (derived from the driver's
  capabilities, `profile_for`); `ApplyResult.acknowledged` (a loaded preset is checked by its base
  device only); `DeviceMap.merge`; `Episode` blocks tasks the backend's own `masked` set rules out and
  records `profile` and `adapter_masked`.
- TOOLSET 1.1: instructions rendered per profile (`load_instructions(profile)`); parameters a DAW
  cannot set are reported as `supported: false` and the agent is told to check before editing.
- Backends are injectable (`backend_factory`), `Backend` v1 adds `load` and `close`, errors moved
  to `openenv_daw.errors.DawError`, profiles/checks/backends have registries.
- `openenv_daw.adapters`: Change ops, `diff`/`apply_changes`, `Driver`, `IdMap`, `DeviceMap`, `MirrorBackend`,
  `FakeDriver` and a conformance helper.
- Parameter gaps: `unsupported_params` per profile, per-episode masks, `param:` requires, `portable_gap` profile.
- `heldout_v4.jsonl` (145 tasks) was the default heldout (now v5, above); v1 removed, v2 and v3 frozen. `ForeignClip.kind` "session" renamed "launcher".
- `uvicorn` moved to the `server` extra; `openenv` pinned to `>=0.13,<0.14`.
