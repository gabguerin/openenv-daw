# Changelog

Add changes under `## Unreleased`. Do not change the version in a feature change; it moves only
when a release is cut (see CONTRIBUTING.md).

The `openenv_daw.adapters` API is provisional until a second driver exists.

## Unreleased (0.1.0)

First release, not yet published. This section describes the current feature set.

- **Tools**: `get_session`, `set_transport`, `create_track`, `delete_track`, `rename_track`,
  `create_midi_clip`, `delete_clip`, `update_midi_clip`, `insert_midi_notes`, `remove_midi_notes`,
  `set_mixer_channel`, `set_send`, `list_devices`, `insert_device`, `delete_device`,
  `get_device_params`, `set_device_param`, plus `submit` in the training environment.
  `list_devices(query=)` searches a canonical library of presets, kits, racks and plugins;
  `set_device_param` takes `display_value` (in the parameter's unit) or `value`.
- **Session schema**: `trk_`, `clp_` and `dev_` ids, dB volumes, normalized device parameters,
  presets, read-only foreign devices (racks and plugins).
- **Reward**: `DawRubric`, a rubric tree (`status`, `safety`, `critical` gates and a `quality`
  blend including musicality). Changing an object the task did not declare touchable is a hard
  gate; extra objects and tool errors are multipliers. The server, the TRL wrapper and the web UI
  all use it.
- **Tasks**: train and held-out splits (`server/tasks/heldout.jsonl`), families including chord
  progressions, note edits, device tweaks, sound loading, disambiguation and compositional tasks,
  with difficulty tags.
- **Task families** `remove_device` and `clip_edit` exercise `delete_device` and `update_midi_clip`;
  the held-out set is 242 tasks (the first 212 are unchanged).
- **Held-out v9**: train and held-out share all 24 keys; `compositional` tasks carry a meter that
  sets clip lengths; plugin requests call `list_devices` before `blocked`; `instructions.md` states
  scale and roman-numeral semitone conventions and that instrument tracks are `kind="midi"`.
- **Shaped GRPO reward**: `DawRubric(shaped=True)` adds `PartialProgress` (capped at 0.5, never above
  the full reward), so tasks the model never passes still yield a graded training signal. Eval keeps
  the strict reward.
- **Held-out v8**: the key pools of `chord_progression` and `melody_in_key` are rebalanced. Train
  covers 20 keys, including sharps and flats; held-out keeps four unseen keys (`Ab major`,
  `B major`, `C# minor`, `F minor`). The melody, chord and compositional held-out tasks are
  regenerated, so v8 is not comparable with v7 on those families; the other families are unchanged.
- **Prompt**: the instructions ask for varied drum velocities and for deciding on `blocked` before
  any edit. the SFT export always writes the deployment prompt as the system message.
- **Scoring**: foreign-device parameters and float round-trip noise in device parameters do not
  count as damage or as a changed state.
- **Training**: `openenv_daw.trl_env` (`DawToolEnv`, `build_dataset`, `tool_schemas`), a TRL
  `environment_factory`. Rollout collection, SFT export and evaluation moved to `ableton-sidekick`
  (`sidekick collect`, `sidekick eval`, `sidekick train export`), so the former
  `openenv_daw.training` package, its CLI and the `runner` extra are gone.
- **Adapters** (provisional): Change ops, `diff`/`apply_changes`, `Driver`, `IdMap`, `DeviceMap`,
  `MirrorBackend`, `FakeDriver` and conformance helpers in `adapters.testing`.
- **Web UI and Space**: DAW Studio (`ui` extra), deployed to the Hugging Face Space on
  `cpu-basic`, image built on `openenv-base` from `uv.lock`. Includes a "Run a model" card to step
  or run an episode through an OpenAI-compatible endpoint locally (`OPENENV_DAW_MODEL_URL`,
  `OPENENV_DAW_MODEL`, `OPENENV_DAW_MODEL_KEY`, `OPENENV_DAW_MODEL_MAX_TOKENS`) or through Hugging
  Face Inference Providers with Sign in with Hugging Face OAuth on the Space.
- **Layout**: task templates live in one `server/tasks/templates.yaml`; `openenv-daw mcp` is
  served from `cli.py`.
- **Discovery**: `discovery.json` for `openenv catalog build`, checked in CI.
