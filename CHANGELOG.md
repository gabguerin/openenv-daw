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
- **Harness**: `collect` (threaded, resumable rollouts on OpenEnv's harness), `teacher_client`,
  scripted policies, SFT export, TRL environment and GRPO example.
- **Adapters** (provisional): Change ops, `diff`/`apply_changes`, `Driver`, `IdMap`, `DeviceMap`,
  `MirrorBackend`, `FakeDriver` and conformance helpers in `adapters.testing`.
- **Web UI and Space**: DAW Studio (`ui` extra), deployed to the Hugging Face Space on
  `cpu-basic`, image built on `openenv-base` from `uv.lock`.
- **Discovery**: `discovery.json` for `openenv catalog build`, checked in CI.
