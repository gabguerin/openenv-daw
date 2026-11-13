# The portable contract

- **Tools** (`TOOLSET_VERSION`): `get_session`, `set_transport`, `create_track`, `delete_track`,
  `rename_track`, `create_midi_clip`, `delete_clip`, `insert_midi_notes`, `remove_midi_notes`,
  `set_mixer_channel`, `set_send`, `list_devices`, `insert_device`, `get_device_params`,
  `set_device_param`, plus `submit` in the training environment only. Adapters never add tools.
- **Session** (`SCHEMA_VERSION`): `trk_`, `clp_` and `dev_` ids, dB volumes, normalized 0..1 device
  parameters, read-only foreign devices and clips.
- **Errors**: `not_found`, `invalid_args`, `conflict`, `unsupported`, `execution_error`,
  `inconsistent_state` (the DAW could not be restored; the episode ends).
- **Profiles** name the tool subset and capabilities of a target; `unsupported_params` lists
  parameters a DAW cannot set (shown to the agent as `supported: false`).
- **Versioning**: any change to tools, schema or the instruction text bumps the matching version and
  is listed in `CHANGELOG.md`. `openenv_daw.adapters` stays provisional until a second driver exists.
