# The portable contract

- **Tools**: `get_session`, `set_transport`, `create_track`, `delete_track`, `rename_track`,
  `create_midi_clip`, `delete_clip`, `update_midi_clip`, `insert_midi_notes`, `remove_midi_notes`,
  `set_mixer_channel`, `set_send`, `list_devices`, `insert_device`, `delete_device`, `get_device_params`,
  `set_device_param`, plus `submit` in the training environment only. Adapters never add tools.
  `list_devices(query=)` searches the canonical library of presets, kits, racks and plugins and
  `insert_device` takes a catalog or library id, so an agent never handles DAW paths or URIs.
- **Session**: `trk_`, `clp_` and `dev_` ids, dB volumes, normalized 0..1 device parameters,
  `Device.preset`, read-only foreign devices (`fdv_`, with `item`/`role` for loaded racks and
  plugins) and clips.
- **Library** (`server/sim/library.yaml`): entries are `preset` (built on a catalog device,
  editable), `rack` or `plugin` (read-only). Adapters map ids in their `DeviceMap` `library:`
  section.
- **Errors**: `not_found`, `invalid_args`, `conflict`, `unsupported`, `execution_error`.
- **Changes**: a change to the tools, the Session schema or `server/instructions.md` is listed under
  `Unreleased` in `CHANGELOG.md`. `openenv_daw.adapters` stays provisional until a second driver exists.
