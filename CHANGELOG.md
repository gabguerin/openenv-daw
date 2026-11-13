# Changelog

Contract versions (`openenv_daw.versions`) change whenever the tools, the Session schema or the
instructions change. The `openenv_daw.adapters` API is provisional until a second driver exists.

## 0.2.0

- TOOLSET 1.1: instructions rendered per profile (`load_instructions(profile)`); parameters a DAW
  cannot set are reported as `supported: false` and the agent is told to check before editing.
- SCHEMA 1.0: unchanged.
- Backends are injectable (`backend_factory`), `Backend` v1 adds `load` and `close`, errors moved
  to `openenv_daw.errors.DawError`, profiles/checks/backends have registries.
- `openenv_daw.adapters`: Change ops, `diff`/`apply_changes`, `Driver`, `IdMap`, `DeviceMap`, `MirrorBackend`,
  `FakeDriver` and a conformance helper.
- Parameter gaps: `unsupported_params` per profile, per-episode masks, `param:` requires, `portable_gap` profile.
- `heldout_v3.jsonl` is the default heldout; v1 removed, v2 frozen. `ForeignClip.kind` "session" renamed "launcher".
- `uvicorn` moved to the `server` extra; `openenv` pinned to `>=0.13,<0.14`.
