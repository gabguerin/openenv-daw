# Changelog

Contract versions (`openenv_daw.versions`) change whenever the tools, the Session schema or the
instructions change. The `openenv_daw.adapters` API is provisional until a second driver exists.

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

- TOOLSET 1.1: instructions rendered per profile (`load_instructions(profile)`); parameters a DAW
  cannot set are reported as `supported: false` and the agent is told to check before editing.
- SCHEMA 1.0: unchanged.
- Backends are injectable (`backend_factory`), `Backend` v1 adds `load` and `close`, errors moved
  to `openenv_daw.errors.DawError`, profiles/checks/backends have registries.
- `openenv_daw.adapters`: Change ops, `diff`/`apply_changes`, `Driver`, `IdMap`, `DeviceMap`, `MirrorBackend`,
  `FakeDriver` and a conformance helper.
- Parameter gaps: `unsupported_params` per profile, per-episode masks, `param:` requires, `portable_gap` profile.
- `heldout_v4.jsonl` (145 tasks) is the default heldout; v1 removed, v2 and v3 frozen. `ForeignClip.kind` "session" renamed "launcher".
- `uvicorn` moved to the `server` extra; `openenv` pinned to `>=0.13,<0.14`.
