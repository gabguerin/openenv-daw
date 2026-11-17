# Design

How `openenv-daw` is built. For a quick start see the [README](../README.md).

## Tools

15 portable tools plus `submit`, declared once as a FastMCP server (`openenv_daw.tools.registry`).
Tool results are JSON. Mutating tools return the affected object. Errors are typed: `not_found`,
`invalid_args`, `conflict`, `unsupported`.

## Simulator

A linear timeline in beats, a dB mixer, return tracks with implicit sends, 1/16-beat grid snapping,
stable ids, a canonical 7-device catalog (`src/openenv_daw/catalog/devices.yaml`) and capability
profiles (`portable` is the default). It is pure Python and deterministic; no audio engine.

## Environment

`DawEnvironment` is an OpenEnv `MCPEnvironment` with one request per episode. The reset observation
is the request text only, so the agent must call `get_session` to see the project. An episode ends
on `submit`, after 30 steps, or after 3 consecutive errors. A verified step log is kept.

## Profiles

`portable`, `mixer_only` and `portable_plugins` map to tools and capability tags. Tasks declare
`requires` (tool names plus tags such as `group_tracks`, `automation`, `third_party_plugin`,
`render`). `resolve_for_profile(task, profile)` turns a task whose requirements a profile does not
cover into a "blocked" one, so the task files stay profile-neutral. `TaskStore.sample` and
`TaskStore.get` take a profile.

## Sounds and plugins

The agent never browses a DAW. `list_devices(query="sub bass")` searches a canonical library
(`catalog/library.yaml`: synth presets, drum kits, effect presets, racks, plugins) and
`insert_device(device=<id>)` loads a hit. Presets load as their catalog device with editable
parameters; racks and plugins load read-only. Plugins need the `third_party_plugin` capability
(profile `portable_plugins`). Adapters map library ids to the DAW's items in their `DeviceMap`
(`library:`); unmapped ids are hidden from the agent.

## Foreign objects

About 20% of tasks start with a read-only group track holding a foreign plugin and audio or session
clips. They are readable, any mutation returns `unsupported`, and changing them is scored as damage.

## Tasks

Seeded families: `session_setup`, `melody_in_key`, `four_on_floor_beat`, `reverb_send`,
`mix_balance`, `chord_progression`, `edit_notes`, `device_tweak` (display-value tolerance of 5%),
`disambiguation` (duplicate clip names), `load_sound` (named or described presets, racks, plugins),
`compositional` (chains of 2-3 families; an infeasible part blocks the whole task), feasible
`near_miss` requests that mention infeasible-sounding words, and about 10% infeasible requests
(`submit("blocked")`). Tasks carry difficulty tags `L1` to `L3`. Every task carries an oracle
solution.

Train tasks are generated from seeds. `heldout_v6.jsonl` (212 tasks, at least 15 per family) is the
default held-out set and uses unseen templates, slot values and, for `compositional`, unseen family
combinations. `OPENENV_DAW_HELDOUT` selects another file. `heldout_v2` to `heldout_v5` are frozen;
v5 is no longer oracle-valid for chord tasks.

## Scoring

The reward comes from one rubric tree, `DawRubric`, which the server, the TRL wrapper and the UI
all use. Its parts are inspectable through `env.rubric.named_rubrics()` (`last_score`).

1. Gates (any failure scores 0): `status` (the submit status must equal the expected one, no
   `submit` scores 0), `safety` (changing a pre-existing object the task did not declare touchable,
   for example a foreign track, is damage) and `critical` (every critical check must pass).
2. Correctness: the fraction of non-critical checks passed, multiplied by `1 - min(0.5, 0.15 n)`
   for `n` extra objects (new tracks or devices not in `allowed_new`, solo, a new fader above 0 dB)
   and by `1 - min(0.2, 0.02 e)` for `e` tool errors.
3. Musicality: for musical tasks, `reward = (1 - W) * correctness + W * min(1, musicality / 0.8)`
   with W = 0.2 (`OPENENV_DAW_MUSIC_WEIGHT`; 0 disables).
4. `metadata["success"]` is binary: gates passed, every check passed, no extra objects. It ignores
   musicality.

`OPENENV_DAW_REWARD_MODE=shaped` gives failed critical checks (right status, no damage) 0.1 times
the fraction of critical checks passed. The default `strict` mode scores it 0. Efficiency (`calls`,
`oracle_len`, `read_first`) is reported on the score, never rewarded.

## Bringing your own DAW (adapter SDK)

`openenv_daw.adapters` lets any DAW be trained against and driven by the same tools. See
[writing-an-adapter.md](writing-an-adapter.md) for the steps. `MirrorBackend` re-reads the DAW on
every call, dry-runs on the simulator, applies one atomic edit, verifies it and rolls back on
mismatch. Adapters never add tools. Parameter gaps are exposed to the agent as `supported: false`;
tasks that need an unsupported parameter expect `submit("blocked")` (see the `portable_gap`
profile). Destructive loading requires the `__openenv_scratch__` marker track. The adapter API is
provisional until a second driver exists.
