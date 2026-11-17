# Design

How `openenv-daw` is built. For a quick start see the [README](../README.md).

## Layout

The package follows the `openenv init` layout: `models.py`, `client.py`, `openenv.yaml` at the root
and the environment under `server/` (`app.py`, `daw_environment.py`, `Dockerfile`). Everything else
is optional tooling: `adapters/` (adapter SDK), `training/` (harness, TRL, SFT export), `examples/`
and `scripts/`.

## Tools

17 portable tools plus `submit`, declared once as a FastMCP server (`server/tools.py`). Tool
results are JSON. Mutating tools return the affected object. Errors are typed: `not_found`,
`invalid_args`, `conflict`, `unsupported`, `execution_error`.

## Simulator

A linear timeline in beats, a dB mixer, return tracks with implicit sends, 1/16-beat grid snapping,
stable ids and a canonical 7-device catalog (`server/sim/devices.yaml`). It is pure Python and
deterministic; no audio engine. Backends (`server/backends.py`) wrap it: `SimBackend` by default,
`MirrorBackend` for a real DAW. Entry-point group `openenv_daw.backends` registers more.

## Environment

`DawEnvironment` is an OpenEnv `MCPEnvironment` with one request per episode. The reset observation
is the request text only, so the agent must call `get_session` to see the project. An episode ends
on `submit`, after 30 steps, or after 3 consecutive errors.

## Sounds and plugins

The agent never browses a DAW. `list_devices(query="sub bass")` searches a canonical library
(`server/sim/library.yaml`: synth presets, drum kits, effect presets, racks, plugins) and
`insert_device(device=<id>)` loads a hit. Presets load as their catalog device with editable
parameters; racks and plugins load read-only.

## Foreign objects

A Session can hold read-only group tracks, foreign plugins and audio clips (what an adapter reports
for things the simulator does not model). They are readable, any mutation returns `unsupported`,
and changing them is scored as damage.

## Tasks

Seeded families: `session_setup`, `melody_in_key`, `four_on_floor_beat`, `reverb_send`,
`mix_balance`, `chord_progression`, `edit_notes`, `device_tweak`, `disambiguation`, `load_sound`,
`compositional` (chains of 2-3 families; an infeasible part blocks the whole task), `near_miss`
(feasible requests that sound impossible) and about 10% infeasible requests
(`submit("blocked")`). Tasks carry difficulty tags `L1` to `L3` and an oracle solution.

Train tasks are generated from seeds. `server/tasks/heldout.jsonl` (212 tasks) is the held-out set
and uses unseen templates and slot values. Rebuild it with `scripts/build_heldout.py`.

## Scoring

The reward is a rubric tree (`DawRubric` in `server/rewards/score.py`); the server, the TRL
wrapper and the web UI all read it.

1. Gates: any of these scores 0: a wrong or missing submit status (`status`), a change to an
   object the task did not declare touchable (`safety`), a failed critical check (`critical`).
2. Correctness: the fraction of non-critical checks passed, multiplied by
   `1 - min(0.5, 0.15 * extra_objects)` and `1 - min(0.2, 0.02 * tool_errors)`.
3. Musicality: for musical tasks, `reward = (1 - W) * correctness + W * min(1, musicality / 0.8)`
   with W = 0.2 (`MUSIC_WEIGHT`). It applies only after the gates pass.
4. `metadata["success"]` is binary and ignores musicality. `named_rubrics()` shows each
   component's last score.

Efficiency (`calls`, `oracle_len`, `read_first`) is reported on the score, never rewarded.

## Bringing your own DAW (adapter SDK)

`openenv_daw.adapters` lets any DAW be trained against and driven by the same tools. See
[writing-an-adapter.md](writing-an-adapter.md). `MirrorBackend` re-reads the DAW on every call,
dry-runs the edit on the simulator, diffs the result and has the driver apply it. Adapters never
add tools. Destructive loading requires the `__openenv_scratch__` marker track.
