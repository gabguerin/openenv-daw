from __future__ import annotations

import json

from openenv_daw.server.backends import SimBackend
from openenv_daw.server.episode import Episode
from openenv_daw.server.sim.session import (
    GRID,
    GRID_DIVISIONS,
    ForeignClip,
    ForeignDevice,
    ForeignParam,
    Session,
    Track,
    snap,
)
from openenv_daw.server.tasks.generators import families, generate
from openenv_daw.server.tasks.store import default_store
from openenv_daw.trl_env import make_environment_class


def foreign_task():
    """A generated task whose project also holds a read-only group track (foreign content)."""
    t = generate("mix_balance", 0, "train")
    s = Session.restore(t.initial_state) if t.initial_state else Session()
    s.tracks.append(
        Track(
            id=s.next_id("trk"),
            name="Legacy Bus",
            kind="group",
            foreign_devices=[
                ForeignDevice(
                    id=s.next_id("fdv"),
                    name="Serum",
                    params=[ForeignParam(name="Cutoff", display="1.2 kHz")],
                )
            ],
            foreign_clips=[
                ForeignClip(id=s.next_id("fcl"), name="Vocal Take", kind="audio", length_beats=16.0)
            ],
        )
    )
    return t.model_copy(update={"initial_state": s.snapshot()})


def test_infeasible_tasks_are_blocked():
    for seed in range(10):
        t = generate("infeasible", seed, "train")
        assert t.expected_status == "blocked" and "blocked" in t.tags
        assert t.oracle[-1].tool == "submit" and t.oracle[-1].args["status"] == "blocked"
        assert {o.tool for o in t.oracle[:-1]} <= {"list_devices"}  # only read-only lookups
        ep = Episode(t)
        ep.run("submit", {"status": "blocked", "reason": "not possible"})
        assert ep.score.success


def test_compositional_with_an_infeasible_part_is_blocked():
    blocked = [
        t
        for t in (generate("compositional", s, "train") for s in range(40))
        if t.expected_status == "blocked"
    ]
    assert blocked and all(
        {o.tool for o in t.oracle[:-1]} <= {"list_devices"} and t.oracle[-1].tool == "submit"
        for t in blocked
    )


def test_trl_reset_samples_by_seed():
    env = make_environment_class()()
    env.reset(seed=1)
    a = env.episode.task.task_id
    env.reset(seed=2)
    b = env.episode.task.task_id
    env.reset(seed=1)
    assert env.episode.task.task_id == a and a != b


def test_state_hash_ignores_empty_foreign_fields():
    s = Session()
    h = s.state_hash()
    SimBackend(s).call("create_track", {"name": "A", "kind": "midi"})
    snap_ = json.loads(json.dumps(s.snapshot()))
    for t in snap_["tracks"]:
        t.pop("foreign_devices")
        t.pop("foreign_clips")
    # a legacy snapshot without the fields loads and hashes identically
    assert Session.restore(snap_).state_hash() == s.state_hash() != h


def test_foreign_objects_are_readable_but_unsupported_to_mutate():
    t = foreign_task()
    ep = Episode(t)
    view = ep.backend.call("get_session", {})
    bus = next(x for x in view["tracks"] if x["kind"] == "group")
    assert bus["foreign_devices"] and bus["foreign_clips"]
    detail = ep.backend.call("get_session", {"track": "Legacy Bus"})["track"]
    dev = detail["foreign_devices"][0]
    assert dev["params"]
    params = ep.backend.call("get_device_params", {"track": "Legacy Bus", "device": dev["id"]})
    assert params["ok"] and params["foreign"] and params["params"][0]["display"]
    clip = detail["foreign_clips"][0]["id"]
    note = [{"pitch": 60, "start_beat": 0, "duration_beats": 1}]
    for tool, args in [
        (
            "set_device_param",
            {"track": "Legacy Bus", "device": dev["id"], "param": "x", "value": 0.5},
        ),
        ("delete_clip", {"clip": clip}),
        ("insert_midi_notes", {"clip": clip, "notes": note}),
        ("remove_midi_notes", {"clip": clip}),
        ("set_mixer_channel", {"track": "Legacy Bus", "mute": True}),
        ("rename_track", {"track": "Legacy Bus", "name": "X"}),
    ]:
        r = ep.backend.call(tool, args)
        assert r["ok"] is False and r["error"]["code"] == "unsupported", (tool, r)


def test_deleting_foreign_track_is_allowed_but_scored_as_damage():
    t = foreign_task()
    ep = Episode(t)
    for o in t.oracle[:-1]:
        ep.run(o.tool, o.args)
    assert ep.backend.call("delete_track", {"track": "Legacy Bus"})["ok"]
    ep.run("submit", {"status": "done"})
    assert any("Legacy Bus" in d for d in ep.score.damage) and not ep.score.success


def test_all_oracles_pass():
    tasks = [generate(f, s, "train") for f in families() for s in range(25)]
    tasks += default_store().heldout
    for t in tasks:
        ep = Episode(t, 40, 3)
        for o in t.oracle:
            ep.run(o.tool, o.args)
        assert ep.score and ep.score.reward == 1.0 and ep.score.success, t.task_id


def test_grid_default_is_sixteenth():
    assert GRID_DIVISIONS == 16 and GRID == 1 / 16 and snap(0.3) == 0.3125
