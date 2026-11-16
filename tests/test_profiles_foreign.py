from __future__ import annotations

import json

from openenv_daw.backends import SimBackend
from openenv_daw.env.episode import Episode
from openenv_daw.schema import Session
from openenv_daw.sim.profiles import PROFILES, covers
from openenv_daw.tasks.generators import families, generate
from openenv_daw.tasks.resolve import resolve_for_profile
from openenv_daw.tasks.store import TaskStore, default_store
from openenv_daw.trl_env import build_dataset, make_environment_class


def test_profiles_have_tools_and_capabilities():
    for p in PROFILES.values():
        assert (
            {"tools", "capabilities"} <= set(p) <= {"tools", "capabilities", "unsupported_params"}
        )


def test_mixer_only_blocks_tasks_that_need_missing_tools():
    t = generate("melody_in_key", 0, "train")
    assert t.expected_status == "done" and "insert_midi_notes" in t.requires
    r = resolve_for_profile(t, "mixer_only")
    assert r.expected_status == "blocked" and [c.kind for c in r.checks] == ["state_unchanged"]
    assert [o.tool for o in r.oracle] == ["submit"] and r.oracle[0].args["status"] == "blocked"
    ep = Episode(r, "mixer_only", 30, 3)
    ep.run("submit", {"status": "blocked", "reason": "no clip tools"})
    assert ep.score.success


def test_mixer_tasks_stay_feasible_in_mixer_only():
    t = generate("mix_balance", 1, "train")
    assert resolve_for_profile(t, "mixer_only").expected_status == "done"
    assert covers("mixer_only", t.requires)


def test_store_threads_profile():
    s = TaskStore()
    blocked = s.sample("train", 3, family="melody_in_key", profile="mixer_only")
    assert blocked.expected_status == "blocked"
    assert s.get("melody_in_key-train-0", "mixer_only").expected_status == "blocked"
    assert s.get("melody_in_key-train-0").expected_status == "done"
    assert len(build_dataset("train", 20, profile="mixer_only")) == 20


def test_infeasible_declares_capability_tags():
    for seed in range(10):
        t = generate("infeasible", seed, "train")
        assert t.expected_status == "blocked"
        assert set(t.requires) & {"group_tracks", "automation", "third_party_plugin", "render"}


def test_trl_reset_samples_by_seed():
    env = make_environment_class()()
    env.reset(seed=1)
    a = env.episode.task.task_id
    env.reset(seed=2)
    b = env.episode.task.task_id
    env.reset(seed=1)
    assert env.episode.task.task_id == a and a != b


def test_heldout_default_is_v4_and_older_frozen(monkeypatch):
    from openenv_daw.tasks.store import heldout_path, load_heldout

    monkeypatch.delenv("OPENENV_DAW_HELDOUT", raising=False)
    assert heldout_path().name == "heldout_v4.jsonl"
    v4 = load_heldout()
    monkeypatch.setenv("OPENENV_DAW_HELDOUT", "heldout_v3.jsonl")
    v3 = load_heldout()
    monkeypatch.setenv("OPENENV_DAW_HELDOUT", "heldout_v2.jsonl")
    v2 = load_heldout()
    assert len(v2) == 70 and len(v3) == 70 and 140 <= len(v4) <= 160
    assert {t.family for t in v4} == set(families())


def test_portable_gap_blocks_every_reverb_send_task():
    from openenv_daw.tasks.store import load_heldout

    ts = [t for t in load_heldout() if t.family == "reverb_send"]
    assert ts
    for t in ts:
        r = resolve_for_profile(t, "portable_gap")
        assert r.expected_status == "blocked"


# --- foreign objects ------------------------------------------------------------------------


def foreign_task():
    for seed in range(100):
        t = generate("mix_balance", seed, "train")
        if "foreign" in t.tags:
            return t
    raise AssertionError


def test_about_a_fifth_of_train_tasks_have_foreign_objects():
    n = sum("foreign" in generate(f, s, "train").tags for f in families() for s in range(60))
    assert 0.1 < n / (60 * len(families())) < 0.3


def test_state_hash_ignores_empty_foreign_fields():
    s = Session()
    h = s.state_hash()
    SimBackend(s).call("create_track", {"name": "A", "kind": "midi"})
    snap = json.loads(json.dumps(s.snapshot()))
    for t in snap["tracks"]:
        t.pop("foreign_devices")
        t.pop("foreign_clips")
    # a legacy snapshot without the fields loads and hashes identically
    assert Session.restore(snap).state_hash() == s.state_hash() != h


def test_foreign_objects_are_readable_but_unsupported_to_mutate():
    t = foreign_task()
    ep = Episode(t, "portable", 30, 3)
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
    ep = Episode(t, "portable", 30, 3)
    for o in t.oracle[:-1]:
        ep.run(o.tool, o.args)
    assert ep.backend.call("delete_track", {"track": "Legacy Bus"})["ok"]
    ep.run("submit", {"status": "done"})
    assert any("Legacy Bus" in d for d in ep.score.damage) and not ep.score.success


def test_all_oracles_pass_including_foreign():
    tasks = [generate(f, s, "train") for f in families() for s in range(25)]
    tasks += default_store().heldout
    assert any("foreign" in t.tags for t in tasks)
    for t in tasks:
        ep = Episode(t, "portable", 40, 3)
        for o in t.oracle:
            ep.run(o.tool, o.args)
        assert ep.score and ep.score.reward == 1.0 and ep.score.success, t.task_id


def test_grid_default_is_sixteenth():
    from openenv_daw.schema.units import GRID, GRID_DIVISIONS, snap

    assert GRID_DIVISIONS == 16 and GRID == 1 / 16 and snap(0.3) == 0.3125
