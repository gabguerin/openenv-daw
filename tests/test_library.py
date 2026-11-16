"""Named presets, racks and plugins: library data, search, insertion and the adapter round trip."""

import pytest

from openenv_daw.adapters import MirrorBackend
from openenv_daw.adapters.testing import FakeDriver
from openenv_daw.backends.sim import SimBackend
from openenv_daw.schema import Session
from openenv_daw.sim.catalog import load_catalog, load_library, search_devices


def sim(profile="portable"):
    b = SimBackend(profile=profile)
    assert b.call("create_track", {"name": "T", "kind": "midi"})["ok"]
    return b


def test_library_is_consistent():
    lib = load_library()
    assert 30 <= len(lib) <= 60
    assert not set(lib) & set(load_catalog())
    assert {i.kind for i in lib.values()} == {"preset", "rack", "plugin"}


@pytest.mark.parametrize(
    "query,expected",
    [
        ("sub bass", "sub_bass"),
        ("808 drum kit", "kit_808"),
        ("robot lead", "robot_lead"),
        ("large hall reverb", "reverb_hall"),
        ("serum", "serum"),
    ],
)
def test_search_ranks_the_obvious_hit_first(query, expected):
    assert search_devices(query)[0]["device"] == expected


def test_search_filters_role_and_plugins():
    assert all(h["role"] == "audio_fx" for h in search_devices("reverb", role="audio_fx"))
    assert all(h["kind"] != "plugin" for h in search_devices("serum", exclude=frozenset({"serum"})))


def test_list_devices_without_query_is_the_catalog_and_with_query_searches():
    b = sim()
    r = b.call("list_devices", {})
    assert {d["device"] for d in r["devices"]} == set(load_catalog()) and "hint" in r
    r = b.call("list_devices", {"query": "warm pad"})
    assert r["devices"][0]["device"] == "warm_pad" and r["devices"][0]["base"] == "simple_synth"
    assert b.call("list_devices", {"query": "zzzz"})["devices"] == []


def test_preset_loads_as_its_base_device_with_overrides():
    b = sim()
    r = b.call("insert_device", {"track": "T", "device": "sub_bass"})
    assert (
        r["ok"] and r["device"]["device"] == "simple_synth" and r["device"]["preset"] == "sub_bass"
    )
    dev = b.session.tracks[0].devices[0]
    assert dev.preset == "sub_bass" and dev.params["cutoff"] == 0.45 and dev.name == "Sub Bass"
    # parameters stay editable
    assert b.call(
        "set_device_param", {"track": "T", "device": dev.id, "param": "cutoff", "value": 0.2}
    )["ok"]


def test_insert_by_exact_name_and_error_suggestions():
    b = sim()
    assert b.call("insert_device", {"track": "T", "device": "sub bass"})["ok"]
    r = sim().call("insert_device", {"track": "T", "device": "sub_baass"})
    assert not r["ok"] and r["error"]["code"] == "not_found" and "sub_bass" in r["error"]["message"]


def test_second_instrument_conflicts_including_racks():
    b = sim()
    assert b.call("insert_device", {"track": "T", "device": "instrument_rack"})["ok"]
    r = b.call("insert_device", {"track": "T", "device": "simple_synth"})
    assert r["error"]["code"] == "conflict"


def test_rack_is_a_read_only_foreign_device():
    b = sim()
    r = b.call("insert_device", {"track": "T", "device": "mixing_chain"})
    assert r["ok"] and r["device"]["foreign"]
    fd = b.session.tracks[0].foreign_devices[0]
    assert fd.item == "mixing_chain" and fd.role == "audio_fx" and fd.id.startswith("fdv_")
    assert (
        b.call("set_device_param", {"track": "T", "device": fd.id, "param": "x", "value": 0.5})[
            "error"
        ]["code"]
        == "unsupported"
    )


def test_plugins_need_the_capability():
    r = sim().call("insert_device", {"track": "T", "device": "serum"})
    assert r["error"]["code"] == "unsupported"
    assert all(
        h["kind"] != "plugin" for h in sim().call("list_devices", {"query": "serum"})["devices"]
    )
    b = sim("portable_plugins")
    assert b.call("insert_device", {"track": "T", "device": "serum"})["ok"]


def test_state_hash_ignores_unset_library_fields():
    b = sim()
    b.call("insert_device", {"track": "T", "device": "reverb"})
    h = b.session.state_hash()
    data = b.session.snapshot()
    assert "preset" in data["tracks"][0]["devices"][0]  # snapshot keeps it, hash drops it
    assert Session.restore(data).state_hash() == h


@pytest.mark.parametrize("device", ["sub_bass", "kit_808", "reverb_hall", "mixing_chain", "serum"])
def test_mirror_round_trip(device):
    m = MirrorBackend(FakeDriver(), "portable_plugins")
    assert m.call(
        "create_track",
        {
            "name": "T",
            "kind": "midi"
            if "bass" in device or "kit" in device or device == "serum"
            else "return",
        },
    )["ok"]
    r = m.call("insert_device", {"track": "T", "device": device})
    assert r["ok"], r
    t = m.session.tracks[0]
    assert [d.preset for d in t.devices] == (
        [device] if device in ("sub_bass", "kit_808", "reverb_hall") else []
    )
    assert [d.item for d in t.foreign_devices] == (
        [device] if device in ("mixing_chain", "serum") else []
    )


# --- tasks ---------------------------------------------------------------------------------


def test_load_sound_oracles_pass_in_the_sim_and_plugins_follow_the_profile():
    from openenv_daw.env.episode import Episode
    from openenv_daw.tasks.generators import generate

    seen = set()
    for split in ("train", "heldout"):
        for seed in range(40):
            blocked = generate("load_sound", seed, split)  # portable: no plugins
            task = generate("load_sound", seed, split, "portable_plugins")
            assert task.expected_status == "done"
            assert blocked.expected_status == (
                "blocked" if "third_party_plugin" in task.requires else "done"
            )
            ep = Episode(task, "portable_plugins", 60, 5)
            for step in task.oracle:
                ep.run(step.tool, step.args)
            assert ep.score is not None and ep.score.reward == 1.0, (task.task_id, ep.score)
            seen.add(blocked.expected_status)
    assert seen == {"blocked", "done"}


def test_an_unavailable_item_reroutes_described_requests_and_blocks_named_ones():
    from openenv_daw.tasks.generators import generate
    from openenv_daw.tasks.resolve import resolve_for_profile

    named = next(
        t for t in (generate("load_sound", i, "train") for i in range(40)) if "Sub Bass" in t.intent
    )
    assert (
        resolve_for_profile(named, "portable", frozenset({"item:sub_bass"})).expected_status
        == "blocked"
    )
    described = next(
        t
        for t in (generate("load_sound", i, "train") for i in range(60))
        if "bass sound" in t.intent
    )
    r = resolve_for_profile(described, "portable", frozenset({"item:sub_bass"}))
    assert r.expected_status == "done"
    (loaded,) = [s.args["device"] for s in r.oracle if s.tool == "insert_device"]
    assert loaded != "sub_bass" and "bass" in load_library()[loaded].tags


def test_has_device_matches_presets_items_and_tags():
    from openenv_daw.rewards.checks import CHECKS

    b = sim("portable_plugins")
    b.call("insert_device", {"track": "T", "device": "sub_bass"})
    b.call("create_track", {"name": "U", "kind": "midi"})
    b.call("insert_device", {"track": "U", "device": "serum"})
    s = b.session
    has = CHECKS["has_device"]
    assert has(s, s, {"track": "T", "device": "simple_synth"})
    assert has(s, s, {"track": "T", "preset": "sub_bass"})
    assert has(s, s, {"track": "T", "tags_any": ["bass", "lead"]})
    assert not has(s, s, {"track": "T", "preset": "robot_lead"})
    assert not has(s, s, {"track": "T", "item": "serum"})
    assert has(s, s, {"track": "U", "item": "serum"})
    assert not has(s, s, {"track": "U", "device": "simple_synth"})


# --- profile-neutral heldout, dual scores, adapter masks ------------------------------------------


def test_heldout_plugin_tasks_are_stored_loadable_and_resolve_per_profile():
    from openenv_daw.tasks.resolve import resolve_for_profile
    from openenv_daw.tasks.store import default_store

    plugin = [t for t in default_store().heldout if "third_party_plugin" in t.requires]
    assert plugin and all(t.expected_status == "done" for t in plugin)
    for t in plugin:
        assert resolve_for_profile(t, "portable").expected_status == "blocked"
        assert resolve_for_profile(t, "portable_plugins").expected_status == "done"


def test_uncatalogued_plugins_stay_blocked_in_every_profile():
    from openenv_daw.tasks.generators import generate
    from openenv_daw.tasks.resolve import resolve_for_profile

    for seed in range(20):
        t = generate("infeasible", seed, "train", None)
        assert t.expected_status == "blocked"
        assert resolve_for_profile(t, "portable_plugins").expected_status == "blocked"


async def test_dual_eval_scores_the_same_tasks_under_both_profiles(tmp_path):
    from openenv_daw.runner.report import evaluate_dual, load_task_set

    tasks = load_task_set("heldout")
    rep = await evaluate_dual("oracle", tasks, tmp_path)
    assert rep["profile"] == "portable" and rep["success"] == len(tasks)
    info = rep["informational"]["portable_plugins"]
    assert info["profile"] == "portable_plugins" and info["success"] == len(tasks)
    # plugin tasks are loadable only in the second report
    assert info["by_expected_status"]["done"]["success"] > rep["by_expected_status"]["done"]["success"]
    assert (tmp_path / "portable_plugins" / "rollouts.jsonl").is_file()


def test_device_map_merge_and_profile_for():
    from openenv_daw.adapters import DeviceMap
    from openenv_daw.sim.profiles import profile_for

    a = DeviceMap.from_dict({"devices": {"reverb": {"native": "Reverb"}}, "library": {}})
    b = a.merge({"library": {"reverb_hall": {"native": "Large Hall"}}})
    assert "reverb_hall" in b.library and not a.library
    assert profile_for({"third_party_plugin"}) == "portable_plugins" and profile_for(set()) == "portable"


def test_an_unmapped_item_blocks_a_named_task_in_the_episode_and_is_recorded():
    from openenv_daw.adapters import DeviceMap
    from openenv_daw.env.episode import Episode
    from openenv_daw.tasks.generators import generate

    task = next(
        t for t in (generate("load_sound", i, "train") for i in range(40)) if "Sub Bass" in t.intent
    )
    assert task.expected_status == "done"
    dm = DeviceMap.from_dict({"devices": {"simple_synth": {"native": "Synth"}}, "library": {}})
    ep = Episode(
        task,
        "portable",
        30,
        3,
        backend_factory=lambda p: MirrorBackend(FakeDriver(), p, dm, allow_destructive_load=True),
    )
    assert ep.task.expected_status == "blocked" and "item:sub_bass" in ep.adapter_masked
    ep.run("submit", {"status": "blocked", "reason": "no such sound"})
    meta = ep.result_metadata()
    assert ep.score.success and meta["profile"] == "portable" and "item:sub_bass" in meta["adapter_masked"]
