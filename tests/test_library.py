"""Named presets, racks and plugins: library data, search, insertion and the adapter round trip."""

import pytest
from openenv_daw.adapters import MirrorBackend
from openenv_daw.adapters.testing import FakeDriver
from openenv_daw.server.backends import SimBackend
from openenv_daw.server.sim.catalog import load_catalog, load_library, search_devices
from openenv_daw.server.sim.session import Session


def sim():
    b = SimBackend()
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
    assert any(h["kind"] == "plugin" for h in search_devices("serum"))


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


def test_plugins_load_as_read_only_devices():
    b = sim()
    assert b.call("insert_device", {"track": "T", "device": "serum"})["ok"]
    assert any(
        h["kind"] == "plugin" for h in sim().call("list_devices", {"query": "serum"})["devices"]
    )


def test_state_hash_ignores_unset_library_fields():
    b = sim()
    b.call("insert_device", {"track": "T", "device": "reverb"})
    h = b.session.state_hash()
    data = b.session.snapshot()
    assert "preset" in data["tracks"][0]["devices"][0]  # snapshot keeps it, hash drops it
    assert Session.restore(data).state_hash() == h


@pytest.mark.parametrize("device", ["sub_bass", "kit_808", "reverb_hall", "mixing_chain", "serum"])
def test_mirror_round_trip(device):
    m = MirrorBackend(FakeDriver())
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


def test_load_sound_oracles_pass_in_the_sim():
    from openenv_daw.server.episode import Episode
    from openenv_daw.server.tasks.generators import generate

    for split in ("train", "heldout"):
        for seed in range(40):
            task = generate("load_sound", seed, split)
            assert task.expected_status == "done"
            ep = Episode(task, 60, 5)
            for step in task.oracle:
                ep.run(step.tool, step.args)
            assert ep.score is not None and ep.score.reward == 1.0, (task.task_id, ep.score)


def test_has_device_matches_presets_items_and_tags():
    from openenv_daw.server.rewards.checks import CHECKS

    b = sim()
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


# --- adapters ----------------------------------------------------------------------------------


def test_uncatalogued_plugins_are_blocked():
    from openenv_daw.server.tasks.generators import generate

    for seed in range(20):
        assert generate("infeasible", seed, "train").expected_status == "blocked"


def test_device_map_merge():
    from openenv_daw.adapters import DeviceMap

    a = DeviceMap.from_dict({"devices": {"reverb": {"native": "Reverb"}}, "library": {}})
    b = a.merge({"library": {"reverb_hall": {"native": "Large Hall"}}})
    assert "reverb_hall" in b.library and not a.library
