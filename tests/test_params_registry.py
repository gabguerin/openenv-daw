import pytest

from openenv_daw import TOOLSET_VERSION, DawError
from openenv_daw.backends import Backend, ProfileGuard, SimBackend, resolve_backend
from openenv_daw.env.episode import Episode
from openenv_daw.rewards.checks import register_check
from openenv_daw.sim.profiles import register_profile
from openenv_daw.tasks.resolve import apply_param_mask, resolve_for_profile, sample_param_mask
from openenv_daw.tasks.store import TaskStore, default_store
from openenv_daw.tools import load_instructions


def _reverb_task():
    store = default_store()
    for i in range(200):
        t = store.sample("train", i, family="reverb_send")
        if t.expected_status == "done":
            return t
    raise AssertionError


def test_backend_protocol_and_registry():
    b = SimBackend()
    assert isinstance(b, Backend)
    assert resolve_backend("sim")("portable").profile == "portable"
    assert resolve_backend("openenv_daw.backends.sim:SimBackend")("portable").profile == "portable"


def test_profile_guard_checks_capabilities_and_gates():
    g = ProfileGuard(SimBackend(profile="portable"), "mixer_only")
    assert g.call("create_track", {"name": "A", "kind": "midi"})["error"]["code"] == "unsupported"
    narrow = SimBackend(profile="mixer_only")
    with pytest.raises(ValueError):
        ProfileGuard(narrow, "portable")


def test_register_check_namespaced_and_unique():
    with pytest.raises(ValueError):
        register_check("nodot")(lambda s, i, a: True)
    register_check("tests.always")(lambda s, i, a: True)
    with pytest.raises(ValueError):
        register_check("tests.always")(lambda s, i, a: True)


def test_register_profile_rejects_unknown_tools():
    with pytest.raises(ValueError):
        register_profile("bad", {"make_coffee"})


def test_instructions_follow_profile():
    assert "set_send" in load_instructions("portable")
    assert "set_send" in load_instructions("mixer_only")
    assert "MIDI pitch" not in load_instructions("mixer_only")
    assert TOOLSET_VERSION


def test_task_store_from_paths(tmp_path):
    t = default_store().get_task("heldout", 0)
    p = tmp_path / "t.jsonl"
    p.write_text(t.model_dump_json() + "\n")
    assert TaskStore.from_paths(p).num_tasks("heldout") == 1


def test_param_mask_blocks_task_and_hides_param():
    t = _reverb_task()
    params = [r for r in t.requires if r.startswith("param:")]
    assert params
    mask = frozenset({params[0][6:]})
    blocked = resolve_for_profile(t, "portable", mask)
    assert blocked.expected_status == "blocked"
    b = Episode(blocked, "portable", 30, 3, masked_params=mask).backend
    dev, param = next(iter(mask)).split(".")
    b.call("create_track", {"name": "R", "kind": "return"})
    b.call("insert_device", {"track": "R", "device": dev})
    got = b.call("get_device_params", {"track": "R", "device": dev})
    assert [p for p in got["params"] if p.get("supported") is False][0]["name"] == param
    assert all("supported" not in p for p in got["params"] if p["name"] != param)
    err = b.call("set_device_param", {"track": "R", "device": dev, "param": param, "value": 0.5})
    assert err["error"]["code"] == "unsupported"
    ep = Episode(blocked, "portable", 30, 3, masked_params=mask)
    ep.run("get_session", {})
    ep.run("submit", {"status": "blocked"})
    assert ep.score.success


def test_sample_mask_is_deterministic_and_train_only():
    t = _reverb_task()
    assert sample_param_mask(t, 1, 1.0) == sample_param_mask(t, 1, 1.0)
    assert sample_param_mask(t, 1, 0.0) == frozenset()
    held = default_store().get_task("heldout", 0)
    assert apply_param_mask(held, "portable", 1.0, 0)[1] == frozenset()


def test_error_codes_are_shared():
    assert DawError("conflict", "x").to_result()["error"]["code"] == "conflict"
