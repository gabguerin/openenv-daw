import pytest

from openenv_daw.adapters.devices import DeviceMap
from openenv_daw.adapters.driver import SCRATCH_TRACK, Driver
from openenv_daw.adapters.mirror import MirrorBackend
from openenv_daw.adapters.testing import FakeDriver, assert_driver_conformance
from openenv_daw.errors import DawError
from openenv_daw.schema import Session
from openenv_daw.tasks.store import default_store

ADD = {"name": "Lead", "kind": "midi"}


def mirror(**kw):
    d = FakeDriver(**{k: kw.pop(k) for k in ("lossy", "native_undo") if k in kw})
    return d, MirrorBackend(d, allow_destructive_load=True, **kw)


def tasks():
    s = default_store()
    return [s.get_task("heldout", i) for i in range(len(s.heldout))] + [
        s.sample("train", i) for i in range(60)
    ]


def test_fake_driver_is_a_driver():
    assert isinstance(FakeDriver(), Driver)


@pytest.mark.parametrize("native_undo", [True, False])
def test_conformance_lossless(native_undo):
    rep = assert_driver_conformance(lambda: FakeDriver(native_undo=native_undo), tasks())
    assert rep["tasks"] >= 120


def test_conformance_lossy_still_scores_one():
    assert_driver_conformance(lambda: FakeDriver(lossy=True), tasks()[:60], lossless=False)


def test_scratch_marker_hidden_and_load_guarded():
    d = FakeDriver()
    m = MirrorBackend(d)
    assert m.session.tracks == []
    with pytest.raises(DawError):
        m.load(Session())
    d2 = FakeDriver(scratch=False)
    with pytest.raises(DawError):
        MirrorBackend(d2, allow_destructive_load=True).load(Session())
    assert any(t.name == SCRATCH_TRACK for t in d.read().tracks)


def test_dry_run_errors_come_from_simulator_without_touching_daw():
    d, m = mirror()
    r = m.call("create_midi_clip", {"track": "nope", "start_beat": 0, "length_beats": 4})
    assert r["error"]["code"] == "not_found" and d.apply_calls == 0


def test_ids_stay_stable_and_user_edits_are_not_errors():
    d, m = mirror()
    a = m.call("create_track", ADD)["track"]["id"]
    # the user edits the DAW directly between calls
    d._s.tracks[-1].volume_db = -9.0
    r = m.call("rename_track", {"track": a, "name": "Bass"})
    assert r["ok"] and m.session.tracks[0].id == a and m.session.tracks[0].volume_db == -9.0


@pytest.mark.parametrize("native_undo", [True, False])
def test_drift_is_undone_and_reported(native_undo):
    d, m = mirror(native_undo=native_undo)
    m.call("create_track", ADD)
    d.corrupt_next_apply = lambda s: setattr(s.tracks[-1], "volume_db", 5.0)
    r = m.call("set_mixer_channel", {"track": "Lead", "volume_db": -3})
    assert r["error"]["code"] == "execution_error"
    assert m.session.tracks[0].volume_db == 0.0 and not m.broken


def test_driver_failure_is_execution_error_and_state_unchanged():
    d, m = mirror()
    d.fail_next_apply = True
    assert m.call("create_track", ADD)["error"]["code"] == "execution_error"
    assert m.session.tracks == []


def test_failed_undo_is_inconsistent_state_and_sticky():
    d, m = mirror()
    m.call("create_track", ADD)
    d.corrupt_next_apply = lambda s: setattr(s.tracks[-1], "volume_db", 5.0)
    d.fail_undo = True
    assert (
        m.call("set_mixer_channel", {"track": "Lead", "volume_db": -3})["error"]["code"]
        == "inconsistent_state"
    )
    assert m.call("get_session", {})["error"]["code"] == "inconsistent_state"


def test_inconsistent_state_ends_episode():
    from openenv_daw.env.episode import Episode

    t = default_store().sample("train", 0)
    d = FakeDriver()
    m = MirrorBackend(d, allow_destructive_load=True)
    ep = Episode(t, "portable", 30, 3, backend_factory=lambda p: m)
    m.broken = True
    ep.run("get_session", {})
    assert ep.done and ep.termination == "inconsistent_state"


def test_device_map_masks_params_and_reports_coverage(tmp_path):
    f = tmp_path / "dm.yaml"
    f.write_text(
        "devices:\n  reverb:\n    native: Reverb\n    params:\n      mix: {native: Dry/Wet}\n      decay: unsupported\n"
    )
    dm = DeviceMap.from_yaml(f)
    assert "reverb.decay" in dm.unsupported_params() and "reverb.mix" not in dm.unsupported_params()
    assert dm.coverage()["devices"] == "1/7"
    d, m = mirror(device_map=dm)
    m.call("create_track", {"name": "Hall", "kind": "return"})
    m.call("insert_device", {"track": "Hall", "device": "reverb"})
    got = m.call("get_device_params", {"track": "Hall", "device": "reverb"})
    flagged = {p["name"] for p in got["params"] if p.get("supported") is False}
    assert "decay" in flagged and "mix" not in flagged
    assert (
        m.call(
            "set_device_param",
            {"track": "Hall", "device": "reverb", "param": "decay", "value": 0.5},
        )["error"]["code"]
        == "unsupported"
    )
    with pytest.raises(ValueError):
        DeviceMap.from_dict({"devices": {"nope": {"native": "x"}}})


def test_profile_must_be_covered_by_driver():
    d = FakeDriver()
    d.capabilities = frozenset({"get_session"})
    with pytest.raises(ValueError):
        MirrorBackend(d, "portable")


async def test_mcp_server_exposes_portable_tools_through_mirror():
    from fastmcp import Client

    from openenv_daw.adapters.serve import make_server

    d, m = mirror()
    server = make_server(m)
    async with Client(server) as c:
        names = {t.name for t in await c.list_tools()}
        assert "submit" not in names and "create_track" in names
        r = await c.call_tool("create_track", {"name": "Lead", "kind": "midi"})
        assert r.data["ok"] is True
    assert [t.name for t in m.session.tracks] == ["Lead"]
