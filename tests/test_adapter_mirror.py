import pytest
from openenv_daw.adapters.devices import DeviceMap
from openenv_daw.adapters.driver import SCRATCH_TRACK, Driver
from openenv_daw.adapters.mirror import MirrorBackend
from openenv_daw.adapters.testing import FakeDriver, assert_driver_conformance
from openenv_daw.server.sim.errors import DawError
from openenv_daw.server.sim.session import Session
from openenv_daw.server.tasks.store import default_store

ADD = {"name": "Lead", "kind": "midi"}


def mirror(**kw):
    d = FakeDriver(**{k: kw.pop(k) for k in ("lossy",) if k in kw})
    return d, MirrorBackend(d, allow_destructive_load=True, **kw)


def tasks():
    s = default_store()
    return [s.get_task("heldout", i) for i in range(len(s.heldout))] + [
        s.sample("train", i) for i in range(60)
    ]


def test_fake_driver_is_a_driver():
    assert isinstance(FakeDriver(), Driver)


def test_conformance_lossless():
    rep = assert_driver_conformance(FakeDriver, tasks())
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


def test_driver_failure_is_execution_error_and_state_unchanged():
    d, m = mirror()
    d.fail_next_apply = True
    assert m.call("create_track", ADD)["error"]["code"] == "execution_error"
    assert m.session.tracks == []


def test_device_map_reports_coverage_and_validates(tmp_path):
    f = tmp_path / "dm.yaml"
    f.write_text(
        "devices:\n  reverb:\n    native: Reverb\n    params:\n      mix: {native: Dry/Wet}\n      decay: unsupported\n"
    )
    dm = DeviceMap.from_yaml(f)
    assert dm.param("reverb", "decay") is None and dm.param("reverb", "mix") is not None
    assert dm.coverage()["devices"] == "1/7"
    with pytest.raises(ValueError):
        DeviceMap.from_dict({"devices": {"nope": {"native": "x"}}})


def test_conformance_skips_tasks_the_device_map_does_not_cover():
    dm = DeviceMap.from_dict({"devices": {"simple_synth": {"native": "Synth"}}, "library": {}})
    rep = assert_driver_conformance(FakeDriver, tasks(), device_map=dm, lossless=False)
    assert 0 < rep["tasks"] < len(tasks())


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
