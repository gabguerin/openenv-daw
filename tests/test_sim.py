from openenv_daw.server.backends import SimBackend
from openenv_daw.server.sim.session import Session


def call(b, tool, **kw):
    return b.call(tool, kw)


def test_tracks_ids_and_return_ordering():
    b = SimBackend()
    r = call(b, "create_track", name="Verb", kind="return")
    assert r["ok"] and r["track"]["id"] == "trk_1"
    call(b, "create_track", name="Bass", kind="midi")
    names = [t["name"] for t in call(b, "get_session")["tracks"]]
    assert names == ["Bass", "Verb"]  # returns stay last
    dup = call(b, "create_track", name="bass", kind="audio")
    assert dup["error"]["code"] == "conflict"


def test_clip_overlap_and_snap():
    b = SimBackend()
    call(b, "create_track", name="Lead", kind="midi")
    c = call(b, "create_midi_clip", track="Lead", start_beat=0.01, length_beats=8)
    assert c["clip"]["start_beat"] == 0.0 and c["snapped"]
    bad = call(b, "create_midi_clip", track="Lead", start_beat=4, length_beats=4)
    assert bad["error"]["code"] == "conflict"
    ok = call(b, "create_midi_clip", track="Lead", start_beat=8, length_beats=4)
    assert ok["ok"]


def test_notes_snap_bounds_overlap():
    b = SimBackend()
    call(b, "create_track", name="Lead", kind="midi")
    call(b, "create_midi_clip", track="Lead", start_beat=0, length_beats=4, name="m")
    r = call(
        b,
        "insert_midi_notes",
        clip="m",
        notes=[{"pitch": 60, "start_beat": 0.26, "duration_beats": 0.5, "velocity": 90}],
    )
    assert r["added"][0]["start_beat"] == 0.25
    assert (
        call(
            b,
            "insert_midi_notes",
            clip="m",
            notes=[{"pitch": 60, "start_beat": 0.5, "duration_beats": 0.5}],
        )["error"]["code"]
        == "conflict"
    )
    assert (
        call(
            b,
            "insert_midi_notes",
            clip="m",
            notes=[{"pitch": 61, "start_beat": 3.75, "duration_beats": 1}],
        )["error"]["code"]
        == "invalid_args"
    )
    assert (
        call(
            b,
            "insert_midi_notes",
            clip="m",
            notes=[{"pitch": 200, "start_beat": 0, "duration_beats": 1}],
        )["error"]["code"]
        == "invalid_args"
    )
    rm = call(b, "remove_midi_notes", clip="m", pitch=60)
    assert rm["removed"] == 1 and rm["note_count"] == 0


def test_mixer_sends_devices():
    b = SimBackend()
    call(b, "create_track", name="Lead", kind="midi")
    call(b, "create_track", name="Verb", kind="return")
    m = call(b, "set_mixer_channel", track="Lead", volume_db=-100, pan=2)
    assert m["track"]["volume_db"] == -70 and m["track"]["pan"] == 1
    assert call(b, "set_send", track="Lead", return_track="Lead", level_db=-6)["ok"] is False
    assert call(b, "set_send", track="Lead", return_track="Verb", level_db=-6)["ok"]
    assert call(b, "insert_device", track="Lead", device="simple_synth")["ok"]
    assert call(b, "insert_device", track="Lead", device="drum_kit")["error"]["code"] == "conflict"
    assert call(b, "insert_device", track="Verb", device="reverb")["ok"]
    p = call(b, "set_device_param", track="Verb", device="reverb", param="mix", value=0.5)
    assert p["param"]["display"] == "50 %"
    assert (
        call(b, "set_device_param", track="Verb", device="reverb", param="nope", value=0.5)["ok"]
        is False
    )
    assert call(b, "delete_track", track="Verb")["ok"]
    assert call(b, "get_session", track="Lead")["track"]["sends"] == {}


def test_snapshot_restore_and_hash_determinism():
    def build():
        b = SimBackend()
        call(b, "create_track", name="A", kind="midi")
        call(b, "set_transport", tempo_bpm=128)
        return b

    b1, b2 = build(), build()
    assert b1.session.state_hash() == b2.session.state_hash()
    snap = b1.snapshot()
    call(b1, "create_track", name="B", kind="audio")
    assert b1.session.state_hash() != b2.session.state_hash()
    b1.restore(snap)
    assert b1.session.state_hash() == b2.session.state_hash()
    assert Session.restore(snap).tempo_bpm == 128


def test_unknown_tool_is_unsupported():
    assert call(SimBackend(), "bogus")["error"]["code"] == "unsupported"


def _lead():
    b = SimBackend()
    call(b, "create_track", name="Lead", kind="midi")
    call(b, "create_midi_clip", track="Lead", start_beat=0, length_beats=4, name="A")
    return b


def test_update_midi_clip():
    b = _lead()
    call(b, "create_midi_clip", track="Lead", start_beat=8, length_beats=4, name="B")
    assert call(b, "update_midi_clip", clip="A")["error"]["code"] == "invalid_args"
    r = call(b, "update_midi_clip", clip="A", name="Hook", start_beat=4.01, length_beats=2)
    assert r["ok"] and r["snapped"] and r["clip"]["start_beat"] == 4.0
    assert call(b, "update_midi_clip", clip="Hook", start_beat=6, length_beats=4)["error"][
        "code"
    ] == "conflict"
    assert call(b, "update_midi_clip", clip="Hook", start_beat=-1)["error"]["code"] == "invalid_args"
    assert call(b, "update_midi_clip", clip="Hook", name=" ")["error"]["code"] == "invalid_args"
    assert call(b, "update_midi_clip", clip="Hook", length_beats=0)["error"]["code"] == "invalid_args"
    call(b, "insert_midi_notes", clip="Hook", notes=[{"pitch": 60, "start_beat": 1, "duration_beats": 1}])
    r = call(b, "update_midi_clip", clip="Hook", length_beats=1.5)
    assert r["error"]["code"] == "invalid_args" and "Remove the notes" in r["error"]["message"]
    assert call(b, "update_midi_clip", clip="Hook", length_beats=2)["ok"]


def test_delete_device():
    b = _lead()
    d = call(b, "insert_device", track="Lead", device="simple_synth")["device"]["id"]
    assert call(b, "delete_device", track="Lead", device="nope")["error"]["code"] == "not_found"
    assert call(b, "delete_device", track="Lead", device=d) == {"ok": True, "deleted": d}
    assert call(b, "get_session", track="Lead")["track"]["devices"] == []


def test_display_value():
    b = _lead()
    call(b, "insert_device", track="Lead", device="simple_synth")
    p = call(b, "get_device_params", track="Lead", device="simple_synth")["params"]
    wf = next(x for x in p if x["name"] == "waveform")
    assert wf["steps"] == ["saw", "square", "sine", "triangle"] and wf["scale"] == "steps"
    cut = next(x for x in p if x["name"] == "cutoff")
    assert (cut["min"], cut["max"], cut["scale"]) == (20, 20000, "log") and "steps" not in cut

    def setp(**kw):
        return call(b, "set_device_param", track="Lead", device="simple_synth", **kw)

    assert setp(param="waveform", display_value="Sine")["param"]["display"] == "sine"
    assert setp(param="cutoff", display_value=1000)["param"]["display"] == "1000 Hz"
    assert setp(param="detune", value=0.5)["ok"]
    for kw in (
        {"param": "cutoff"},
        {"param": "cutoff", "value": 0.5, "display_value": 100},
        {"param": "cutoff", "display_value": 10},
        {"param": "cutoff", "display_value": 30000},
        {"param": "cutoff", "display_value": "sine"},
        {"param": "waveform", "display_value": "wobble"},
        {"param": "waveform", "display_value": 1.5},
    ):
        assert setp(**kw)["error"]["code"] == "invalid_args", kw
    assert "20..20000" in setp(param="cutoff", display_value=10)["error"]["message"]
