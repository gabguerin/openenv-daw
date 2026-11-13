from openenv_daw.backends import SimBackend
from openenv_daw.schema import Session


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
    r = call(b, "insert_midi_notes", clip="m", notes=[
        {"pitch": 60, "start_beat": 0.26, "duration_beats": 0.5, "velocity": 90}])
    assert r["added"][0]["start_beat"] == 0.25
    assert call(b, "insert_midi_notes", clip="m", notes=[
        {"pitch": 60, "start_beat": 0.5, "duration_beats": 0.5}])["error"]["code"] == "conflict"
    assert call(b, "insert_midi_notes", clip="m", notes=[
        {"pitch": 61, "start_beat": 3.75, "duration_beats": 1}])["error"]["code"] == "invalid_args"
    assert call(b, "insert_midi_notes", clip="m", notes=[
        {"pitch": 200, "start_beat": 0, "duration_beats": 1}])["error"]["code"] == "invalid_args"
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
    assert call(b, "set_device_param", track="Verb", device="reverb", param="nope", value=0.5)["ok"] is False
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


def test_profile_gating_and_render_unsupported():
    b = SimBackend(profile="mixer_only")
    assert call(b, "create_track", name="x", kind="midi")["error"]["code"] == "unsupported"
    assert call(b, "set_transport", tempo_bpm=100)["ok"]
    assert b.render(0, 8)["error"]["code"] == "unsupported"
    assert call(SimBackend(), "bogus")["error"]["code"] == "unsupported"
