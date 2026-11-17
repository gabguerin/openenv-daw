import pytest
from openenv_daw.adapters.changes import (
    AddTrack,
    RemoveTrack,
    SetNotes,
    apply_changes,
    diff,
    same_session,
)
from openenv_daw.server.backends import SimBackend
from openenv_daw.server.sim.session import Session
from openenv_daw.server.tasks.store import default_store


def oracle_states(task):
    b = SimBackend()
    b.load(Session.restore(task.initial_state) if task.initial_state else Session())
    states = [Session.restore(b.session.snapshot())]
    for step in task.oracle:
        if step.tool == "submit":
            continue
        b.call(step.tool, step.args)
        states.append(Session.restore(b.session.snapshot()))
    return states


def tasks():
    store = default_store()
    out = [store.get_task("heldout", i) for i in range(len(store.heldout))]
    out += [store.sample("train", i) for i in range(80)]
    return out


def test_diff_roundtrip_every_oracle_step():
    n = 0
    for task in tasks():
        states = oracle_states(task)
        for a, b in zip(states, states[1:], strict=False):
            assert same_session(apply_changes(a, diff(a, b)), b), (task.task_id, diff(a, b))
            n += 1
        first, last = states[0], states[-1]  # and across the whole episode
        assert same_session(apply_changes(first, diff(first, last)), last), task.task_id
    assert n > 200


def test_apply_does_not_mutate_input():
    s = Session()
    b = SimBackend()
    b.call("create_track", {"name": "A", "kind": "midi"})
    apply_changes(s, diff(s, b.session))
    assert s.tracks == []


def test_diff_identical_is_empty():
    s = Session()
    assert diff(s, s) == []


def test_track_removal_precedes_additions_and_notes_edit_is_minimal():
    b = SimBackend()
    b.call("create_track", {"name": "A", "kind": "midi"})
    before = Session.restore(b.session.snapshot())
    b.call("delete_track", {"track": "A"})
    b.call("create_track", {"name": "B", "kind": "midi"})
    kinds = [type(c) for c in diff(before, b.session)]
    assert kinds.index(RemoveTrack) < kinds.index(AddTrack)
    b.call("create_midi_clip", {"track": "B", "start_beat": 0, "length_beats": 4})
    mid = Session.restore(b.session.snapshot())
    note = {"pitch": 60, "start_beat": 0, "duration_beats": 1}
    b.call("insert_midi_notes", {"clip": "clp_1", "notes": [note]})
    (ch,) = [c for c in diff(mid, b.session) if isinstance(c, SetNotes)]
    assert len(ch.added) == 1 and not ch.removed and len(ch.final) == 1


@pytest.mark.parametrize("bad", [None])
def test_apply_unknown_track_raises(bad):
    with pytest.raises(KeyError):
        apply_changes(Session(), [RemoveTrack("trk_9")])
