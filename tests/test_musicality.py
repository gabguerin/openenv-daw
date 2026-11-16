"""Musicality heuristics: calibrated on folk melodies, never a gate."""

from __future__ import annotations

import copy
import json
import random
import statistics
from pathlib import Path

import pytest

from openenv_daw.env.episode import Episode
from openenv_daw.music import scale_pitches
from openenv_daw.rewards.musicality import chords_score, drums_score, melody_score
from openenv_daw.tasks.generators import generate

FIXTURE = Path(__file__).parent / "fixtures" / "reference_melodies.json"


def _human():
    return json.loads(FIXTURE.read_text())["melodies"]


def test_reference_corpus_scores_high():
    rows = [
        melody_score([(n["start"], n["pitch"], n["duration"], 90) for n in m["notes"]], m["key"])[0]
        for m in _human()
    ]
    assert len(rows) >= 40
    assert statistics.mean(rows) >= 0.8
    assert sum(r >= 0.7 for r in rows) / len(rows) >= 0.9


def test_random_in_key_notes_score_low():
    rng = random.Random(0)
    rows = []
    for _ in range(300):
        key = rng.choice(["C major", "A minor", "G major", "D minor", "Bb major", "F# minor"])
        sp = scale_pitches(key, 48, 84)
        bars, notes, t = rng.choice([1, 2, 4, 8]), [], 0.0
        while t < bars * 4:
            d = rng.choice([0.5, 1.0, 1.5, 2.0])
            notes.append((t, rng.choice(sp), d, 90))
            t += d
        rows.append(melody_score(notes, key)[0])
    assert statistics.mean(rows) <= 0.4
    assert sorted(rows)[int(0.9 * len(rows))] <= 0.55


def test_degenerate_melodies_score_low():
    scale = scale_pitches("C major", 60, 84)[:8]
    run = [(float(i), p, 1.0, 90) for i, p in enumerate(scale * 2)]
    one = [(float(i), 60, 1.0, 90) for i in range(16)]
    assert melody_score(run, "C major")[0] < 0.8
    assert melody_score(one, "C major")[0] < 0.5


def test_chord_voice_leading():
    smooth = [(0.0, p, 4.0, 90) for p in (60, 64, 67)] + [(4.0, p, 4.0, 90) for p in (59, 62, 67)]
    jumpy = [(0.0, p, 4.0, 90) for p in (48, 52, 55)] + [(4.0, p, 4.0, 90) for p in (55, 59, 62)]
    assert chords_score(smooth)[0] > chords_score(jumpy)[0]


def test_flat_drum_velocity_scores_below_accented():
    flat = [(float(b), 36, 0.25, 100) for b in range(8)]
    acc = [(float(b), 36, 0.25, 115 if b % 4 == 0 else 100) for b in range(8)]
    assert drums_score(acc)[0] > drums_score(flat)[0]


@pytest.mark.parametrize(
    "family", ["melody_in_key", "chord_progression", "four_on_floor_beat", "compositional"]
)
def test_oracle_is_musical_and_full_reward(family):
    seen = 0
    for split in ("train", "heldout"):
        for seed in range(25):
            t = generate(family, seed, split)
            if not t.music or t.expected_status != "done":
                continue
            ep = Episode(t, "portable", max_steps=40, max_consecutive_errors=3)
            for o in t.oracle:
                ep.run(o.tool, copy.deepcopy(o.args))
            s = ep.score
            assert s.success and s.reward == 1.0, t.task_id
            assert s.musicality is not None and s.musicality >= 0.8, (t.task_id, s.music_detail)
            seen += 1
    assert seen


def test_unmusical_but_correct_melody_earns_less_but_still_succeeds():
    t = next(
        x
        for x in (generate("melody_in_key", s, "train") for s in range(50))
        if x.expected_status == "done"
    )
    steps = [(o.tool, copy.deepcopy(o.args)) for o in t.oracle]
    for tool, args in steps:
        if tool == "insert_midi_notes":
            # same pitches, shuffled: still in key, in range, covering every bar
            rng = random.Random(1)
            pitches = [n["pitch"] for n in args["notes"]]
            rng.shuffle(pitches)
            for n, p in zip(args["notes"], pitches, strict=True):
                n["pitch"] = p
                n["velocity"] = 90
    ep = Episode(t, "portable", max_steps=40, max_consecutive_errors=3)
    for tool, args in steps:
        ep.run(tool, args)
    assert ep.score.success  # correctness is unchanged
    assert ep.score.reward < 1.0 or ep.score.musicality >= 0.8


def test_musicality_never_lifts_a_failed_gate(monkeypatch):
    t = generate("melody_in_key", 0, "train")
    ep = Episode(t, "portable", max_steps=40, max_consecutive_errors=3)
    ep.run("submit", {"status": "done"})  # nothing built: critical checks fail
    assert ep.score.reward == 0.0


def test_music_weight_zero_disables_bonus(monkeypatch):
    monkeypatch.setenv("OPENENV_DAW_MUSIC_WEIGHT", "0")
    t = generate("melody_in_key", 0, "train")
    ep = Episode(t, "portable", max_steps=40, max_consecutive_errors=3)
    for o in t.oracle:
        ep.run(o.tool, copy.deepcopy(o.args))
    assert ep.score.reward == 1.0
