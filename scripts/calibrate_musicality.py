"""Print musicality statistics on the reference melodies and on random in-key notes."""

import json
import random
import statistics

from openenv_daw.server.music import scale_pitches
from openenv_daw.server.rewards.musicality import melody_score

d = json.load(open("tests/fixtures/reference_melodies.json"))["melodies"]
sc = []
for m in d:
    notes = [(n["start"], n["pitch"], n["duration"], 90) for n in m["notes"]]
    s, subs = melody_score(notes, m["key"])
    sc.append(s)
print(
    "human n",
    len(sc),
    "min",
    round(min(sc), 2),
    "mean",
    round(statistics.mean(sc), 2),
    "below0.7",
    sum(x < 0.7 for x in sc),
)
for m, s in sorted(zip(d, sc, strict=True), key=lambda x: x[1])[:4]:
    notes = [(n["start"], n["pitch"], n["duration"], 90) for n in m["notes"]]
    print(round(s, 2), m["key"], melody_score(notes, m["key"])[1])
rng = random.Random(0)
rs = []
for _ in range(300):
    key = rng.choice(["C major", "A minor", "G major", "D minor", "Bb major", "F# minor"])
    sp = scale_pitches(key, 48, 84)
    bars = rng.choice([1, 2, 4, 8])
    mode = rng.choice(["quarters", "mixed"])
    notes = []
    t = 0
    while t < bars * 4:
        dur = 1.0 if mode == "quarters" else rng.choice([0.5, 1.0, 1.5, 2.0])
        notes.append((t, rng.choice(sp), dur, 90))
        t += dur
    rs.append(melody_score(notes, key)[0])
print(
    "random mean",
    round(statistics.mean(rs), 2),
    "max",
    round(max(rs), 2),
    "above0.4",
    sum(x > 0.4 for x in rs),
)
print(
    "random p90",
    round(sorted(rs)[int(0.9 * len(rs))], 2),
    "p95",
    round(sorted(rs)[int(0.95 * len(rs))], 2),
)
