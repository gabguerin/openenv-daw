"""Build tests/fixtures/reference_melodies.json: folk melodies for calibrating the musicality
heuristics, converted from the Nottingham Music Database (ABC; a public folk-tune collection,
https://github.com/jukedeck/nottingham-dataset).

uv run python scripts/build_reference_melodies.py [n=50]

Only simple tunes are kept (no tuplets, meter or key changes, no ties) whose first section ends on
the tonic. The output has pitch, start and duration in quarter-note beats (first section only)."""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from fractions import Fraction
from pathlib import Path

BASE = "https://raw.githubusercontent.com/jukedeck/nottingham-dataset/master/ABC_cleaned/"
FILES = ["ashover", "hpps", "jigs", "morris", "playford", "reelsa-c", "reelsd-g", "reelsh-l"]
LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARPS = ["F", "C", "G", "D", "A", "E", "B"]
FLATS = ["B", "E", "A", "D", "G", "C", "F"]
MAJOR_FIFTHS = {
    "C": 0,
    "G": 1,
    "D": 2,
    "A": 3,
    "E": 4,
    "B": 5,
    "F": -1,
    "Bb": -2,
    "Eb": -3,
    "Ab": -4,
}
TOKEN = re.compile(
    r"(?P<acc>\^\^|__|\^|_|=)?(?P<note>[A-Ga-gz])(?P<oct>[',]*)(?P<num>\d*)(?P<slash>/*)(?P<den>\d*)"
)


def key_info(k: str) -> tuple[str, int, str] | None:
    m = re.fullmatch(r"([A-G][b#]?)(m|min|maj|major|minor)?", k.strip())
    if not m:
        return None
    root, mode = (
        m.group(1),
        "minor"
        if (m.group(2) or "").startswith("m") and m.group(2) != "maj" and m.group(2) != "major"
        else "major",
    )
    major_root = root
    if mode == "minor":
        pcs = (
            LETTERS[root[0]] + (1 if root.endswith("#") else -1 if root.endswith("b") else 0) + 3
        ) % 12
        names = {
            0: "C",
            7: "G",
            2: "D",
            9: "A",
            4: "E",
            11: "B",
            5: "F",
            10: "Bb",
            3: "Eb",
            8: "Ab",
        }
        if pcs not in names:
            return None
        major_root = names[pcs]
    if major_root not in MAJOR_FIFTHS:
        return None
    return root, MAJOR_FIFTHS[major_root], mode


def parse(tune: str) -> dict | None:
    head: dict[str, str] = {}
    body: list[str] = []
    for line in tune.splitlines():
        if re.match(r"^[A-Z]:", line):
            if line[0] in "MLK" and line[0] in head and line[0] != "K":
                return None  # meter/length change
            if line[0] == "K" and "K" in head:
                return None
            head.setdefault(line[0], line[2:].strip())
        elif line.strip() and not line.startswith("%"):
            body.append(line)
    if "K" not in head or "M" not in head:
        return None
    ki = key_info(head["K"])
    if ki is None:
        return None
    root, fifths, mode = ki
    try:
        num, den = (int(x) for x in head["M"].split("/"))
    except ValueError:
        return None
    unit = (
        Fraction(head["L"])
        if "L" in head
        else (Fraction(1, 16) if num / den < 0.75 else Fraction(1, 8))
    )
    text = " ".join(body)
    text = re.sub(r'"[^"]*"', "", text)
    if any(c in text for c in "(-[{") and not re.search(r"\[[12]", text):
        return None
    if "(" in text or "-" in text or "{" in text:
        return None
    text = re.sub(r"\[[12]", "|", text).replace("\\", "")
    sig = {n: 1 for n in SHARPS[: max(fifths, 0)]} | {n: -1 for n in FLATS[: max(-fifths, 0)]}
    notes: list[tuple[float, int, float]] = []
    t = 0.0
    section_end = None
    i = 0
    bar_acc: dict[str, int] = {}
    tonic = (LETTERS[root[0]] + (1 if root.endswith("#") else -1 if root.endswith("b") else 0)) % 12
    while i < len(text):
        c = text[i]
        if c in "|:]":
            j = i
            while j < len(text) and text[j] in "|:] ":
                j += 1
            seg = text[i:j]
            bar_acc = {}
            if notes and (":|" in seg or "||" in seg or "|]" in seg):
                section_end = len(notes)
                break
            i = j
            continue
        m = TOKEN.match(text, i)
        if not m or m.start() != i:
            i += 1
            continue
        i = m.end()
        n = m.group("note")
        mult = Fraction(int(m.group("num") or 1))
        if m.group("slash"):
            d = int(m.group("den")) if m.group("den") else 2 ** len(m.group("slash"))
            mult = Fraction(int(m.group("num") or 1), d)
        dur = float(mult * unit * 4)
        if n != "z":
            letter = n.upper()
            acc = m.group("acc")
            if acc:
                bar_acc[letter] = {"^": 1, "^^": 2, "_": -1, "__": -2, "=": 0}[acc]
            alt = bar_acc.get(letter, sig.get(letter, 0))
            octv = 5 if n.islower() else 4
            octv += m.group("oct").count("'") - m.group("oct").count(",")
            pitch = 12 * (octv + 1) + LETTERS[letter] + alt
            notes.append((t, pitch, dur))
        t += dur
    if section_end is None or section_end < 8 or not notes:
        return None
    notes = notes[:section_end]
    if (notes[-1][1] % 12) != tonic or t > 70:
        return None
    end = notes[-1][0] + notes[-1][2]
    if end > 66 or len(notes) > 130:
        return None
    return {
        "key": f"{root} {mode}",
        "title": head.get("T", ""),
        "notes": [{"start": round(s, 4), "pitch": p, "duration": round(d, 4)} for s, p, d in notes],
    }


def main(n: int = 50) -> None:
    out = []
    for f in FILES:
        raw = urllib.request.urlopen(BASE + f + ".abc", timeout=30).read().decode("latin-1")
        for tune in re.split(r"\n\s*\n(?=X:)", raw):
            p = parse(tune)
            if p and 8 <= len(p["notes"]):
                p["source"] = f"Nottingham Music Database: {f}.abc"
                out.append(p)
    step = max(1, len(out) // n)
    chosen = out[::step][:n]
    dest = Path("tests/fixtures/reference_melodies.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "provenance": "First sections of folk tunes from the Nottingham Music Database "
                "(public folk-tune collection; ABC transcriptions by Eric Foxley), "
                "https://github.com/jukedeck/nottingham-dataset. Converted by "
                "scripts/build_reference_melodies.py: pitch, start and duration in quarter-note beats.",
                "melodies": chosen,
            },
            indent=1,
        )
    )
    print(f"{len(out)} usable tunes, wrote {len(chosen)} to {dest}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
