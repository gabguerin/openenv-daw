"""Build the fixed heldout split (unseen templates + unseen slot values) as JSONL, in profile-neutral form:
plugin tasks are loadable and resolve_for_profile blocks them where the profile cannot load them.

uv run python scripts/build_heldout.py [out.jsonl]   # default: heldout_v6.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

from openenv_daw.tasks.generators import families, generate

PER_FAMILY = {  # at least 15 per family, so per-family rates have a usable interval
    "session_setup": 15,
    "melody_in_key": 16,
    "four_on_floor_beat": 15,
    "reverb_send": 15,
    "mix_balance": 15,
    "near_miss": 15,
    "infeasible": 15,
    "chord_progression": 16,
    "edit_notes": 16,
    "device_tweak": 16,
    "disambiguation": 15,
    "compositional": 28,
    "load_sound": 15,
}


def main(out: str = "src/openenv_daw/tasks/heldout_v6.jsonl") -> None:
    seen: set[str] = set()
    lines = []
    for fam in families():
        n, seed = 0, 0
        while n < PER_FAMILY[fam] and seed < 3000:
            t = generate(fam, seed, "heldout", None)  # profile-neutral
            seed += 1
            if t.intent in seen:
                continue
            seen.add(t.intent)
            lines.append(t.model_dump_json())
            n += 1
        if n < PER_FAMILY[fam]:
            raise SystemExit(f"{fam}: only {n} unique heldout tasks, wanted {PER_FAMILY[fam]}")
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} heldout tasks to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
