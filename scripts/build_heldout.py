"""Build the fixed heldout split (unseen templates + unseen slot values) as JSONL."""

from __future__ import annotations

import sys
from pathlib import Path

from openenv_daw.tasks.generators import families, generate

PER_FAMILY = {
    "session_setup": 12,
    "melody_in_key": 16,
    "four_on_floor_beat": 10,
    "reverb_send": 8,
    "mix_balance": 8,
    "infeasible": 10,
}


def main(out: str = "src/openenv_daw/tasks/heldout.jsonl") -> None:
    seen: set[str] = set()
    lines = []
    for fam in families():
        n, seed = 0, 0
        while n < PER_FAMILY[fam] and seed < 2000:
            t = generate(fam, seed, "heldout")
            seed += 1
            if t.intent in seen:
                continue
            seen.add(t.intent)
            lines.append(t.model_dump_json())
            n += 1
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} heldout tasks to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
