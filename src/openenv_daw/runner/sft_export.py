"""SFT export: keep successful episodes only (no plain-text nudges, musical tasks above a
musicality floor), deduplicated by task + action sequence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MIN_MUSICALITY = 0.6  # musical tasks: a correct but unmusical episode is not worth imitating


def export_sft(rollouts: Path, out: Path, min_musicality: float = MIN_MUSICALITY) -> dict[str, int]:
    seen: set[str] = set()
    total = kept = nudged = unmusical = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for line in rollouts.read_text().splitlines():
            if not line.strip():
                continue
            rec: dict[str, Any] = json.loads(line)
            total += 1
            if not rec.get("success"):
                continue
            if rec.get("nudges"):  # TRL never nudges: the model must learn to call tools unprompted
                nudged += 1
                continue
            if rec.get("musicality") is not None and rec["musicality"] < min_musicality:
                unmusical += 1
                continue
            key = json.dumps([rec["task_id"], rec["actions"]], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            kept += 1
            f.write(json.dumps({"messages": rec["messages"], "tools": rec["tools"]}) + "\n")
    return {
        "episodes": total,
        "kept": kept,
        "dropped": total - kept,
        "nudged": nudged,
        "unmusical": unmusical,
    }
