"""SFT export: keep successful episodes only, deduplicated by task + action sequence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_sft(rollouts: Path, out: Path) -> dict[str, int]:
    seen: set[str] = set()
    total = kept = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for line in rollouts.read_text().splitlines():
            if not line.strip():
                continue
            rec: dict[str, Any] = json.loads(line)
            total += 1
            if not rec.get("success"):
                continue
            key = json.dumps([rec["task_id"], rec["actions"]], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            kept += 1
            f.write(json.dumps({"messages": rec["messages"], "tools": rec["tools"]}) + "\n")
    return {"episodes": total, "kept": kept, "dropped": total - kept}
