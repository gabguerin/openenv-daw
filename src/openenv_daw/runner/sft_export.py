"""SFT export: filter collected episodes by the environment's verdict (success, musicality floor,
deduplicated by task + action sequence) into chat rows for TRL's ``SFTTrainer``:
``{"messages", "tools", "reward", "task_id"}``; ``tools`` is the JSON string TRL accepts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from openenv_daw.harness import read_results
from openenv_daw.sim.profiles import DEFAULT_PROFILE

MIN_MUSICALITY = 0.6  # musical tasks: a correct but unmusical episode is not worth imitating


def export_sft(
    rollouts: Path | Iterable[Path], out: Path, min_musicality: float = MIN_MUSICALITY
) -> dict[str, int]:
    from openenv_daw.trl_env import tool_schemas

    dirs = [rollouts] if isinstance(rollouts, Path) else list(rollouts)
    seen: set[str] = set()
    total = kept = unmusical = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for d in dirs:
            for rec in read_results(d):
                total += 1
                m: dict[str, Any] = rec["verify_metrics"]
                if not m.get("success"):
                    continue
                if m.get("musicality") is not None and m["musicality"] < min_musicality:
                    unmusical += 1
                    continue
                key = json.dumps([rec["task"], m["actions"]], sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                kept += 1
                profile = rec.get("extra", {}).get("profile") or DEFAULT_PROFILE
                f.write(
                    json.dumps(
                        {
                            "messages": rec["messages"],
                            "tools": json.dumps(list(tool_schemas(profile))),
                            "reward": rec["reward"],
                            "task_id": rec["task"],
                        }
                    )
                    + "\n"
                )
    return {"episodes": total, "kept": kept, "dropped": total - kept, "unmusical": unmusical}
