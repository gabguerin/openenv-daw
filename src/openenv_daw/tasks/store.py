"""Task access: train tasks are generated on the fly from seeds; heldout is a fixed JSONL."""

from __future__ import annotations

import json
import os
import random
import re
from functools import cache
from importlib import resources
from pathlib import Path

from .generators import families, generate
from .model import Task

_ID = re.compile(r"^(?P<family>[a-z_]+)-(?P<split>train|heldout)-(?P<seed>\d+)$")


class TaskStore:
    def __init__(self, heldout: list[Task] | None = None) -> None:
        self.heldout = heldout if heldout is not None else load_heldout()
        self._by_id = {t.task_id: t for t in self.heldout}

    def list_splits(self) -> list[str]:
        return ["train", "heldout"]

    def num_tasks(self, split: str) -> int:
        return len(self.heldout) if split == "heldout" else 10**9  # train is unbounded

    def get(self, task_id: str) -> Task:
        if task_id in self._by_id:
            return self._by_id[task_id]
        m = _ID.match(task_id)
        if m and m["split"] == "train" and m["family"] in families():
            return generate(m["family"], int(m["seed"]), "train")
        raise KeyError(f"unknown task {task_id!r}")

    def sample(self, split: str, seed: int = 0, family: str | None = None) -> Task:
        if split == "heldout":
            pool = [t for t in self.heldout if family in (None, t.family)]
            if not pool:
                raise KeyError("no heldout tasks")
            return pool[seed % len(pool)]
        rng = random.Random(f"family:{seed}")
        fams = families()
        fam = family or rng.choices(list(fams), weights=list(fams.values()))[0]
        return generate(fam, seed, "train")

    def get_task(self, split: str, index: int) -> Task:
        if split == "heldout":
            return self.heldout[index]
        return self.sample("train", index)


def heldout_path() -> Path:
    env = os.getenv("OPENENV_DAW_HELDOUT")
    if env:
        return Path(env)
    return Path(str(resources.files("openenv_daw.tasks").joinpath("heldout.jsonl")))


def load_heldout() -> list[Task]:
    p = heldout_path()
    if not p.is_file():
        return []
    return [Task.model_validate(json.loads(line)) for line in p.read_text().splitlines() if line]


@cache
def default_store() -> TaskStore:
    return TaskStore()
