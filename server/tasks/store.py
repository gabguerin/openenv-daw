"""Task access: train tasks are generated on the fly from seeds; heldout is a fixed JSONL."""

from __future__ import annotations

import json
import random
import re
from functools import cache
from importlib import resources
from pathlib import Path

from .generators import families, generate
from .model import Task

_ID = re.compile(r"^(?P<family>[a-z_]+)-(?P<split>train|heldout)-(?P<seed>\d+)$")
HELDOUT_FILE = "heldout.jsonl"


def load_tasks(path: str | Path) -> list[Task]:
    """Tasks from any JSONL file (for example a private benchmark)."""
    return [Task.model_validate(json.loads(x)) for x in Path(path).read_text().splitlines() if x]


def load_heldout() -> list[Task]:
    path = Path(str(resources.files("openenv_daw.server.tasks").joinpath(HELDOUT_FILE)))
    return load_tasks(path) if path.is_file() else []


class TaskStore:
    def __init__(self, heldout: list[Task] | None = None) -> None:
        self.heldout = heldout if heldout is not None else load_heldout()
        self._by_id = {t.task_id: t for t in self.heldout}

    @classmethod
    def from_paths(cls, *paths: str | Path) -> TaskStore:
        """A store whose heldout split is the union of the given JSONL files."""
        return cls([t for p in paths for t in load_tasks(p)])

    def get(self, task_id: str) -> Task:
        if task_id in self._by_id:
            return self._by_id[task_id]
        m = _ID.match(task_id)
        if m and m["split"] == "train" and m["family"] in families():
            return generate(m["family"], int(m["seed"]), "train")
        raise KeyError(f"unknown task {task_id!r}")

    def sample(
        self,
        split: str,
        seed: int = 0,
        family: str | None = None,
        tags: list[str] | None = None,
    ) -> Task:
        """A task of ``split``; ``tags`` (for example ``["L2"]``) keeps heldout tasks that carry
        all of them and, for train, regenerates (up to 200 seeds ahead) until one matches."""
        if split == "heldout":
            pool = [
                t
                for t in self.heldout
                if family in (None, t.family) and set(tags or ()) <= set(t.tags)
            ]
            if not pool:
                raise KeyError("no heldout tasks")
            return pool[seed % len(pool)]
        fams = families()
        task = None
        for i in range(200):
            rng = random.Random(f"family:{seed}" if i == 0 else f"family:{seed}:{i}")
            fam = family or rng.choices(list(fams), weights=list(fams.values()))[0]
            task = generate(fam, seed + i * 7919, "train")
            if set(tags or ()) <= set(task.tags):
                break
        assert task is not None
        return task

    def get_task(self, split: str, index: int) -> Task:
        if split == "heldout":
            return self.heldout[index]
        return self.sample("train", index)


@cache
def default_store() -> TaskStore:
    return TaskStore()
