"""Task access: train tasks are generated on the fly from seeds; heldout is a fixed JSONL."""

from __future__ import annotations

import json
import os
import random
import re
from functools import cache
from importlib import resources
from pathlib import Path

from openenv_daw.sim.profiles import DEFAULT_PROFILE

from .generators import families, generate
from .model import Task
from .resolve import resolve_for_profile

_ID = re.compile(r"^(?P<family>[a-z_]+)-(?P<split>train|heldout)-(?P<seed>\d+)$")


class TaskStore:
    def __init__(self, heldout: list[Task] | None = None) -> None:
        self.heldout = heldout if heldout is not None else load_heldout()
        self._by_id = {t.task_id: t for t in self.heldout}

    @classmethod
    def from_paths(cls, *paths: str | Path) -> TaskStore:
        """A store whose heldout split is the union of the given JSONL files."""
        return cls([t for p in paths for t in load_tasks(p)])

    def list_splits(self) -> list[str]:
        return ["train", "heldout"]

    def num_tasks(self, split: str) -> int:
        return len(self.heldout) if split == "heldout" else 10**9  # train is unbounded

    def get(self, task_id: str, profile: str = DEFAULT_PROFILE) -> Task:
        if task_id in self._by_id:
            return resolve_for_profile(self._by_id[task_id], profile)
        m = _ID.match(task_id)
        if m and m["split"] == "train" and m["family"] in families():
            return generate(m["family"], int(m["seed"]), "train", profile)
        raise KeyError(f"unknown task {task_id!r}")

    def sample(
        self,
        split: str,
        seed: int = 0,
        family: str | None = None,
        profile: str = DEFAULT_PROFILE,
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
            return resolve_for_profile(pool[seed % len(pool)], profile)
        rng = random.Random(f"family:{seed}")
        fams = families()
        fam = family or rng.choices(list(fams), weights=list(fams.values()))[0]
        task = generate(fam, seed, "train", profile)
        for i in range(1, 200):
            if set(tags or ()) <= set(task.tags):
                break
            rng = random.Random(f"family:{seed}:{i}")
            fam = family or rng.choices(list(fams), weights=list(fams.values()))[0]
            task = generate(fam, seed + i * 7919, "train", profile)
        return resolve_for_profile(task, profile)

    def get_task(self, split: str, index: int, profile: str = DEFAULT_PROFILE) -> Task:
        if split == "heldout":
            return resolve_for_profile(self.heldout[index], profile)
        return self.sample("train", index, profile=profile)


HELDOUT_DEFAULT = "heldout_v5.jsonl"  # v2, v3 and v4 are frozen (v2 as published in 0.1.0)


def heldout_path() -> Path:
    """``OPENENV_DAW_HELDOUT``: a file path, or a bundled file name such as ``heldout_v2.jsonl``."""
    env = os.getenv("OPENENV_DAW_HELDOUT")
    if env and Path(env).is_file():
        return Path(env)
    name = env or HELDOUT_DEFAULT
    return Path(str(resources.files("openenv_daw.tasks").joinpath(name)))


def load_tasks(path: str | Path) -> list[Task]:
    """Tasks from any JSONL file (for example a private benchmark)."""
    return [Task.model_validate(json.loads(x)) for x in Path(path).read_text().splitlines() if x]


def load_heldout() -> list[Task]:
    p = heldout_path()
    if not p.is_file():
        return []
    return [Task.model_validate(json.loads(line)) for line in p.read_text().splitlines() if line]


@cache
def default_store() -> TaskStore:
    return TaskStore()
