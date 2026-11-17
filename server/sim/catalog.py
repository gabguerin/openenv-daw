"""Canonical device catalog loaded from devices.yaml."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from typing import Any

import yaml


@dataclass(frozen=True)
class ParamSpec:
    name: str
    default: float
    min: float
    max: float
    scale: str
    unit: str
    steps: tuple[str, ...] = ()

    def to_value(self, norm: float) -> float:
        """Numeric display value for a normalized 0..1 position (step index for ``steps``)."""
        norm = min(max(norm, 0.0), 1.0)
        if self.scale == "steps":
            return float(min(int(norm * len(self.steps)), len(self.steps) - 1))
        if self.scale == "log":
            return self.min * math.pow(self.max / self.min, norm)
        return self.min + (self.max - self.min) * norm

    def to_norm(self, value: float) -> float:
        """Inverse of ``to_value`` (step centers for ``steps``)."""
        if self.scale == "steps":
            return (min(max(int(value), 0), len(self.steps) - 1) + 0.5) / len(self.steps)
        value = min(max(value, self.min), self.max)
        if self.scale == "log":
            return math.log(value / self.min) / math.log(self.max / self.min)
        return (value - self.min) / (self.max - self.min)

    def display(self, norm: float) -> str:
        if self.scale == "steps":
            return self.steps[min(int(norm * len(self.steps)), len(self.steps) - 1)]
        if self.scale == "log":
            v = self.min * math.pow(self.max / self.min, norm)
        else:
            v = self.min + (self.max - self.min) * norm
        text = f"{v:.0f}" if abs(v) >= 100 else f"{v:.2f}".rstrip("0").rstrip(".")
        return f"{text} {self.unit}".strip()


@dataclass(frozen=True)
class DeviceSpec:
    id: str
    role: str
    description: str
    params: dict[str, ParamSpec] = field(default_factory=dict)


@cache
def load_catalog() -> dict[str, DeviceSpec]:
    raw = yaml.safe_load(
        resources.files("openenv_daw.server.sim").joinpath("devices.yaml").read_text()
    )
    out: dict[str, DeviceSpec] = {}
    for did, d in raw["devices"].items():
        params = {
            n: ParamSpec(
                name=n,
                default=float(p["default"]),
                min=float(p["min"]),
                max=float(p["max"]),
                scale=p["scale"],
                unit=p.get("unit", ""),
                steps=tuple(p.get("steps", ())),
            )
            for n, p in d["params"].items()
        }
        out[did] = DeviceSpec(did, d["role"], d["description"], params)
    return out


# --- library: named presets, racks and plugins -------------------------------------------------


@dataclass(frozen=True)
class LibraryItem:
    id: str
    name: str
    kind: str  # preset | rack | plugin
    role: str  # instrument | audio_fx
    base: str | None = None  # catalog device a preset is built on
    tags: tuple[str, ...] = ()
    overrides: dict[str, float] = field(default_factory=dict)  # normalized 0..1
    params: tuple[tuple[str, str], ...] = ()  # (name, display) of read-only items


@cache
def load_library() -> dict[str, LibraryItem]:
    raw = yaml.safe_load(
        resources.files("openenv_daw.server.sim").joinpath("library.yaml").read_text()
    )
    cat = load_catalog()
    out: dict[str, LibraryItem] = {}
    names: set[str] = set()
    for iid, d in raw["items"].items():
        if iid in cat:
            raise ValueError(f"library id {iid!r} clashes with a catalog device")
        kind = d["kind"]
        base = d.get("base")
        if kind == "preset":
            if base not in cat:
                raise ValueError(f"library {iid!r}: unknown base {base!r}")
            unknown = set(d.get("overrides", {})) - set(cat[base].params)
            if unknown:
                raise ValueError(f"library {iid!r}: unknown params {sorted(unknown)}")
            role = cat[base].role
        elif kind in ("rack", "plugin"):
            base, role = None, d["role"]
        else:
            raise ValueError(f"library {iid!r}: unknown kind {kind!r}")
        if d["name"].lower() in names:
            raise ValueError(f"library name {d['name']!r} is used twice")
        names.add(d["name"].lower())
        out[iid] = LibraryItem(
            id=iid,
            name=d["name"],
            kind=kind,
            role=role,
            base=base,
            tags=tuple(str(t).lower() for t in d.get("tags", ())),  # YAML reads 808 as an int
            overrides={k: float(v) for k, v in d.get("overrides", {}).items()},
            params=tuple((p["name"], str(p["display"])) for p in d.get("params", ())),
        )
    return out


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def find_item(ref: str) -> LibraryItem | None:
    """A library item by id or exact (case-insensitive) name."""
    lib = load_library()
    if ref in lib:
        return lib[ref]
    low = ref.strip().lower()
    return next((i for i in lib.values() if i.name.lower() == low), None)


def search_devices(query: str, role: str | None = None, limit: int = 8) -> list[dict[str, Any]]:
    """Catalog devices and library items ranked by how well ``query`` matches their id, name and
    tags. Hits are ``{device, name, role, kind, base?}``; ``device`` is what insert_device takes."""
    q = _tokens(query)
    rows: list[tuple[int, str, dict[str, Any]]] = []
    entries: list[tuple[dict[str, Any], list[str], list[str]]] = []
    for d in load_catalog().values():
        entries.append(
            (
                {"device": d.id, "name": d.id, "role": d.role, "kind": "device"},
                _tokens(d.id),
                _tokens(d.description),
            )
        )
    for i in load_library().values():
        row = {"device": i.id, "name": i.name, "role": i.role, "kind": i.kind}
        if i.base:
            row["base"] = i.base
        entries.append((row, _tokens(i.id) + _tokens(i.name), list(i.tags) + _tokens(i.base or "")))
    for row, name_toks, tag_toks in entries:
        if role and row["role"] != role:
            continue
        score = 0
        for t in q:
            if t in name_toks:
                score += 3
            elif t in tag_toks:
                score += 2
            elif any(t in x for x in name_toks + tag_toks if len(t) >= 3):
                score += 1
        if q and score:
            if " ".join(q) == " ".join(_tokens(row["name"])):
                score += 5
            rows.append((score, row["device"], row))
    rows.sort(key=lambda r: (-r[0], r[1]))
    return [r[2] for r in rows[:limit]]
