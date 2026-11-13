"""Canonical device catalog loaded from catalog/devices.yaml."""

from __future__ import annotations

import math
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
    daw_ids: dict[str, Any] = field(default_factory=dict)


@cache
def load_catalog() -> dict[str, DeviceSpec]:
    raw = yaml.safe_load(
        resources.files("openenv_daw.catalog").joinpath("devices.yaml").read_text()
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
        out[did] = DeviceSpec(did, d["role"], d["description"], params, d.get("daw_ids", {}))
    return out
