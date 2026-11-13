"""Canonical device catalog loaded from catalog/devices.yaml."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cache
from importlib import resources

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
        out[did] = DeviceSpec(did, d["role"], d["description"], params)
    return out
