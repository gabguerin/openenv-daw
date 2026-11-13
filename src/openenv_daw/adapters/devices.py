"""Adapter-owned device maps: canonical catalog device/parameter -> the DAW's own names.

```yaml
devices:
  reverb:
    native: "Reverb"
    params:
      mix: {native: "Dry/Wet", min: 0, max: 100}   # optional min/max: native units of the range
      decay: unsupported
```
Parameters marked ``unsupported`` (or missing) are reported to the agent as ``supported: false``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from openenv_daw.sim.catalog import ParamSpec, load_catalog


@dataclass(frozen=True)
class ParamMap:
    native: str
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True)
class DeviceEntry:
    native: str
    params: dict[str, ParamMap | None] = field(default_factory=dict)  # None: unsupported


@dataclass
class DeviceMap:
    devices: dict[str, DeviceEntry]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DeviceMap:
        out: dict[str, DeviceEntry] = {}
        for did, d in raw.get("devices", {}).items():
            params: dict[str, ParamMap | None] = {}
            for name, p in (d.get("params") or {}).items():
                params[name] = (
                    None
                    if p == "unsupported"
                    else ParamMap(p["native"], p.get("min"), p.get("max"))
                )
            out[did] = DeviceEntry(d["native"], params)
        dm = cls(out)
        dm.validate()
        return dm

    @classmethod
    def from_yaml(cls, path: str | Path) -> DeviceMap:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def validate(self) -> None:
        cat = load_catalog()
        for did, entry in self.devices.items():
            if did not in cat:
                raise ValueError(f"device map: unknown catalog device {did!r}")
            unknown = set(entry.params) - set(cat[did].params)
            if unknown:
                raise ValueError(f"device map: {did!r} has unknown params {sorted(unknown)}")

    def native_device(self, catalog_id: str) -> str | None:
        e = self.devices.get(catalog_id)
        return e.native if e else None

    def catalog_device(self, native_name: str) -> str | None:
        for did, e in self.devices.items():
            if e.native.lower() == native_name.lower():
                return did
        return None

    def param(self, catalog_id: str, param: str) -> ParamMap | None:
        e = self.devices.get(catalog_id)
        return e.params.get(param) if e else None

    def spec(self, catalog_id: str, param: str) -> ParamSpec:
        return load_catalog()[catalog_id].params[param]

    def unsupported_params(self) -> frozenset[str]:
        """``<device>.<param>`` entries the DAW cannot set (whole devices included)."""
        out = set()
        for did, spec in load_catalog().items():
            e = self.devices.get(did)
            for p in spec.params:
                if e is None or e.params.get(p) is None:
                    out.add(f"{did}.{p}")
        return frozenset(out)

    def coverage(self) -> dict[str, Any]:
        cat = load_catalog()
        total = sum(len(d.params) for d in cat.values())
        gaps = self.unsupported_params()
        return {
            "devices": f"{len(self.devices)}/{len(cat)}",
            "params": f"{total - len(gaps)}/{total}",
            "unsupported": sorted(gaps),
        }
