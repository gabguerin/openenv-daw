"""Adapter-owned device maps: canonical catalog device/parameter -> the DAW's own names.

```yaml
devices:
  reverb:
    native: "Reverb"
    locator: "audio_effects/Reverb"   # optional: where the DAW's browser holds it
    params:
      mix: {native: "Dry/Wet", min: 0, max: 100}   # optional min/max: native units of the range
      decay: unsupported
library:                            # optional: named presets, racks and plugins
  sub_bass: {native: "Sub Bass", locator: "instruments/Drift/Bass/Sub Bass"}
```
Parameters marked ``unsupported`` (or missing) are reported to the agent as ``supported: false``.
Library items that are not mapped are hidden from the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from openenv_daw.sim.catalog import ParamSpec, load_catalog, load_library


_SUFFIXES = (".adv", ".adg", ".adp", ".amxd", ".vstpreset", ".fxp")


def _bare(name: str) -> str:
    """A DAW item name without a preset-file extension, lower case."""
    low = name.lower()
    return next((low[: -len(s)] for s in _SUFFIXES if low.endswith(s)), low)


@dataclass(frozen=True)
class ParamMap:
    native: str
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True)
class DeviceEntry:
    native: str
    params: dict[str, ParamMap | None] = field(default_factory=dict)  # None: unsupported
    locator: str | None = None  # opaque hint for the driver to find the item without searching


@dataclass
class DeviceMap:
    devices: dict[str, DeviceEntry]
    # library id -> how to load that preset, rack or plugin (``params`` is unused: a preset's
    # parameters are the base device's, a rack or plugin has none)
    library: dict[str, DeviceEntry] = field(default_factory=dict)

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
            out[did] = DeviceEntry(d["native"], params, d.get("locator"))
        lib = {
            iid: DeviceEntry(d["native"], {}, d.get("locator"))
            for iid, d in (raw.get("library") or {}).items()
        }
        dm = cls(out, lib)
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
        items = load_library()
        for iid in self.library:
            if iid not in items:
                raise ValueError(f"device map: unknown library item {iid!r}")
            base = items[iid].base
            if base is not None and base not in self.devices:
                raise ValueError(f"device map: {iid!r} is a preset of unmapped device {base!r}")

    def native_device(self, catalog_id: str) -> str | None:
        e = self.devices.get(catalog_id)
        return e.native if e else None

    def entries(self) -> list[tuple[str, DeviceEntry]]:
        """Every mapped catalog device and library item with its entry."""
        return [*self.devices.items(), *self.library.items()]

    def locator(self, catalog_id: str) -> str | None:
        e = self.devices.get(catalog_id) or self.library.get(catalog_id)
        return e.locator if e else None

    def native_item(self, library_id: str) -> str | None:
        e = self.library.get(library_id)
        return e.native if e else None

    def library_item(
        self, native_name: str, base: str | None = None, foreign: bool = False
    ) -> str | None:
        """The library id a native device name stands for: a preset of catalog device ``base``,
        or (``foreign``) a rack or plugin."""
        low = _bare(native_name)
        items = load_library()
        for iid, e in self.library.items():
            if _bare(e.native) != low and _bare(items[iid].name) != low:
                continue
            wanted_base = None if foreign else base
            if (foreign or base is not None) and items[iid].base == wanted_base:
                return iid
        return None

    def unsupported_items(self) -> frozenset[str]:
        """Library ids this DAW has no mapping for (hidden from the agent)."""
        return frozenset(set(load_library()) - set(self.library))

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

    def masked(self) -> frozenset[str]:
        """Everything the DAW cannot do, in the form the simulator masks: ``<device>.<param>``
        and ``item:<library id>`` entries."""
        return self.unsupported_params() | {f"item:{i}" for i in self.unsupported_items()}

    def coverage(self) -> dict[str, Any]:
        cat = load_catalog()
        total = sum(len(d.params) for d in cat.values())
        gaps = self.unsupported_params()
        return {
            "devices": f"{len(self.devices)}/{len(cat)}",
            "library": f"{len(self.library)}/{len(load_library())}",
            "params": f"{total - len(gaps)}/{total}",
            "unsupported": sorted(gaps),
        }
