"""Damage: changes to pre-existing objects that the task did not declare touchable."""

from __future__ import annotations

from typing import Any

from openenv_daw.server.sim.session import Session, Track

ASPECTS = ("mixer", "sends", "clips", "devices", "foreign_devices", "foreign_clips")


def _aspect(t: Track, aspect: str) -> Any:
    d = t.model_dump(mode="json")
    if aspect == "mixer":
        return {k: d[k] for k in ("name", "kind", "volume_db", "pan", "mute", "solo")}
    if aspect == "devices":  # a DAW may round-trip normalized values with float noise
        return [
            {**x, "params": {k: round(v, 6) for k, v in x["params"].items()}} for x in d[aspect]
        ]
    if aspect == "foreign_devices":  # read-only; a DAW reports its own parameters for them
        return [{k: v for k, v in f.items() if k != "params"} for f in d[aspect]]
    return d[aspect]


def _allowed(touchable: list[str], t: Track, aspect: str) -> bool:
    if "*" in touchable:
        return True
    keys = {f"track:{t.name}", f"track:{t.name}:{aspect}"}
    return bool(keys & set(touchable))


def damage_violations(initial: Session, final: Session, touchable: list[str]) -> list[str]:
    """Human-readable list of untouchable things that changed."""
    out: list[str] = []
    if "*" not in touchable and "transport" not in touchable:
        if (initial.tempo_bpm, tuple(initial.time_signature)) != (
            final.tempo_bpm,
            tuple(final.time_signature),
        ):
            out.append("transport")
    final_by_id = {t.id: t for t in final.tracks}
    for t in initial.tracks:
        f = final_by_id.get(t.id)
        if f is None:
            if not _allowed(touchable, t, "*") and f"track:{t.name}" not in touchable:
                out.append(f"track {t.name} deleted")
            continue
        for aspect in ASPECTS:
            if _aspect(t, aspect) != _aspect(f, aspect) and not _allowed(touchable, t, aspect):
                out.append(f"track {t.name} {aspect}")
    return out


def _clip_budget(allowed: set[str]) -> dict[str, int] | None:
    """``clip:<track>[:<n>]`` entries of ``allowed_new``: clips a new track may hold (default
    1). None when the task declares no clip budget (older tasks), so nothing is checked."""
    entries = [a for a in allowed if a.startswith("clip:")]
    if not entries:
        return None
    out: dict[str, int] = {}
    for a in entries:
        name, _, n = a[5:].rpartition(":")
        if name and n.isdigit():
            out[name] = int(n)
        else:
            out[a[5:]] = 1
    return out


def extraneous_violations(initial: Session, final: Session, task: Any) -> list[str]:
    """Objects the task did not ask for: unlisted new tracks or devices, solo anywhere, and a new
    track fader above 0 dB. ``task.allowed_new`` lists track names, ``device:<id>`` entries and ``solo:<track>`` entries
    (for tasks that explicitly ask for a solo)."""
    allowed = {a.lower() for a in task.allowed_new}
    out: list[str] = []
    budget = _clip_budget(allowed)
    old = {t.id: t for t in initial.tracks}
    for t in final.tracks:
        before = old.get(t.id)
        if before is None:
            if t.name.lower() not in allowed:
                out.append(f"extra track {t.name}")
            if budget is not None:
                extra = len(t.clips) - budget.get(t.name.lower(), 0)
                out.extend(f"extra clip on {t.name}" for _ in range(max(extra, 0)))
            if t.volume_db > 0.0:
                out.append(f"new track {t.name} fader above 0 dB")
        if t.solo and not (before and before.solo) and f"solo:{t.name}".lower() not in allowed:
            out.append(f"track {t.name} soloed")
        known = {d.id for d in before.devices} if before else set()
        for d in t.devices:
            if d.id not in known and f"device:{d.device}".lower() not in allowed:
                out.append(f"extra device {d.device} on {t.name}")
        known_foreign = {d.id for d in before.foreign_devices} if before else set()
        for f in t.foreign_devices:  # racks and plugins loaded from the library
            if f.id not in known_foreign and f"device:{f.item or f.name}".lower() not in allowed:
                out.append(f"extra device {f.item or f.name} on {t.name}")
    return out
