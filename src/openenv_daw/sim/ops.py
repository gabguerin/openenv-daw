"""Pure state mutations of the JSON DAW simulator. Every function takes a Session and plain
arguments, mutates the session and returns a JSON-serializable result dict with ``ok: True``.
Errors are raised as ``SimError``."""

from __future__ import annotations

from typing import Any

from openenv_daw.schema import Clip, Device, ForeignDevice, Note, Session, Track
from openenv_daw.schema.units import (
    DB_MAX,
    DB_MIN,
    GRID,
    GRID_DIVISIONS,
    MAX_NOTES_PER_CALL,
    PAN_MAX,
    PAN_MIN,
    PITCH_MAX,
    PITCH_MIN,
    VEL_MAX,
    VEL_MIN,
    clamp,
    r2,
    snap,
)

from .catalog import DeviceSpec, load_catalog
from .errors import conflict, invalid, not_found, unsupported

# --- resolution ----------------------------------------------------------------------------


def find_track(s: Session, ref: str) -> Track:
    for t in s.tracks:
        if t.id == ref:
            return t
    named = [t for t in s.tracks if t.name.lower() == ref.lower()]
    if len(named) == 1:
        return named[0]
    raise not_found(f"no track {ref!r}") if not named else invalid(f"ambiguous track {ref!r}")


def find_clip(s: Session, ref: str) -> tuple[Track, Clip]:
    hits = [(t, c) for t in s.tracks for c in t.clips if c.id == ref]
    if not hits:
        hits = [(t, c) for t in s.tracks for c in t.clips if c.name.lower() == ref.lower()]
    if not hits:
        for t in s.tracks:
            for fc in t.foreign_clips:
                if ref in (fc.id,) or ref.lower() == fc.name.lower():
                    raise unsupported(
                        f"clip {fc.name!r} ({fc.id}) is a {fc.kind} clip; it is read-only here"
                    )
        raise not_found(f"no clip {ref!r}")
    if len(hits) > 1:
        raise invalid(f"ambiguous clip {ref!r}; use its id")
    return hits[0]


def find_foreign_device(t: Track, ref: str) -> ForeignDevice | None:
    for d in t.foreign_devices:
        if d.id == ref or d.name.lower() == ref.lower():
            return d
    return None


def _writable(t: Track) -> Track:
    if t.kind == "group":
        raise unsupported(f"track {t.name!r} is a group track; it is read-only here")
    return t


def find_device(t: Track, ref: str) -> Device:
    for d in t.devices:
        if d.id == ref:
            return d
    hits = [d for d in t.devices if ref.lower() in (d.device.lower(), d.name.lower())]
    if not hits:
        if (fd := find_foreign_device(t, ref)) is not None:
            raise unsupported(
                f"device {fd.name!r} ({fd.id}) is foreign; its parameters are read-only"
            )
        raise not_found(f"no device {ref!r} on track {t.name!r}")
    if len(hits) > 1:
        raise invalid(f"ambiguous device {ref!r} on track {t.name!r}; use its id")
    return hits[0]


def _check_name(s: Session, name: str, exclude: Track | None = None) -> str:
    name = name.strip()
    if not name:
        raise invalid("name must not be empty")
    if any(t.name.lower() == name.lower() and t is not exclude for t in s.tracks):
        raise conflict(f"a track named {name!r} already exists")
    return name


# --- views ----------------------------------------------------------------------------------


def note_view(n: Note) -> dict[str, Any]:
    return n.model_dump()


def clip_view(c: Clip, track: Track | None = None, notes: bool = False) -> dict[str, Any]:
    v: dict[str, Any] = {
        "id": c.id,
        "name": c.name,
        "start_beat": c.start_beat,
        "length_beats": c.length_beats,
        "note_count": len(c.notes),
    }
    if track is not None:
        v["track"] = track.id
    if notes:
        v["notes"] = [note_view(n) for n in c.notes]
    return v


def device_view(d: Device, params: bool = False) -> dict[str, Any]:
    v: dict[str, Any] = {"id": d.id, "device": d.device, "name": d.name, "role": d.role}
    if params:
        v["params"] = param_infos(d)
    return v


def param_infos(d: Device, query: str | None = None) -> list[dict[str, Any]]:
    spec = load_catalog()[d.device]
    out = []
    for name, ps in spec.params.items():
        if query and query.lower() not in name.lower():
            continue
        val = d.params.get(name, ps.default)
        out.append(
            {
                "name": name,
                "value": round(val, 4),
                "display": ps.display(val),
                "default": ps.default,
                "unit": ps.unit,
            }
        )
    return out


def track_view(t: Track, detail: bool = True, notes: bool = False) -> dict[str, Any]:
    v: dict[str, Any] = {
        "id": t.id,
        "name": t.name,
        "kind": t.kind,
        "volume_db": t.volume_db,
        "pan": t.pan,
        "mute": t.mute,
        "solo": t.solo,
        "sends": dict(t.sends),
        "devices": [device_view(d, params=detail) for d in t.devices],
        "clips": [clip_view(c, notes=notes) for c in t.clips],
    }
    if t.foreign_devices:
        v["foreign_devices"] = [
            {
                "id": d.id,
                "name": d.name,
                **({"params": [p.model_dump() for p in d.params]} if detail else {}),
            }
            for d in t.foreign_devices
        ]
    if t.foreign_clips:
        v["foreign_clips"] = [c.model_dump() for c in t.foreign_clips]
    return v


# --- operations -----------------------------------------------------------------------------


def get_session(
    s: Session, track: str | None = None, include_notes: bool = False
) -> dict[str, Any]:
    if track is not None:
        t = find_track(s, track)
        return {"ok": True, "track": track_view(t, detail=True, notes=include_notes)}
    return {
        "ok": True,
        "tempo_bpm": s.tempo_bpm,
        "time_signature": list(s.time_signature),
        "tracks": [track_view(t, detail=False, notes=include_notes) for t in s.tracks],
    }


def set_transport(
    s: Session,
    tempo_bpm: float | None = None,
    time_signature: tuple[int, int] | list[int] | None = None,
) -> dict[str, Any]:
    if tempo_bpm is None and time_signature is None:
        raise invalid("provide tempo_bpm and/or time_signature")
    if tempo_bpm is not None:
        if not 20 <= tempo_bpm <= 300:
            raise invalid("tempo_bpm must be within 20..300")
        s.tempo_bpm = r2(float(tempo_bpm))
    if time_signature is not None:
        if len(time_signature) != 2:
            raise invalid("time_signature must be [numerator, denominator]")
        num, den = int(time_signature[0]), int(time_signature[1])
        if not 1 <= num <= 32 or den not in (1, 2, 4, 8, 16, 32):
            raise invalid("numerator must be 1..32 and denominator one of 1,2,4,8,16,32")
        s.time_signature = (num, den)
    return {"ok": True, "tempo_bpm": s.tempo_bpm, "time_signature": list(s.time_signature)}


def create_track(s: Session, name: str, kind: str) -> dict[str, Any]:
    if kind not in ("midi", "audio", "return"):
        raise invalid("kind must be one of midi, audio, return")
    name = _check_name(s, name)
    t = Track(id=s.next_id("trk"), name=name, kind=kind)  # type: ignore[arg-type]
    if kind == "return":
        s.tracks.append(t)
    else:  # regular tracks stay before the return tracks
        idx = next((i for i, x in enumerate(s.tracks) if x.kind == "return"), len(s.tracks))
        s.tracks.insert(idx, t)
    return {"ok": True, "track": track_view(t)}


def delete_track(s: Session, track: str) -> dict[str, Any]:
    t = find_track(s, track)
    s.tracks.remove(t)
    for o in s.tracks:
        o.sends.pop(t.id, None)
    return {"ok": True, "deleted": t.id}


def rename_track(s: Session, track: str, name: str) -> dict[str, Any]:
    t = _writable(find_track(s, track))
    t.name = _check_name(s, name, exclude=t)
    return {"ok": True, "track": {"id": t.id, "name": t.name}}


def create_midi_clip(
    s: Session, track: str, start_beat: float, length_beats: float, name: str | None = None
) -> dict[str, Any]:
    t = find_track(s, track)
    if t.kind != "midi":
        raise invalid(f"track {t.name!r} is a {t.kind} track; MIDI clips need a midi track")
    if start_beat < 0:
        raise invalid("start_beat must be >= 0")
    start, length = snap(start_beat), snap(length_beats)
    if length < GRID:
        raise invalid(f"length_beats must be at least 1/{GRID_DIVISIONS} beat")
    end = start + length
    for c in t.clips:
        if start < c.end_beat and c.start_beat < end:
            raise conflict(
                f"clip would overlap {c.name!r} ({c.id}) at beats {c.start_beat}..{c.end_beat}"
            )
    cid = s.next_id("clp")
    clip = Clip(
        id=cid,
        name=(name or f"Clip {cid.split('_')[1]}").strip(),
        start_beat=start,
        length_beats=length,
    )
    t.clips.append(clip)
    t.clips.sort(key=lambda c: c.start_beat)
    return {
        "ok": True,
        "clip": clip_view(clip, t),
        "snapped": (start, length) != (start_beat, length_beats),
    }


def delete_clip(s: Session, clip: str) -> dict[str, Any]:
    t, c = find_clip(s, clip)
    t.clips.remove(c)
    return {"ok": True, "deleted": c.id}


def insert_midi_notes(s: Session, clip: str, notes: list[dict[str, Any]]) -> dict[str, Any]:
    _, c = find_clip(s, clip)
    if not notes:
        raise invalid("notes must not be empty")
    if len(notes) > MAX_NOTES_PER_CALL:
        raise invalid(f"at most {MAX_NOTES_PER_CALL} notes per call")
    new: list[Note] = []
    for i, raw in enumerate(notes):
        try:
            pitch = int(raw["pitch"])
            start = float(raw["start_beat"])
            dur = float(raw["duration_beats"])
            vel = int(raw.get("velocity", 100))
        except (KeyError, TypeError, ValueError) as e:
            raise invalid(f"note {i}: needs pitch, start_beat, duration_beats ({e})") from None
        if not PITCH_MIN <= pitch <= PITCH_MAX:
            raise invalid(f"note {i}: pitch must be 0..127")
        if not VEL_MIN <= vel <= VEL_MAX:
            raise invalid(f"note {i}: velocity must be 1..127")
        if start < 0:
            raise invalid(f"note {i}: start_beat must be >= 0")
        st, du = snap(start), max(snap(dur), GRID)
        if st + du > c.length_beats + 1e-9:
            raise invalid(
                f"note {i}: ends at beat {st + du} but the clip is {c.length_beats} beats long"
            )
        new.append(Note(pitch=pitch, start_beat=st, duration_beats=du, velocity=vel))
    existing = list(c.notes)
    for n in new:
        for o in existing:
            if (
                o.pitch == n.pitch
                and n.start_beat < o.start_beat + o.duration_beats
                and (o.start_beat < n.start_beat + n.duration_beats)
            ):
                raise conflict(
                    f"pitch {n.pitch} at beat {n.start_beat} overlaps an existing note of the same pitch"
                )
        existing.append(n)
    c.notes = sorted(existing, key=lambda n: (n.start_beat, n.pitch))
    return {
        "ok": True,
        "clip": c.id,
        "added": [note_view(n) for n in new],
        "note_count": len(c.notes),
    }


def remove_midi_notes(
    s: Session,
    clip: str,
    pitch: int | None = None,
    start_beat: float | None = None,
    end_beat: float | None = None,
) -> dict[str, Any]:
    _, c = find_clip(s, clip)
    keep, removed = [], 0
    for n in c.notes:
        hit = (
            (pitch is None or n.pitch == pitch)
            and (start_beat is None or n.start_beat >= start_beat - 1e-9)
            and (end_beat is None or n.start_beat < end_beat - 1e-9)
        )
        if hit:
            removed += 1
        else:
            keep.append(n)
    c.notes = keep
    return {"ok": True, "clip": c.id, "removed": removed, "note_count": len(keep)}


def set_mixer_channel(
    s: Session,
    track: str,
    volume_db: float | None = None,
    pan: float | None = None,
    mute: bool | None = None,
    solo: bool | None = None,
) -> dict[str, Any]:
    t = _writable(find_track(s, track))
    if volume_db is None and pan is None and mute is None and solo is None:
        raise invalid("provide at least one of volume_db, pan, mute, solo")
    if volume_db is not None:
        t.volume_db = r2(clamp(float(volume_db), DB_MIN, DB_MAX))
    if pan is not None:
        t.pan = r2(clamp(float(pan), PAN_MIN, PAN_MAX))
    if mute is not None:
        t.mute = bool(mute)
    if solo is not None:
        t.solo = bool(solo)
    return {
        "ok": True,
        "track": {
            "id": t.id,
            "name": t.name,
            "volume_db": t.volume_db,
            "pan": t.pan,
            "mute": t.mute,
            "solo": t.solo,
        },
    }


def set_send(s: Session, track: str, return_track: str, level_db: float) -> dict[str, Any]:
    t = _writable(find_track(s, track))
    r = find_track(s, return_track)
    if r.kind != "return":
        raise invalid(f"{r.name!r} is not a return track")
    if t.kind == "return":
        raise invalid("return tracks cannot send to other return tracks")
    level = r2(clamp(float(level_db), DB_MIN, DB_MAX))
    if level <= DB_MIN:
        t.sends.pop(r.id, None)
    else:
        t.sends[r.id] = level
    return {"ok": True, "track": t.id, "return_track": r.id, "level_db": level}


def list_devices(s: Session, role: str | None = None) -> dict[str, Any]:
    if role not in (None, "instrument", "audio_fx"):
        raise invalid("role must be instrument or audio_fx")
    devs = [
        {"device": d.id, "role": d.role, "description": d.description, "params": list(d.params)}
        for d in load_catalog().values()
        if role is None or d.role == role
    ]
    return {"ok": True, "devices": devs}


def insert_device(
    s: Session, track: str, device: str, position: int | None = None
) -> dict[str, Any]:
    t = _writable(find_track(s, track))
    spec: DeviceSpec | None = load_catalog().get(device)
    if spec is None:
        raise not_found(f"no device {device!r}; call list_devices")
    if spec.role == "instrument":
        if t.kind != "midi":
            raise invalid(f"instruments go on midi tracks; {t.name!r} is a {t.kind} track")
        if any(d.role == "instrument" for d in t.devices):
            raise conflict(f"track {t.name!r} already has an instrument")
    n_instr = sum(1 for d in t.devices if d.role == "instrument")
    if spec.role == "instrument":
        pos = 0
    else:
        pos = len(t.devices) if position is None else int(position)
        pos = int(clamp(pos, n_instr, len(t.devices)))
    dev = Device(
        id=s.next_id("dev"),
        device=spec.id,
        name=spec.id,
        role=spec.role,  # type: ignore[arg-type]
        params={n: p.default for n, p in spec.params.items()},
    )
    t.devices.insert(pos, dev)
    return {"ok": True, "track": t.id, "position": pos, "device": device_view(dev, params=True)}


def get_device_params(
    s: Session, track: str, device: str, query: str | None = None
) -> dict[str, Any]:
    t = find_track(s, track)
    if (
        not any(d.id == device for d in t.devices)
        and (fd := find_foreign_device(t, device)) is not None
    ):
        ps = [p.model_dump() for p in fd.params if not query or query.lower() in p.name.lower()]
        return {"ok": True, "track": t.id, "device": fd.id, "foreign": True, "params": ps}
    d = find_device(t, device)
    return {"ok": True, "track": t.id, "device": d.id, "params": param_infos(d, query)}


def set_device_param(
    s: Session, track: str, device: str, param: str, value: float
) -> dict[str, Any]:
    t = find_track(s, track)
    d = find_device(t, device)
    spec = load_catalog()[d.device]
    match = [n for n in spec.params if n.lower() == param.lower()]
    if not match:
        raise not_found(f"device {d.device!r} has no parameter {param!r}; have {list(spec.params)}")
    if not 0.0 <= float(value) <= 1.0:
        raise invalid("value must be normalized within 0..1")
    d.params[match[0]] = round(float(value), 4)
    return {
        "ok": True,
        "track": t.id,
        "device": d.id,
        "param": next(p for p in param_infos(d) if p["name"] == match[0]),
    }
