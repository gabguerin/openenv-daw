"""A closed vocabulary of basic edits, derived from the difference between two ``Session`` states.

The simulator defines what every tool does. An adapter never re-implements tool rules: it only
has to carry out these edits in the DAW (``Driver.apply``). Changes use canonical ids; objects a
change creates carry the id the mirror assigned, which the ``IdMap`` binds to the native key.

``apply_changes`` is the reference semantics, so ``apply_changes(a, diff(a, b)) == b`` (counters
aside) is the property every driver has to preserve."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from openenv_daw.schema import Clip, Device, ForeignDevice, ForeignParam, Note, Session, Track
from openenv_daw.sim.catalog import find_item


@dataclass(frozen=True)
class NoteSpec:
    pitch: int
    start_beat: float
    duration_beats: float
    velocity: int = 100

    @classmethod
    def of(cls, n: Note) -> NoteSpec:
        return cls(n.pitch, n.start_beat, n.duration_beats, n.velocity)

    def to_note(self) -> Note:
        return Note(
            pitch=self.pitch,
            start_beat=self.start_beat,
            duration_beats=self.duration_beats,
            velocity=self.velocity,
        )


@dataclass(frozen=True)
class SetTransport:
    tempo_bpm: float | None = None
    time_signature: tuple[int, int] | None = None


@dataclass(frozen=True)
class AddTrack:
    id: str
    name: str
    kind: Literal["midi", "audio", "return"]
    index: int


@dataclass(frozen=True)
class RemoveTrack:
    id: str


@dataclass(frozen=True)
class UpdateTrack:
    id: str
    name: str | None = None
    volume_db: float | None = None
    pan: float | None = None
    mute: bool | None = None
    solo: bool | None = None


@dataclass(frozen=True)
class SetSend:
    track: str
    return_track: str
    level_db: float | None  # None: send off


@dataclass(frozen=True)
class AddClip:
    id: str
    track: str
    name: str
    start_beat: float
    length_beats: float


@dataclass(frozen=True)
class RemoveClip:
    id: str
    track: str


@dataclass(frozen=True)
class UpdateClip:
    id: str
    track: str
    name: str | None = None
    start_beat: float | None = None
    length_beats: float | None = None


@dataclass(frozen=True)
class SetNotes:
    """``added``/``removed`` are the edit; ``final`` the clip's notes afterwards, so a driver may
    replace everything (when the clip holds no DAW-only note data) or edit note by note."""

    clip: str
    track: str
    added: tuple[NoteSpec, ...] = ()
    removed: tuple[NoteSpec, ...] = ()
    final: tuple[NoteSpec, ...] = ()


@dataclass(frozen=True)
class AddDevice:
    id: str
    track: str
    device: str  # catalog id
    name: str
    role: Literal["instrument", "audio_fx"]
    position: int
    params: dict[str, float] = field(default_factory=dict)
    preset: str | None = None  # library id: the DAW should load that named preset


@dataclass(frozen=True)
class AddForeignDevice:
    """A library rack or plugin: loaded by name, read-only afterwards (no parameter edits)."""

    id: str
    track: str
    item: str  # library id
    name: str
    role: Literal["instrument", "audio_fx"]


@dataclass(frozen=True)
class RemoveDevice:
    id: str
    track: str


@dataclass(frozen=True)
class SetParam:
    device: str
    track: str
    param: str
    value: float  # normalized 0..1


Change = (
    SetTransport
    | AddTrack
    | RemoveTrack
    | UpdateTrack
    | SetSend
    | AddClip
    | RemoveClip
    | UpdateClip
    | SetNotes
    | AddDevice
    | AddForeignDevice
    | RemoveDevice
    | SetParam
)


def _notes(c: Clip) -> list[NoteSpec]:
    return [NoteSpec.of(n) for n in c.notes]


def diff(before: Session, after: Session) -> list[Change]:
    """The edits that turn ``before`` into ``after``. Foreign objects are read-only and ignored."""
    out: list[Change] = []
    if (before.tempo_bpm, tuple(before.time_signature)) != (
        after.tempo_bpm,
        tuple(after.time_signature),
    ):
        out.append(
            SetTransport(
                tempo_bpm=after.tempo_bpm if after.tempo_bpm != before.tempo_bpm else None,
                time_signature=tuple(after.time_signature)  # type: ignore[arg-type]
                if tuple(after.time_signature) != tuple(before.time_signature)
                else None,
            )
        )
    bt = {t.id: t for t in before.tracks}
    at = {t.id: t for t in after.tracks}
    removed = [t for t in before.tracks if t.id not in at]
    added = [(i, t) for i, t in enumerate(after.tracks) if t.id not in bt]
    for t in removed:
        out.append(RemoveTrack(t.id))
    for i, t in added:
        out.append(AddTrack(t.id, t.name, t.kind, i))  # type: ignore[arg-type]
    for tid, new in at.items():
        old = bt.get(tid)
        if old is None:
            old = Track(id=tid, name=new.name, kind=new.kind)  # fresh: only the delta below
            old.name = new.name
        _diff_track(out, old, new, fresh=tid not in bt, return_ids=set(at))
    return out


def _diff_track(
    out: list[Change], old: Track, new: Track, fresh: bool, return_ids: set[str]
) -> None:
    upd = {
        k: getattr(new, k)
        for k in ("name", "volume_db", "pan", "mute", "solo")
        if getattr(new, k) != getattr(old, k)
    }
    if upd:
        out.append(UpdateTrack(new.id, **upd))
    for rid in sorted(set(old.sends) | set(new.sends)):
        if rid not in return_ids:
            continue
        if old.sends.get(rid) != new.sends.get(rid):
            out.append(SetSend(new.id, rid, new.sends.get(rid)))
    oc = {c.id: c for c in old.clips}
    nc = {c.id: c for c in new.clips}
    for cid in oc:
        if cid not in nc:
            out.append(RemoveClip(cid, new.id))
    for cid, c in nc.items():
        if cid not in oc:
            out.append(AddClip(cid, new.id, c.name, c.start_beat, c.length_beats))
            before_notes: list[NoteSpec] = []
        else:
            o = oc[cid]
            upd_c = {
                k: getattr(c, k)
                for k in ("name", "start_beat", "length_beats")
                if getattr(c, k) != getattr(o, k)
            }
            if upd_c:
                out.append(UpdateClip(cid, new.id, **upd_c))
            before_notes = _notes(o)
        after_notes = _notes(c)
        if before_notes != after_notes:
            bc, ac = Counter(before_notes), Counter(after_notes)
            out.append(
                SetNotes(
                    cid,
                    new.id,
                    added=tuple((ac - bc).elements()),
                    removed=tuple((bc - ac).elements()),
                    final=tuple(after_notes),
                )
            )
    od = {d.id: d for d in old.devices}
    nd = {d.id: d for d in new.devices}
    for did in od:
        if did not in nd:
            out.append(RemoveDevice(did, new.id))
    for pos, d in enumerate(new.devices):
        if d.id not in od:
            out.append(
                AddDevice(d.id, new.id, d.device, d.name, d.role, pos, dict(d.params), d.preset)
            )
        else:
            for k, v in d.params.items():
                if od[d.id].params.get(k) != v:
                    out.append(SetParam(d.id, new.id, k, v))

    old_foreign = {d.id for d in old.foreign_devices}
    for fd in new.foreign_devices:
        if fd.id not in old_foreign and fd.item and fd.role:
            out.append(AddForeignDevice(fd.id, new.id, fd.item, fd.name, fd.role))


def _track(s: Session, tid: str) -> Track:
    for t in s.tracks:
        if t.id == tid:
            return t
    raise KeyError(f"no track {tid!r}")


def _bump(s: Session, prefix: str, ident: str) -> None:
    try:
        s.counters[prefix] = max(s.counters.get(prefix, 0), int(ident.rsplit("_", 1)[1]))
    except (IndexError, ValueError):  # native keys are opaque
        pass


def apply_changes(session: Session, changes: list[Change]) -> Session:
    """Reference semantics of ``Change``: returns a new session, ``session`` is not modified."""
    s = Session.restore(session.snapshot())
    for ch in changes:
        match ch:
            case SetTransport():
                if ch.tempo_bpm is not None:
                    s.tempo_bpm = ch.tempo_bpm
                if ch.time_signature is not None:
                    s.time_signature = ch.time_signature
            case AddTrack():
                _bump(s, "trk", ch.id)
                s.tracks.insert(ch.index, Track(id=ch.id, name=ch.name, kind=ch.kind))
            case RemoveTrack():
                s.tracks.remove(_track(s, ch.id))
                for o in s.tracks:
                    o.sends.pop(ch.id, None)
            case UpdateTrack():
                t = _track(s, ch.id)
                for k in ("name", "volume_db", "pan", "mute", "solo"):
                    if (v := getattr(ch, k)) is not None:
                        setattr(t, k, v)
            case SetSend():
                t = _track(s, ch.track)
                if ch.level_db is None:
                    t.sends.pop(ch.return_track, None)
                else:
                    t.sends[ch.return_track] = ch.level_db
            case AddClip():
                _bump(s, "clp", ch.id)
                t = _track(s, ch.track)
                t.clips.append(
                    Clip(
                        id=ch.id,
                        name=ch.name,
                        start_beat=ch.start_beat,
                        length_beats=ch.length_beats,
                    )
                )
                t.clips.sort(key=lambda c: c.start_beat)
            case RemoveClip():
                t = _track(s, ch.track)
                t.clips = [c for c in t.clips if c.id != ch.id]
            case UpdateClip():
                t = _track(s, ch.track)
                c = next(c for c in t.clips if c.id == ch.id)
                for k in ("name", "start_beat", "length_beats"):
                    if (v := getattr(ch, k)) is not None:
                        setattr(c, k, v)
                t.clips.sort(key=lambda c: c.start_beat)
            case SetNotes():
                c = next(c for c in _track(s, ch.track).clips if c.id == ch.clip)
                c.notes = [n.to_note() for n in ch.final]
            case AddDevice():
                _bump(s, "dev", ch.id)
                _track(s, ch.track).devices.insert(
                    ch.position,
                    Device(
                        id=ch.id,
                        device=ch.device,
                        name=ch.name,
                        role=ch.role,
                        params=dict(ch.params),
                        preset=ch.preset,
                    ),
                )
            case AddForeignDevice():
                _bump(s, "fdv", ch.id)
                lib = find_item(ch.item)
                _track(s, ch.track).foreign_devices.append(
                    ForeignDevice(
                        id=ch.id,
                        name=ch.name,
                        params=[
                            ForeignParam(name=n, display=d) for n, d in (lib.params if lib else ())
                        ],
                        item=ch.item,
                        role=ch.role,
                    )
                )
            case RemoveDevice():
                t = _track(s, ch.track)
                t.devices = [d for d in t.devices if d.id != ch.id]
            case SetParam():
                d = next(d for d in _track(s, ch.track).devices if d.id == ch.device)
                d.params[ch.param] = ch.value
    return s


def same_session(a: Session, b: Session, ignore_counters: bool = True) -> bool:
    x, y = a.snapshot(), b.snapshot()
    if ignore_counters:
        x.pop("counters"), y.pop("counters")
    return x == y
