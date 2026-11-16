"""Pure HTML renderers for the web UI (no Gradio import). All user-controlled text is escaped."""

from __future__ import annotations

import html
from typing import Any

from openenv_daw.rewards.checks import run_check
from openenv_daw.rewards.rubrics import score_episode
from openenv_daw.schema import Clip, Session, Track
from openenv_daw.tasks.model import Task

_e = html.escape

CSS = """
<style>
.dw-track{border:1px solid #8884;border-radius:8px;margin:6px 0;padding:6px 8px}
.dw-track.foreign{opacity:.55;background:repeating-linear-gradient(45deg,#8881,#8881 6px,#0000 6px,#0000 12px)}
.dw-head{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-size:.9em}
.dw-badge{border-radius:4px;padding:0 6px;font-size:.75em;background:#6366f1;color:#fff}
.dw-badge.return{background:#0d9488}.dw-badge.group{background:#78716c}.dw-badge.audio{background:#d97706}
.dw-chip{border:1px solid #8886;border-radius:10px;padding:0 6px;font-size:.75em}
.dw-lane{position:relative;height:44px;margin-top:6px;background:#8881;border-radius:4px;overflow:hidden}
.dw-clip{position:absolute;top:2px;bottom:2px;background:#6366f144;border:1px solid #6366f1;border-radius:3px;overflow:hidden}
.dw-clip span{font-size:.65em;position:absolute;left:3px;top:0}
.dw-ok{color:#16a34a}.dw-bad{color:#dc2626}.dw-mut{opacity:.7}
.dw-checks li{list-style:none;margin:2px 0}
.dw-log li{font-size:.85em;margin:2px 0}
</style>
"""


def _db(v: float) -> str:
    return f"{v:+.1f} dB" if v > -70 else "-inf"


def piano_roll(clip: Clip) -> str:
    if not clip.notes or clip.length_beats <= 0:
        return ""
    lo = min(n.pitch for n in clip.notes)
    hi = max(n.pitch for n in clip.notes)
    span = max(hi - lo, 11) + 1
    rects = "".join(
        f'<rect x="{n.start_beat / clip.length_beats * 100:.3f}" '
        f'y="{(hi - n.pitch) / span * 100:.3f}" '
        f'width="{max(n.duration_beats / clip.length_beats * 100, 0.4):.3f}" '
        f'height="{100 / span:.3f}" fill="#4f46e5"/>'
        for n in clip.notes
    )
    return f'<svg viewBox="0 0 100 100" preserveAspectRatio="none" width="100%" height="100%">{rects}</svg>'


def _track_html(t: Track, s: Session, total: float, names: dict[str, str]) -> str:
    foreign = t.kind == "group"
    flags = "".join(f" {x}" for x, on in (("M", t.mute), ("S", t.solo)) if on)
    chips = "".join(f'<span class="dw-chip">{_e(d.device)}</span>' for d in t.devices)
    chips += "".join(
        f'<span class="dw-chip">{_e(d.name)} (foreign)</span>' for d in t.foreign_devices
    )
    sends = ", ".join(f"{_e(names.get(r, r))} {_db(v)}" for r, v in t.sends.items())
    clips = "".join(
        f'<div class="dw-clip" style="left:{c.start_beat / total * 100:.3f}%;'
        f'width:{c.length_beats / total * 100:.3f}%" title="{_e(c.name)}">'
        f"<span>{_e(c.name)}</span>{piano_roll(c)}</div>"
        for c in t.clips
    )
    clips += "".join(
        f'<div class="dw-clip" style="left:{fc.start_beat / total * 100:.3f}%;'
        f'width:{max(fc.length_beats, 1) / total * 100:.3f}%;border-style:dashed">'
        f"<span>{_e(fc.name)} ({_e(fc.kind)})</span></div>"
        for fc in t.foreign_clips
    )
    return (
        f'<div class="dw-track{" foreign" if foreign else ""}"><div class="dw-head">'
        f'<span class="dw-badge {_e(t.kind)}">{_e(t.kind)}</span><b>{_e(t.name)}</b>'
        f'<span class="dw-mut">{_db(t.volume_db)} · pan {t.pan:+.2f}{_e(flags)}</span>{chips}'
        f"{f'<span class=dw-mut>sends: {sends}</span>' if sends else ''}"
        f"{'<span class=dw-mut>read-only</span>' if foreign else ''}</div>"
        f'<div class="dw-lane">{clips}</div></div>'
    )


def session_html(s: Session) -> str:
    end = max(
        [c.start_beat + c.length_beats for t in s.tracks for c in t.clips]
        + [fc.start_beat + fc.length_beats for t in s.tracks for fc in t.foreign_clips]
        + [4.0]
    )
    names = {t.id: t.name for t in s.tracks}
    head = (
        f'<div class="dw-head"><b>{s.tempo_bpm:g} BPM</b>'
        f"<span>{s.time_signature[0]}/{s.time_signature[1]}</span>"
        f'<span class="dw-mut">{len(s.tracks)} tracks · {end:g} beats</span></div>'
    )
    body = "".join(_track_html(t, s, end, names) for t in s.tracks) or (
        '<p class="dw-mut">Empty project.</p>'
    )
    return f"{CSS}<div>{head}{body}</div>"


def _mark(ok: bool) -> str:
    return '<span class="dw-ok">✔</span>' if ok else '<span class="dw-bad">✘</span>'


def project(task: Task, initial: Session, current: Session, errors: int = 0) -> dict[str, float]:
    """Reward if the episode were submitted now as done / as blocked."""
    return {
        status: score_episode(task, initial, current, status, errors, False).reward
        for status in ("done", "blocked")
    }


def checks_html(
    task: Task,
    initial: Session,
    current: Session,
    errors: int = 0,
    final: dict[str, Any] | None = None,
) -> str:
    """Live per-check status, the damage list and projected rewards. With ``final`` (the
    ``EpisodeScore`` dict of a finished episode) the real outcome is shown instead."""
    results = [run_check(c.kind, current, initial, c.args) for c in task.checks]
    items = "".join(
        f"<li>{_mark(ok)} <code>{_e(c.kind)}</code>"
        f"{' <small>critical</small>' if c.critical else ''} "
        f'<small class="dw-mut">{_e(", ".join(f"{k}={v}" for k, v in c.args.items())[:120])}</small></li>'
        for c, ok in zip(task.checks, results, strict=True)
    )
    s = final or score_episode(task, initial, current, None, errors, False).as_dict()
    dmg = "".join(f"<li>{_e(d)}</li>" for d in s["damage"])
    if final:
        summary = (
            f"<p><b>Episode over</b>: reward <b>{final['reward']:.3f}</b>, "
            f"{'success' if final['success'] else 'not a success'}, submitted "
            f"<code>{_e(str(final['status']))}</code>, expected "
            f"<code>{_e(task.expected_status)}</code>.</p>"
        )
    else:
        p = project(task, initial, current, errors)
        summary = (
            f"<p>If you submit now: <b>done</b> → {p['done']:.3f}, "
            f"<b>blocked</b> → {p['blocked']:.3f}</p>"
        )
    return (
        f'<div><ul class="dw-checks">{items}</ul>{summary}'
        f"{f'<p>Damage:</p><ul>{dmg}</ul>' if dmg else ''}</div>"
    )


def step_log_html(step_log: list[dict[str, Any]]) -> str:
    if not step_log:
        return '<p class="dw-mut">No steps yet.</p>'
    rows = []
    for e in step_log:
        err = e.get("error")
        detail = (
            f" — {_e(str(err.get('code')))}: {_e(str(err.get('message'))[:120])}" if err else ""
        )
        args = _e(", ".join(f"{k}={v}" for k, v in e["arguments"].items())[:100])
        rows.append(
            f"<li>{_mark(e['ok'])} {e['step']}. <code>{_e(e['tool'])}({args})</code>{detail}</li>"
        )
    return f'<ol class="dw-log">{"".join(rows)}</ol>'
