"""Pure HTML renderers for the web UI (no Gradio import). All user-controlled text is escaped."""

from __future__ import annotations

import html
import math
import re
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


# --- DAW Studio view ------------------------------------------------------------------------
# Built on the OpenEnv Playground look (light theme, oe-* classes); colours come from Gradio's
# theme variables so the arrangement reads in light and dark.

STUDIO_CSS = """
<style>
.ds-mut{color:var(--body-text-color-subdued)}
.ds-ok{color:#16a34a}.ds-bad{color:#dc2626}
.dark .ds-ok{color:#4ade80}.dark .ds-bad{color:#f87171}
.ds-card{border:1px solid var(--border-color-primary);border-radius:12px;padding:14px 18px;
 background:var(--background-fill-primary)}
.ds-card h4{margin:0 0 8px;font-size:15px;font-weight:600}
.ds-card ul,.ds-card ol{margin:0;padding:0}.ds-card li{list-style:none;margin:3px 0;font-size:13px}
.ds-card code{font-family:var(--font-mono);background:none;padding:0}
.ds-arr{display:grid;grid-template-columns:minmax(110px,200px) 1fr;border:1px solid var(--border-color-primary);
 border-radius:10px;overflow:hidden;grid-auto-rows:min-content;font-size:13px}
.ds-ruler{display:grid;grid-column:2;height:24px;border-bottom:1px solid var(--border-color-primary)}
.ds-corner{height:24px;border-bottom:1px solid var(--border-color-primary);border-right:1px solid var(--border-color-primary)}
.ds-ruler span{border-left:1px solid var(--border-color-primary);padding-left:4px;font-size:.8em;color:var(--body-text-color-subdued)}
.ds-head{border-right:1px solid var(--border-color-primary);border-bottom:1px solid var(--border-color-primary);
 padding:8px 10px;border-left:4px solid var(--c);min-height:110px;display:flex;flex-direction:column;gap:4px}
.ds-lane{position:relative;border-bottom:1px solid var(--border-color-primary);min-height:110px;
 background-image:repeating-linear-gradient(90deg,var(--border-color-primary) 0,var(--border-color-primary) 1px,transparent 1px,transparent var(--bar))}
.ds-foreign{background-color:var(--background-fill-secondary)}
.ds-dmg{outline:2px solid #ef4444;outline-offset:-2px}
.ds-touched{box-shadow:inset 0 0 0 2px var(--color-accent)}
.ds-name{font-weight:600;display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.ds-badge{font-size:.7em;border-radius:4px;padding:0 6px;color:#fff;background:var(--c)}
.ds-new,.ds-chg{font-size:.65em;border-radius:4px;padding:0 5px}
.ds-new{background:#16a34a;color:#fff}.ds-chg{background:#ca8a04;color:#fff}
.ds-meter{height:5px;background:var(--background-fill-secondary);border-radius:3px;overflow:hidden;width:100%}
.ds-meter i{display:block;height:100%;background:var(--c)}
.ds-chip{border:1px solid var(--border-color-primary);border-radius:10px;padding:0 6px;font-size:.75em;margin-right:3px}
.ds-clip{position:absolute;top:8px;bottom:8px;border:1px solid var(--c);
 background:color-mix(in srgb,var(--c) 22%,transparent);border-radius:4px;overflow:hidden}
.ds-clip span{position:absolute;left:3px;top:0;font-size:.7em;z-index:1;background:var(--background-fill-primary);
 border-radius:3px;padding:0 4px;opacity:.9}
.ds-clip.f{border-style:dashed}
.ds-ms{display:inline-flex;gap:3px;margin-right:4px}
.ds-ms b,.ds-ms span{border-radius:4px;padding:0 5px;font-size:.75em;background:var(--background-fill-secondary)}
.ds-ms b{background:#dc2626;color:#fff}
.ds-gauge{display:flex;align-items:baseline;gap:18px}
.ds-gauge big{font-size:2em;font-weight:700;line-height:1}
.ds-ep ol{margin:0;padding:0}
.ds-ep li{list-style:none;display:flex;gap:12px;padding:7px 4px;border-top:1px solid var(--border-color-primary);cursor:pointer;border-radius:4px}
.ds-ep li:hover{background:var(--background-fill-secondary)}
.ds-ep li.ds-cur{background:var(--color-accent-soft)}
.ds-ep li>span{flex:none;width:20px;font-family:var(--font-mono);font-size:12px;color:var(--body-text-color-subdued)}
.ds-ep code{display:block;font-size:13px;word-break:break-word}
.ds-ep small{display:block;font-size:12px;color:var(--body-text-color-subdued);word-break:break-word}
.ds-ep small.ds-bad{color:#dc2626}
.ds-delta{font-size:.7em;border-radius:4px;padding:0 5px;margin-left:4px;color:#fff}
.ds-delta.up{background:#16a34a}.ds-delta.down{background:#dc2626}
.ds-task p{margin:0 0 8px;font-size:15px}
.ds-py pre{margin:0;padding:10px 12px;border-radius:8px;background:var(--background-fill-secondary);
 border:1px solid var(--border-color-primary);font-family:var(--font-mono);font-size:12px;white-space:pre-wrap;word-break:break-word}
</style>
"""

KIND_COLOR = {"midi": "#818cf8", "audio": "#f59e0b", "return": "#2dd4bf", "group": "#a8a29e"}


def _dump(t: Track) -> dict[str, Any]:
    return t.model_dump(mode="json")


def track_diff(initial: Session, prev: Session | None, current: Session) -> dict[str, list[str]]:
    """Per track id: ``new`` (not in the initial project), ``changed`` (differs from it) and
    ``touched`` (differs from the previous step)."""
    init = {t.id: _dump(t) for t in initial.tracks}
    before = {t.id: _dump(t) for t in prev.tracks} if prev else {}
    out: dict[str, list[str]] = {}
    for t in current.tracks:
        d = _dump(t)
        flags = []
        if t.id not in init:
            flags.append("new")
        elif init[t.id] != d:
            flags.append("changed")
        if prev is not None and before.get(t.id) != d:
            flags.append("touched")
        out[t.id] = flags
    return out


def _bars_total(s: Session) -> tuple[float, float]:
    per_bar = s.time_signature[0] * 4 / s.time_signature[1]
    end = max(
        [c.start_beat + c.length_beats for t in s.tracks for c in t.clips]
        + [fc.start_beat + fc.length_beats for t in s.tracks for fc in t.foreign_clips]
        + [per_bar]
    )
    bars = max(math.ceil(end / per_bar), 4)
    return bars * per_bar, per_bar


def _db_pct(v: float) -> float:
    return max(0.0, min(100.0, (v + 70) / 76 * 100))


def arrangement_html(
    s: Session,
    initial: Session,
    prev: Session | None = None,
    selected: str | None = None,
    damage: list[str] | None = None,
) -> str:
    total, per_bar = _bars_total(s)
    bars = round(total / per_bar)
    diff = track_diff(initial, prev, s)
    names = {t.id: t.name for t in s.tracks}
    dmg = damage or []
    ruler = (
        f'<div class="ds-corner"></div><div class="ds-ruler" style="grid-template-columns:'
        f'repeat({bars},1fr)">' + "".join(f"<span>{i + 1}</span>" for i in range(bars)) + "</div>"
    )
    rows = []
    for t in s.tracks:
        color = KIND_COLOR.get(t.kind, "#999")
        flags = diff.get(t.id, [])
        foreign = t.kind == "group"
        name_re = re.compile(rf"(?<!\w){re.escape(t.name)}(?!\w)")
        hit = any(name_re.search(d) for d in dmg)
        sel = " ds-sel" if t.id == selected else ""
        badges = ("<span class=ds-new>NEW</span>" if "new" in flags else "") + (
            "<span class=ds-chg>CHANGED</span>" if "changed" in flags else ""
        )
        lock = " 🔒" if foreign else ""
        mix = (
            '<span class="ds-ms">'
            + "".join(
                f"<b>{x}</b>" if on else f"<span class=ds-mut>{x}</span>"
                for x, on in (("M", t.mute), ("S", t.solo))
            )
            + "</span>"
        )
        chips = "".join(
            f'<span class="ds-chip">{_e(d.device)}</span>' for d in t.devices
        ) + "".join(f'<span class="ds-chip">{_e(d.name)} 🔒</span>' for d in t.foreign_devices)
        sends = "".join(
            f'<span class="ds-chip">→ {_e(names.get(r, r))} {_db(v)}</span>'
            for r, v in t.sends.items()
        )
        head_cls = (
            "ds-head"
            + sel
            + (" ds-dmg" if hit else "")
            + (" ds-touched" if "touched" in flags else "")
        )
        lane_cls = "ds-lane" + sel + (" ds-foreign" if foreign else "") + (" ds-dmg" if hit else "")
        clips = "".join(
            f'<div class="ds-clip" style="left:{c.start_beat / total * 100:.3f}%;'
            f'width:{c.length_beats / total * 100:.3f}%" title="{_e(c.name)}">'
            f"<span>{_e(c.name)}</span>{piano_roll(c)}</div>"
            for c in t.clips
        ) + "".join(
            f'<div class="ds-clip f" style="left:{fc.start_beat / total * 100:.3f}%;'
            f'width:{max(fc.length_beats, 1) / total * 100:.3f}%">'
            f"<span>{_e(fc.name)} 🔒 {_e(fc.kind)}</span></div>"
            for fc in t.foreign_clips
        )
        tid = _e(t.id)
        rows.append(
            f'<div class="{head_cls}" data-track="{tid}" style="--c:{color}">'
            f'<div class="ds-name"><span class="ds-badge">{_e(t.kind)}</span>{_e(t.name)}{lock}'
            f"{badges}</div><div>{mix} <span class=ds-mut>{_db(t.volume_db)} · pan {t.pan:+.2f}</span></div>"
            f'<div class="ds-meter"><i style="width:{_db_pct(t.volume_db):.0f}%"></i></div>'
            f"<div>{chips}{sends}</div></div>"
            f'<div class="{lane_cls}" data-track="{tid}" style="--c:{color};--bar:{100 / bars:.4f}%">'
            f"{clips}</div>"
        )
    body = (
        "".join(rows)
        or '<div class="ds-head" style="--c:#555">Empty project</div><div class="ds-lane"></div>'
    )
    return f'<div class="ds-arr">{ruler}{body}</div>'


def task_html(task: Task | None, profile: str) -> str:
    if task is None:
        return (
            f'<div class="ds-task"><span class="ds-mut">No task yet. Press Reset '
            f"(profile {_e(profile)}).</span></div>"
        )
    return f'<div class="ds-task"><p>{_e(task.intent)}</p></div>'


def reward_compact_html(
    task: Task,
    initial: Session,
    current: Session,
    errors: int,
    final: dict[str, Any] | None,
) -> str:
    results = [run_check(c.kind, current, initial, c.args) for c in task.checks]
    passed = f"{sum(results)}/{len(results)}"
    if final:
        label = "final reward" + ("" if final["success"] else " · not a success")
        value = f"{final['reward']:.3f}"
    else:
        label = "if you submit done"
        value = f"{project(task, initial, current, errors)['done']:.3f}"
    return (
        f'<div class="ds-card"><h4>Reward</h4><div class="ds-gauge"><div><big>{value}</big>'
        f'<div class="ds-mut">{label}</div></div><div><big>{passed}</big>'
        f'<div class="ds-mut">checks passed</div></div></div></div>'
    )


def reward_placeholder_html() -> str:
    return (
        '<div class="ds-card"><h4>Reward</h4>'
        '<span class="ds-mut">Appears after you reset.</span></div>'
    )


def episode_html(
    step_log: list[dict[str, Any]], current: int | None, subtitle: str, started: bool
) -> str:
    if not started:
        body = '<p class="ds-mut">Nothing yet. Reset, then call a tool.</p>'
    else:
        rows = [
            '<li data-step="0"><span>R</span><div><code>reset()</code><small>new episode</small></div></li>'
        ]
        for e in step_log:
            err = e.get("error")
            args = _e(", ".join(f"{k}={v}" for k, v in e["arguments"].items())[:90])
            if err:
                note = f'<small class="ds-bad">{_e(str(err.get("code")))}: {_e(str(err.get("message"))[:100])}</small>'
            else:
                note = f"<small>{_e(str(e.get('result'))[:90])}</small>"
            cur = ' class="ds-cur"' if current == e["step"] else ""
            rows.append(
                f'<li data-step="{e["step"]}"{cur}><span>{e["step"]}</span><div>'
                f"<code>{_e(e['tool'])}({args})</code>{note}</div></li>"
            )
        body = f"<ol>{''.join(rows)}</ol>"
    sub = f'<div class="ds-mut" style="margin-bottom:6px">{_e(subtitle)}</div>' if subtitle else ""
    return f'<div class="ds-card ds-ep"><h4>Episode</h4>{sub}{body}</div>'


def python_html(reset_kw: dict[str, Any] | None, last: dict[str, Any] | None) -> str:
    kw = ", ".join(f"{k}={v!r}" for k, v in (reset_kw or {}).items())
    lines = [f"r = await env.reset({kw})"]
    if last:
        args = ", ".join(f"{k}={v!r}" for k, v in last["arguments"].items())
        lines.append(f'r = await env.call("{last["tool"]}"{", " + args if args else ""})')
    else:
        lines.append('r = await env.call("get_session")')
    code = "from openenv_daw.env.client import DawEnv\n\n" + "\n".join(lines)
    return f'<div class="ds-card ds-py"><h4>Same call in Python</h4><pre>{_e(code)}</pre></div>'


def dark_checks_html(
    task: Task,
    initial: Session,
    current: Session,
    errors: int = 0,
    final: dict[str, Any] | None = None,
    prev: Session | None = None,
) -> str:
    results = [run_check(c.kind, current, initial, c.args) for c in task.checks]
    before = (
        [run_check(c.kind, prev, initial, c.args) for c in task.checks]
        if prev is not None
        else None
    )

    def delta(i: int, ok: bool) -> str:
        if before is None or before[i] == ok:
            return ""
        return (
            '<span class="ds-delta up">now passing</span>'
            if ok
            else ('<span class="ds-delta down">now failing</span>')
        )

    items = "".join(
        f"<li>{'<span class=ds-ok>✔</span>' if ok else '<span class=ds-bad>✘</span>'} <code>{_e(c.kind)}</code>"
        f"{' <small class=ds-bad>critical</small>' if c.critical else ''} "
        f'<small class="ds-mut">{_e(", ".join(f"{k}={v}" for k, v in c.args.items())[:120])}</small>'
        f"{delta(i, ok)}</li>"
        for i, (c, ok) in enumerate(zip(task.checks, results, strict=True))
    )
    s = final or score_episode(task, initial, current, None, errors, False).as_dict()
    dmg = "".join(f"<li class=ds-bad>{_e(d)}</li>" for d in s["damage"])
    return (
        f'<div class="ds-card"><h4>Checks · {sum(results)}/{len(results)}</h4><ul>{items}</ul></div>'
        + (f'<div class="ds-card"><h4>Damage</h4><ul>{dmg}</ul></div>' if dmg else "")
    )


def answer_key_html(task: Task | None) -> str:
    if task is None:
        return ""
    steps = "".join(
        f"<li><code>{_e(o.tool)}({_e(', '.join(f'{k}={v}' for k, v in o.args.items())[:100])})</code></li>"
        for o in task.oracle
    )
    return f"<p>Expected status: <b>{_e(task.expected_status)}</b></p><ol>{steps}</ol>"


def env_config_html(env: Any) -> str:
    ep = env.episode
    masked = ", ".join(sorted(ep.masked_params)) if ep and ep.masked_params else "none"
    return (
        f'<div class="ds-card"><h4>Environment</h4>profile <b>{_e(env.profile)}</b> · max steps '
        f"<b>{env.max_steps}</b> · error limit <b>{env.max_consecutive_errors}</b> · param mask "
        f'<b>{env.param_mask_prob:g}</b><br><span class="ds-mut">masked: {_e(masked)}</span></div>'
    )
