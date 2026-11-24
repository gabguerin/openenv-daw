"""Gradio-free DAW Studio logic, shared by the Gradio UI (``studio.py``) and the browser build.

``BrowserStudio`` is the Python half of the static Hugging Face Space: it runs inside Pyodide,
keeps one episode, and returns JSON strings (HTML views, a message and the episode state) that
``static/app.js`` renders. It imports neither ``gradio``, ``fastmcp`` nor ``httpx``, so the same
file runs under pytest and in the browser. Manual tool calls and oracle stepping only: model
stepping needs a model endpoint and lives in the Gradio UI.
"""

from __future__ import annotations

import json
import random
from types import SimpleNamespace
from typing import Any

from openenv_daw.server.episode import MAX_CONSECUTIVE_ERRORS, MAX_STEPS, Episode
from openenv_daw.server.rewards.score import score_episode
from openenv_daw.server.sim.session import Session as SessionModel
from openenv_daw.server.tasks.generators import families
from openenv_daw.server.tasks.store import TaskStore, default_store

from . import agent, render

VIEW_KEYS = ("task", "arrangement", "reward", "checks", "log")


def skeleton(schema: dict[str, Any]) -> str:
    """A JSON args template for a tool: its required parameters with placeholder values."""
    zero = {"string": "", "integer": 0, "number": 0, "boolean": False, "array": [], "object": {}}
    props = schema.get("properties", {})
    out: dict[str, Any] = {}
    for k in schema.get("required", []):
        p = props.get(k, {})
        if p.get("enum"):
            out[k] = p["enum"][0]
        elif "default" in p:
            out[k] = p["default"]
        else:
            out[k] = zero.get(p.get("type", "string"))
    return json.dumps(out, indent=2)


def reset_kwargs(
    split: str, family: str | None, number: Any, randomize: bool = False
) -> dict[str, Any]:
    """Arguments for ``env.reset``. The number is a seed on both splits (heldout wraps within
    the family-filtered pool); a random task draws its own seed so it can be reproduced."""
    kw: dict[str, Any] = {"split": split}
    if family and family != "(any)":
        kw["family"] = family
    kw["seed"] = random.randrange(2**31) if randomize else max(int(number or 0), 0)
    return kw


def sync_snapshots(ep: Episode | None, snapshots: list[dict[str, Any]]) -> None:
    """Keep ``snapshots`` (in place) aligned with the episode's steps: [0] is the initial
    project, [k] the project after step k."""
    if ep is None:
        snapshots.clear()
        return
    if not snapshots:
        snapshots.append(ep.initial.snapshot())
    while len(snapshots) < ep.step_count + 1:
        snapshots.append(ep.backend.session.snapshot())


def empty_views() -> tuple[str, ...]:
    """task, project, reward, checks, episode before the first reset."""
    empty = SessionModel()
    return (
        render.task_html(None),
        render.arrangement_html(empty, empty),
        render.reward_placeholder_html(),
        "",
        render.episode_html([], None, "", False),
    )


def episode_views(
    ep: Episode, snapshots: list[dict[str, Any]], view_step: int | None
) -> tuple[str, ...]:
    """task, project, reward, checks, episode for the episode, at ``view_step`` if set."""
    sync_snapshots(ep, snapshots)
    latest = len(snapshots) - 1
    vs = view_step if view_step is not None and view_step < latest else None
    k = latest if vs is None else vs
    cur = SessionModel.restore(snapshots[k])
    prev = SessionModel.restore(snapshots[k - 1]) if k > 0 else None
    final = ep.score.as_dict() if ep.score else None
    shown_final = final if vs is None else None
    damage = [*final["damage"], *final["extraneous"]] if vs is None and final else None
    if damage is None:
        sc = score_episode(ep.task, ep.initial, cur, None, ep.error_count, False)
        damage = [*sc.damage, *sc.extraneous]
    subtitle = f"step {ep.step_count}/{ep.max_steps} · {'done' if ep.done else 'running'}"
    if vs is not None:
        subtitle += f" · viewing step {vs}"
    return (
        render.task_html(ep.task),
        render.arrangement_html(cur, ep.initial, prev, None, damage),
        render.reward_compact_html(ep.task, ep.initial, cur, ep.error_count, shown_final),
        render.dark_checks_html(ep.task, ep.initial, cur, ep.error_count, shown_final, prev),
        render.episode_html(ep.step_log, k if k > 0 else None, subtitle, True),
    )


def reply_failure(reply: agent.ModelReply) -> str | None:
    """Why a model reply cannot be executed (an error, or no tool call), else ``None``."""
    if reply.error:
        return reply.error
    if reply.tool is None:
        if reply.text:
            return f"Model returned no tool call:\n\n{reply.text}"
        return "Model returned no tool call."
    return None


def format_model_reply(reply: agent.ModelReply, last: dict[str, Any]) -> str:
    """Markdown for an executed model step: its text, the call and the tool result."""
    parts: list[str] = []
    if reply.text:
        parts.append(reply.text)
    parts.append(f"`{reply.tool}`\n```json\n{json.dumps(reply.args, indent=2)}\n```")
    parts.append("```json\n" + json.dumps(last["result"], indent=1)[:4000] + "\n```")
    return "\n\n".join(parts)


class BrowserStudio:
    """One visitor's DAW Studio in the browser. Every public method returns a JSON string."""

    def __init__(self, tools: list[dict[str, Any]], store: TaskStore | None = None) -> None:
        self.store = store or default_store()
        self.tools = [
            SimpleNamespace(
                name=t["name"],
                description=t.get("description") or "",
                input_schema=t.get("input_schema") or {},
            )
            for t in tools
        ]
        self.episode: Episode | None = None
        self.oracle_i = 0
        self.snapshots: list[dict[str, Any]] = []
        self.view_step: int | None = None

    # --- plumbing ----------------------------------------------------------------------------
    def _state(self) -> dict[str, Any]:
        ep = self.episode
        return {
            "started": ep is not None,
            "done": bool(ep and ep.done),
            "step_count": ep.step_count if ep else 0,
            "max_steps": ep.max_steps if ep else MAX_STEPS,
        }

    def _out(self, message: str = "", ok: bool = True, **extra: Any) -> str:
        views = (
            episode_views(self.episode, self.snapshots, self.view_step)
            if self.episode is not None
            else empty_views()
        )
        return json.dumps(
            {
                "views": dict(zip(VIEW_KEYS, views, strict=True)),
                "message": message,
                "ok": ok,
                "state": self._state(),
                **extra,
            }
        )

    def _run(self, tool: str, args: dict[str, Any]) -> None:
        """Execute one tool call in the episode (the direct, non-MCP path), recording it."""
        ep = self.episode
        assert ep is not None
        if tool == "submit":
            data = ep.on_submit(str(args.get("status", "done")), str(args.get("reason", "")))
        else:
            try:
                data = ep.backend.call(tool, args)
            except Exception as e:  # a tool failing on odd arguments is a tool error, not a crash
                data = {"ok": False, "error": {"code": "invalid_args", "message": str(e)}}
        ep.record(tool, args, data)
        ep.after_step()
        self.view_step = None

    def _open_episode(self) -> tuple[Episode | None, str | None]:
        ep = self.episode
        if ep is None:
            return None, "Reset first."
        if ep.done:
            return None, "The episode is over; reset."
        return ep, None

    # --- public API --------------------------------------------------------------------------
    def meta(self) -> str:
        return json.dumps(
            {
                "families": ["(any)", *families()],
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "skeleton": skeleton(t.input_schema),
                    }
                    for t in self.tools
                ],
                "css": render.STUDIO_CSS,
            }
        )

    def init(self) -> str:
        return self._out()

    def reset(
        self, split: str = "train", family: str = "(any)", number: Any = 0, randomize: bool = False
    ) -> str:
        kw = reset_kwargs(split, family, number, randomize)
        task = self.store.sample(kw["split"], kw["seed"], kw.get("family"))
        self.episode = Episode(task, MAX_STEPS, MAX_CONSECUTIVE_ERRORS)
        self.oracle_i = 0
        self.snapshots, self.view_step = [], None
        return self._out(seed=kw["seed"])

    def manual(self, tool: str, raw: str) -> str:
        ep, why = self._open_episode()
        if ep is None:
            return self._out(why or "", ok=False)
        try:
            args = json.loads(raw or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as e:
            return self._out(f"Invalid JSON: {e}", ok=False)
        self._run(tool, args)
        last = ep.step_log[-1]
        return self._out("```json\n" + json.dumps(last["result"], indent=1)[:4000] + "\n```")

    def oracle_step(self) -> str:
        ep = self.episode
        if ep is None or ep.done or self.oracle_i >= len(ep.task.oracle):
            return self._out("Nothing left to replay.", ok=False)
        st = ep.task.oracle[self.oracle_i]
        self.oracle_i += 1
        self._run(st.tool, st.args)
        return self._out(f"oracle: `{st.tool}`")

    def select_step(self, step: Any) -> str:
        self.view_step = int(step)
        return self._out()
