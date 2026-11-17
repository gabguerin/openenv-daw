"""The "DAW Studio" Gradio tab: one private environment per visitor, live checks, manual tool
calls and oracle stepping. Built through ``create_app(gradio_builder=...)``."""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from typing import Any

import gradio as gr
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from openenv_daw.server.daw_environment import DawEnvironment
from openenv_daw.server.rewards.score import score_episode
from openenv_daw.server.sim.session import Session as SessionModel
from openenv_daw.server.tasks.generators import families

from . import render


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


class Session:
    """Per-visitor state held in ``gr.State``."""

    def __init__(self, env: DawEnvironment) -> None:
        self.env = env
        self.oracle_i = 0
        self.snapshots: list[dict[str, Any]] = []  # [0] = initial, [k] = after step k
        self.view_step: int | None = None

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


def sync_snapshots(sess: Session) -> None:
    """Keep ``snapshots`` aligned with the episode's steps (one snapshot per logged step)."""
    ep = sess.env.episode
    if ep is None:
        sess.snapshots = []
        return
    if not sess.snapshots:
        sess.snapshots = [ep.initial.snapshot()]
    while len(sess.snapshots) < ep.step_count + 1:
        sess.snapshots.append(ep.backend.session.snapshot())


def _views(sess: Session | None) -> tuple[Any, ...]:
    """task, project, reward, checks, episode."""
    if sess is None or sess.env.episode is None:
        empty = SessionModel()
        return (
            render.task_html(None),
            render.arrangement_html(empty, empty),
            render.reward_placeholder_html(),
            "",
            render.episode_html([], None, "", False),
        )
    sync_snapshots(sess)
    ep = sess.env.episode
    latest = len(sess.snapshots) - 1
    vs = sess.view_step if sess.view_step is not None and sess.view_step < latest else None
    k = latest if vs is None else vs
    cur = SessionModel.restore(sess.snapshots[k])
    prev = SessionModel.restore(sess.snapshots[k - 1]) if k > 0 else None
    final = ep.score.as_dict() if ep.score else None
    shown_final = final if vs is None else None
    damage = (final or {}).get("damage") if vs is None and final else None
    if damage is None:
        damage = score_episode(ep.task, ep.initial, cur, None, ep.error_count, False).damage
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


async def _call(sess: Session, tool: str, args: dict[str, Any]) -> None:
    # step_async: the sync step returns an un-awaited rubric coroutine inside a running loop
    await sess.env.step_async(CallToolAction(tool_name=tool, arguments=args))
    sess.view_step = None


def build_daw_studio(
    web_manager: Any,
    action_fields: Any,
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: str | None,
    *,
    env_factory: Callable[[], DawEnvironment],
) -> gr.Blocks:
    probe = env_factory()
    tools = probe.step(ListToolsAction()).tools  # type: ignore[attr-defined]
    schemas = {t.name: t.input_schema for t in tools}
    descriptions = {t.name: (t.description or "") for t in tools}
    probe.close()
    fams = ["(any)", *families()]

    def new_session() -> Session:
        return Session(env_factory())

    def need(sess: Session | None) -> Session:
        return sess or new_session()

    def init(sess):
        sess = need(sess)
        return (sess, *_views(sess))

    def reset(sess, split, family, number, randomize=False):
        sess = need(sess)
        kw = reset_kwargs(split, family, number, randomize)
        sess.env.reset(**kw)
        sess.oracle_i = 0
        sess.snapshots, sess.view_step = [], None
        return (sess, *_views(sess), "", kw["seed"])

    def random_task(sess, split, family, number):
        return reset(sess, split, family, number, True)

    async def manual(sess, tool, raw):
        sess = need(sess)
        if sess.env.episode is None:
            return (sess, *_views(sess), "Reset first.")
        if sess.env.episode.done:
            return (sess, *_views(sess), "The episode is over; reset.")
        try:
            args = json.loads(raw or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as e:
            return (sess, *_views(sess), f"Invalid JSON: {e}")
        await _call(sess, tool, args)
        last = sess.env.episode.step_log[-1]
        return (
            sess,
            *_views(sess),
            "```json\n" + json.dumps(last["result"], indent=1)[:4000] + "\n```",
        )

    def pick_tool(tool):
        return skeleton(schemas.get(tool, {})), descriptions.get(tool, "")

    def select_step(sess, evt: gr.EventData):
        sess = need(sess)
        data = getattr(evt, "_data", None) or {}
        if isinstance(data, dict) and str(data.get("step", "")).isdigit():
            sess.view_step = int(data["step"])
        return (sess, *_views(sess))

    async def oracle_step(sess):
        sess = need(sess)
        ep = sess.env.episode
        if ep is None or ep.done or sess.oracle_i >= len(ep.task.oracle):
            return (sess, *_views(sess), "Nothing left to replay.")
        st = ep.task.oracle[sess.oracle_i]
        sess.oracle_i += 1
        await _call(sess, st.tool, st.args)
        return (sess, *_views(sess), f"oracle: `{st.tool}`")

    step_js = (
        "element.addEventListener('click', e => { const r = e.target.closest('[data-step]');"
        " if (r) trigger('click', {step: r.dataset.step}); });"
    )
    header = (
        '<div class="oe-header"><span class="oe-eyebrow">openenv_daw · MCP environment</span>'
        "<h1>DAW</h1><p>A tool-use environment for music production: edit a project with "
        "portable DAW tools and get scored on the result.</p></div>"
    )
    with gr.Blocks(title=title) as demo:
        state = gr.State(None, time_to_live=3600, delete_callback=lambda s: s and s.close())
        gr.HTML(render.STUDIO_CSS)
        gr.HTML(header)
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=360):
                with gr.Column(elem_classes="oe-card"):
                    with gr.Row(elem_classes="oe-row", equal_height=True):
                        gr.HTML(
                            '<div class="oe-step"><h2>1. Start an episode</h2>'
                            "<p><code>reset()</code> returns the task.</p></div>"
                        )
                        reset_btn = gr.Button("Reset", scale=0, min_width=110)
                        random_btn = gr.Button("Random task", scale=0, min_width=130)
                    task_card = gr.HTML()
                    with gr.Accordion("Choose a task", open=False, visible=False):
                        with gr.Row():
                            split = gr.Radio(["train", "heldout"], value="train", label="Split")
                            family = gr.Dropdown(fams, value="(any)", label="Family")
                            number = gr.Number(
                                0,
                                precision=0,
                                minimum=0,
                                label="Seed",
                                info="heldout: wraps within the family's tasks",
                            )
                with gr.Column(elem_classes="oe-card"):
                    gr.HTML(
                        '<div class="oe-step"><h2>2. Call a tool</h2>'
                        "<p><code>step()</code> sends a tool call. Pick one:</p></div>"
                    )
                    arrangement = gr.HTML(render.arrangement_html(SessionModel(), SessionModel()))
                    first = next(iter(schemas))
                    tool = gr.Radio(
                        list(schemas), value=first, show_label=False, elem_classes="oe-tools"
                    )
                    tool_doc = gr.Markdown(descriptions.get(first, ""))
                    args = gr.Code(skeleton(schemas[first]), language="json", label="Arguments")
                    run_btn = gr.Button("Run", variant="primary")
                    step_btn = gr.Button("Step oracle")
                    result = gr.Markdown()
            with gr.Column(scale=2, min_width=320):
                with gr.Column(elem_classes=["oe-card", "ds-epsec"]):
                    log = gr.HTML(js_on_load=step_js)
                    reward = gr.HTML()
                    with gr.Accordion("Checks", open=False):
                        checks = gr.HTML()
        views = [task_card, arrangement, reward, checks, log]
        outs = [state, *views, result]
        free: dict[str, Any] = {"concurrency_limit": None}
        demo.load(init, state, [state, *views], **free)
        reset_btn.click(reset, [state, split, family, number], [*outs, number], **free)
        random_btn.click(random_task, [state, split, family, number], [*outs, number], **free)
        tool.change(pick_tool, tool, [args, tool_doc], **free)
        run_btn.click(manual, [state, tool, args], outs, **free)
        step_btn.click(oracle_step, state, outs, **free)
        log.click(select_step, state, [state, *views], **free)
    return demo
