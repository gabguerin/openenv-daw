"""The "DAW Studio" Gradio tab: one private environment per visitor, live checks, manual tool
calls, oracle stepping and model stepping. Built through ``create_app(gradio_builder=...)``."""

from __future__ import annotations

import json
import random
from collections.abc import Awaitable, Callable
from typing import Any

import gradio as gr
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from openenv_daw.server.daw_environment import DawEnvironment
from openenv_daw.server.rewards.score import score_episode
from openenv_daw.server.sim.session import Session as SessionModel
from openenv_daw.server.tasks.generators import families

from . import agent, render


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


async def _call(sess: Session, tool: str, args: dict[str, Any]) -> None:
    # step_async: the sync step returns an un-awaited rubric coroutine inside a running loop
    await sess.env.step_async(CallToolAction(tool_name=tool, arguments=args))
    sess.view_step = None


def _format_model_reply(reply: agent.ModelReply, last: dict[str, Any]) -> str:
    parts: list[str] = []
    if reply.text:
        parts.append(reply.text)
    parts.append(f"`{reply.tool}`\n```json\n{json.dumps(reply.args, indent=2)}\n```")
    parts.append("```json\n" + json.dumps(last["result"], indent=1)[:4000] + "\n```")
    return "\n\n".join(parts)


async def step_with_model(
    sess: Session,
    cfg: agent.ModelConfig,
    oa_tools: list[dict[str, Any]],
    *,
    caller: Callable[
        [agent.ModelConfig, list[dict[str, Any]], list[dict[str, Any]]],
        Awaitable[agent.ModelReply],
    ] = agent.next_call,
) -> tuple[bool, str]:
    """Ask the configured model for one tool call and execute it in ``sess``."""
    ep = sess.env.episode
    if ep is None:
        return False, "Reset first."
    if ep.done:
        return False, "The episode is over; reset."
    messages = agent.build_messages(ep.task.intent, ep.step_log)
    reply = await caller(cfg, messages, oa_tools)
    if reply.error:
        return False, reply.error
    if reply.tool is None:
        if reply.text:
            return False, f"Model returned no tool call:\n\n{reply.text}"
        return False, "Model returned no tool call."
    await _call(sess, reply.tool, reply.args)
    last = ep.step_log[-1]
    return True, _format_model_reply(reply, last)


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
    oa_tools = agent.openai_tools(tools)
    probe.close()
    fams = ["(any)", *families()]
    on_space = agent.is_space()
    defaults = agent.default_config()

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
        return (sess, *_views(sess), "", "", kw["seed"])

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

    def _resolve_cfg(
        url: str,
        model: str,
        key: str,
        max_tokens: Any,
        oauth_token: gr.OAuthToken | None,
    ) -> tuple[agent.ModelConfig | None, str | None]:
        try:
            mt = max(int(max_tokens), 1) if max_tokens is not None else defaults.max_tokens
        except (TypeError, ValueError):
            mt = defaults.max_tokens
        if on_space:
            if oauth_token is None or not oauth_token.token:
                return None, "Sign in with Hugging Face first."
            return (
                agent.ModelConfig(
                    base_url=agent.HF_ROUTER_URL,
                    model=(model or "").strip() or defaults.model,
                    api_key=oauth_token.token,
                    max_tokens=mt,
                ),
                None,
            )
        return (
            agent.ModelConfig(
                base_url=(url or "").strip() or defaults.base_url,
                model=(model or "").strip() or defaults.model,
                api_key=(key or "").strip(),
                max_tokens=mt,
            ),
            None,
        )

    async def model_step(
        sess: Session | None,
        url: str,
        model: str,
        key: str,
        max_tokens: Any,
        oauth_token: gr.OAuthToken | None = None,
    ):
        sess = need(sess)
        cfg, err = _resolve_cfg(url, model, key, max_tokens, oauth_token)
        if err is not None or cfg is None:
            return (sess, *_views(sess), err or "Missing model configuration.")
        _, msg = await step_with_model(sess, cfg, oa_tools)
        return (sess, *_views(sess), msg)

    async def model_run(
        sess: Session | None,
        url: str,
        model: str,
        key: str,
        max_tokens: Any,
        oauth_token: gr.OAuthToken | None = None,
    ):
        sess = need(sess)
        cfg, err = _resolve_cfg(url, model, key, max_tokens, oauth_token)
        if err is not None or cfg is None:
            yield (sess, *_views(sess), err or "Missing model configuration.")
            return
        ep = sess.env.episode
        if ep is None:
            yield (sess, *_views(sess), "Reset first.")
            return
        if ep.done:
            yield (sess, *_views(sess), "The episode is over; reset.")
            return
        for _ in range(ep.max_steps):
            ep = sess.env.episode
            if ep is None or ep.done:
                break
            ok, msg = await step_with_model(sess, cfg, oa_tools)
            yield (sess, *_views(sess), msg)
            if not ok:
                break

    step_js = (
        "element.addEventListener('click', e => { const r = e.target.closest('[data-step]');"
        " if (r) trigger('click', {step: r.dataset.step}); });"
    )
    header = (
        '<div class="oe-header"><span class="oe-eyebrow">openenv_daw · MCP environment</span>'
        "<h1>DAW</h1><p>A tool-use environment for music production: edit a project with "
        "portable DAW tools — manually, with the oracle, or with a model — and get scored on "
        "the result.</p></div>"
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
                    family = gr.Dropdown(fams, value="(any)", label="Task family")
                    task_card = gr.HTML()
                    with gr.Accordion("Choose a task", open=False, visible=False):
                        with gr.Row():
                            split = gr.Radio(["train", "heldout"], value="train", label="Split")
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
                with gr.Column(elem_classes="oe-card"):
                    gr.HTML(
                        '<div class="oe-step"><h2>3. Run a model</h2>'
                        "<p>Ask a model for the next tool call and run it.</p></div>"
                    )
                    if on_space:
                        gr.LoginButton()
                        with gr.Row():
                            model_name = gr.Textbox(
                                value=defaults.model,
                                label="Model",
                                placeholder=agent.DEFAULT_SPACE_MODEL,
                                info="Default: our SFT model (or any HF Inference Providers model, e.g. Qwen/Qwen3-32B).",
                                scale=3,
                            )
                            model_max_tokens = gr.Number(
                                value=defaults.max_tokens,
                                precision=0,
                                minimum=1,
                                label="Max tokens",
                                scale=1,
                            )
                        model_url: Any = gr.State(agent.HF_ROUTER_URL)
                        model_key: Any = gr.State("")
                    else:
                        with gr.Accordion("Model endpoint", open=False):
                            model_url = gr.Textbox(
                                value=defaults.base_url,
                                label="Base URL",
                                placeholder=agent.DEFAULT_LOCAL_URL,
                            )
                            model_name = gr.Textbox(
                                value=defaults.model,
                                label="Model",
                                placeholder=agent.DEFAULT_LOCAL_MODEL,
                            )
                            model_key = gr.Textbox(
                                value=defaults.api_key,
                                label="API key",
                                type="password",
                                placeholder="optional",
                            )
                            model_max_tokens = gr.Number(
                                value=defaults.max_tokens,
                                precision=0,
                                minimum=1,
                                label="Max tokens",
                            )
                    with gr.Row():
                        model_step_btn = gr.Button("Step model", variant="primary")
                        model_run_btn = gr.Button("Run to end")
                    model_out = gr.Markdown()
            with gr.Column(scale=2, min_width=320):
                with gr.Column(elem_classes=["oe-card", "ds-epsec"]):
                    log = gr.HTML(js_on_load=step_js)
                    reward = gr.HTML()
                    with gr.Accordion("Checks", open=False):
                        checks = gr.HTML()
        views = [task_card, arrangement, reward, checks, log]
        outs = [state, *views, result]
        model_inputs = [state, model_url, model_name, model_key, model_max_tokens]
        model_outs = [state, *views, model_out]
        free: dict[str, Any] = {"concurrency_limit": None}
        demo.load(init, state, [state, *views], **free)
        reset_btn.click(
            reset, [state, split, family, number], [*outs, model_out, number], **free
        )
        random_btn.click(
            random_task, [state, split, family, number], [*outs, model_out, number], **free
        )
        family.input(
            random_task, [state, split, family, number], [*outs, model_out, number], **free
        )
        tool.change(pick_tool, tool, [args, tool_doc], **free)
        run_btn.click(manual, [state, tool, args], outs, **free)
        step_btn.click(oracle_step, state, outs, **free)
        model_step_btn.click(model_step, model_inputs, model_outs, **free)
        model_run_btn.click(model_run, model_inputs, model_outs, **free)
        log.click(select_step, state, [state, *views], **free)
    return demo
