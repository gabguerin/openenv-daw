"""The "DAW Studio" Gradio tab: one private environment per visitor, live checks, manual tool
calls, oracle stepping and model stepping. Built through ``create_app(gradio_builder=...)``."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import gradio as gr
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from openenv_daw.server.daw_environment import DawEnvironment
from openenv_daw.server.sim.session import Session as SessionModel
from openenv_daw.server.tasks.generators import families

from . import agent, render, static_bridge
from .static_bridge import reset_kwargs, skeleton


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
    static_bridge.sync_snapshots(sess.env.episode, sess.snapshots)


def _views(sess: Session | None) -> tuple[Any, ...]:
    """task, project, reward, checks, episode."""
    if sess is None or sess.env.episode is None:
        return static_bridge.empty_views()
    return static_bridge.episode_views(sess.env.episode, sess.snapshots, sess.view_step)


async def _call(sess: Session, tool: str, args: dict[str, Any]) -> None:
    # step_async: the sync step returns an un-awaited rubric coroutine inside a running loop
    await sess.env.step_async(CallToolAction(tool_name=tool, arguments=args))
    sess.view_step = None


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
    failure = static_bridge.reply_failure(reply)
    if failure is not None or reply.tool is None:
        return False, failure or "Model returned no tool call."
    await _call(sess, reply.tool, reply.args)
    return True, static_bridge.format_model_reply(reply, ep.step_log[-1])


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
        reset_btn.click(reset, [state, split, family, number], [*outs, model_out, number], **free)
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
