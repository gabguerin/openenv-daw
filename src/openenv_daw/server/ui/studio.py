"""The "DAW Studio" Gradio tab: one private environment per visitor, live checks, manual tool
calls, oracle replay and an LLM agent. Built through ``create_app(gradio_builder=...)``."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Callable
from typing import Any

import gradio as gr
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from openenv_daw.env.environment import DawEnvironment
from openenv_daw.runner.rollout import InProcessAdapter, iter_episode
from openenv_daw.tasks.generators import families

from . import render
from .quota import QUOTA

HF_ROUTER = "https://router.huggingface.co/v1"
DEFAULT_MODELS = [
    "Qwen/Qwen3-32B",
    "meta-llama/Llama-3.3-70B-Instruct",
    "openai/gpt-oss-120b",
]


def ui_models() -> list[str]:
    raw = os.getenv("OPENENV_DAW_UI_MODELS", "")
    return [m.strip() for m in raw.split(",") if m.strip()] or DEFAULT_MODELS


def skeleton(schema: dict[str, Any]) -> str:
    """A JSON args template for a tool: its required parameters with placeholder values."""
    zero = {"string": "", "integer": 0, "number": 0, "boolean": False, "array": [], "object": {}}
    props = schema.get("properties", {})
    out = {
        k: zero.get(props.get(k, {}).get("type", "string"), None)
        for k in schema.get("required", [])
    }
    return json.dumps(out, indent=2)


class Session:
    """Per-visitor state held in ``gr.State``."""

    def __init__(self, env: DawEnvironment) -> None:
        self.env = env
        self.oracle_i = 0
        self.records: list[dict[str, Any]] = []
        self.running = False

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


def _views(sess: Session | None) -> tuple[str, str, str, str]:
    if sess is None or sess.env.episode is None:
        return "Press **Reset** to start an episode.", "", "", ""
    ep = sess.env.episode
    final = ep.score.as_dict() if ep.score else None
    return (
        f"**Task** `{ep.task.task_id}` · profile `{sess.env.profile}`\n\n> {ep.task.intent}",
        render.session_html(ep.backend.session),
        render.checks_html(ep.task, ep.initial, ep.backend.session, ep.error_count, final),
        render.step_log_html(ep.step_log),
    )


def _call(sess: Session, tool: str, args: dict[str, Any]) -> None:
    sess.env.step(CallToolAction(tool_name=tool, arguments=args))


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
    profile = probe.profile
    probe.close()
    fams = ["(any)", *families()]
    models = ui_models()

    def new_session() -> Session:
        return Session(env_factory())

    def need(sess: Session | None) -> Session:
        return sess or new_session()

    def reset(sess, split, family, number, randomize=False):
        sess = need(sess)
        kw: dict[str, Any] = {"split": split}
        if family and family != "(any)":
            kw["family"] = family
        if not randomize:
            kw["index" if split == "heldout" else "seed"] = int(number or 0)
        sess.env.reset(**kw)
        sess.oracle_i = 0
        return (sess, *_views(sess), "")

    def random_task(sess, split, family, number):
        return reset(sess, split, family, number, True)

    def manual(sess, tool, raw):
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
        _call(sess, tool, args)
        last = sess.env.episode.step_log[-1]
        return (
            sess,
            *_views(sess),
            "```json\n" + json.dumps(last["result"], indent=1)[:4000] + "\n```",
        )

    def pick_tool(tool):
        return skeleton(schemas.get(tool, {}))

    def oracle_step(sess):
        sess = need(sess)
        ep = sess.env.episode
        if ep is None or ep.done or sess.oracle_i >= len(ep.task.oracle):
            return (sess, *_views(sess), "Nothing left to replay.")
        st = ep.task.oracle[sess.oracle_i]
        sess.oracle_i += 1
        _call(sess, st.tool, st.args)
        return (sess, *_views(sess), f"oracle: `{st.tool}`")

    async def oracle_play(sess):
        sess = need(sess)
        ep = sess.env.episode
        while ep is not None and not ep.done and sess.oracle_i < len(ep.task.oracle):
            st = ep.task.oracle[sess.oracle_i]
            sess.oracle_i += 1
            _call(sess, st.tool, st.args)
            yield (sess, *_views(sess), f"oracle: `{st.tool}`")
            await asyncio.sleep(0.4)
        yield (sess, *_views(sess), "Oracle finished.")

    async def run_agent(
        sess, model, max_turns, temperature, oauth_token: gr.OAuthToken | None, request: gr.Request
    ):
        sess = need(sess)
        ep = sess.env.episode
        if ep is None or ep.done:
            yield (sess, *_views(sess), "Reset to a fresh task first.", gr.update())
            return
        if sess.running:
            yield (sess, *_views(sess), "An agent is already running.", gr.update())
            return
        oauth = oauth_token is not None
        token = oauth_token.token if oauth_token else os.getenv("HF_TOKEN")
        if not token:
            yield (sess, *_views(sess), "Sign in with Hugging Face to run an agent.", gr.update())
            return
        refusal = QUOTA.acquire(getattr(request, "session_hash", "anon") or "anon", oauth)
        if refusal:
            yield (sess, *_views(sess), refusal, gr.update())
            return
        from openenv_daw.runner.client import OpenAIPolicy

        sess.running = True
        text = f"**{model}**\n\n"
        try:
            policy = OpenAIPolicy(model, HF_ROUTER, token, float(temperature))
            adapter = InProcessAdapter.from_env(sess.env)
            task_id = ep.task.task_id
            async for ev in _agent_events(
                policy, adapter, task_id, QUOTA.turns(oauth, int(max_turns))
            ):
                if ev["type"] == "assistant" and ev["message"].get("content"):
                    text += f"{ev['message']['content']}\n\n"
                elif ev["type"] == "call":
                    text += f"`{ev['name']}({json.dumps(ev['args'])[:120]})` → {'ok' if (ev['out'] or {}).get('ok') else 'error'}\n\n"
                elif ev["type"] == "end":
                    sess.records.append(ev["record"])
                yield (sess, *_views(sess), text, gr.update(value=_dump(sess), visible=True))
        except Exception as e:  # network/auth errors from the provider
            yield (sess, *_views(sess), text + f"\n**Error:** {e}", gr.update())
        finally:
            sess.running = False

    def _dump(sess: Session) -> str:
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for r in sess.records:
            f.write(json.dumps(r) + "\n")
        f.close()
        return f.name

    def download(sess):
        if not sess:
            return gr.update(value=None)
        ep = sess.env.episode
        if (
            ep is not None
            and ep.score
            and not any(r["step_log"] == ep.step_log for r in sess.records)
        ):  # a manually played episode
            sess.records.append(
                {
                    "task_id": ep.task.task_id,
                    "intent": ep.task.intent,
                    "reward": ep.score.reward,
                    "success": ep.score.success,
                    "step_log": ep.step_log,
                }
            )
        return gr.update(value=_dump(sess), visible=True)

    with gr.Blocks(title=title) as demo:
        state = gr.State(None, time_to_live=3600, delete_callback=lambda s: s and s.close())
        gr.Markdown(
            f"## DAW Studio\nYour own environment (profile `{profile}`). Edit a project with the same tools an agent gets."
        )
        with gr.Row():
            with gr.Column(scale=3):
                intent = gr.Markdown("Press **Reset** to start an episode.")
                with gr.Row():
                    split = gr.Radio(["train", "heldout"], value="train", label="Split")
                    family = gr.Dropdown(fams, value="(any)", label="Family")
                    number = gr.Number(0, precision=0, label="Seed / heldout index")
                with gr.Row():
                    reset_btn = gr.Button("Reset", variant="primary")
                    random_btn = gr.Button("Random task")
                arrangement = gr.HTML()
                with gr.Accordion("Call a tool", open=True):
                    tool = gr.Dropdown(list(schemas), value=next(iter(schemas)), label="Tool")
                    args = gr.Code(
                        skeleton(schemas[next(iter(schemas))]), language="json", label="Arguments"
                    )
                    run_btn = gr.Button("Run")
                    result = gr.Markdown()
            with gr.Column(scale=2):
                gr.Markdown("### Checks")
                checks = gr.HTML()
                gr.Markdown("### Steps")
                log = gr.HTML()
                with gr.Row():
                    step_btn = gr.Button("Step oracle")
                    play_btn = gr.Button("Play oracle")
                with gr.Accordion("Run agent", open=False):
                    gr.LoginButton()
                    model = gr.Dropdown(models, value=models[0], label="Model")
                    turns = gr.Slider(1, 40, value=20, step=1, label="Max turns")
                    temp = gr.Slider(0, 1.5, value=0.3, step=0.1, label="Temperature")
                    agent_btn = gr.Button("Run agent")
                    transcript = gr.Markdown()
                    dl = gr.Button("Prepare trajectory download (JSONL)")
                    dl_file = gr.File(label="Trajectory", visible=False)
        views = [intent, arrangement, checks, log]
        outs = [state, *views, result]
        reset_btn.click(reset, [state, split, family, number], outs)
        random_btn.click(random_task, [state, split, family, number], outs)
        tool.change(pick_tool, tool, args)
        run_btn.click(manual, [state, tool, args], outs)
        step_btn.click(oracle_step, state, outs)
        play_btn.click(oracle_play, state, outs)
        agent_btn.click(
            run_agent, [state, model, turns, temp], [state, *views, transcript, dl_file]
        )
        dl.click(download, state, dl_file)
    return demo


async def _agent_events(policy, adapter, task_id, max_turns):
    """Run the agent on the visitor's current episode (``start`` is a no-op reset)."""
    env = adapter.env
    ep = env.episode

    class _Live(InProcessAdapter):
        async def start(self, _tid: str) -> str:  # keep the episode the visitor already has
            return ep.task.intent

    live = _Live.from_env(env)
    async for ev in iter_episode(policy, live, task_id, max_turns):
        yield ev
