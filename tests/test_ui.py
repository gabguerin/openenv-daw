import json
import os

import pytest
from fastapi.testclient import TestClient
from openenv_daw.server.daw_environment import DawEnvironment
from openenv_daw.server.rewards.score import score_episode
from openenv_daw.server.sim.session import Session
from openenv_daw.server.tasks.generators import families, generate
from openenv_daw.server.ui import render


@pytest.mark.parametrize("family", list(families()))
def test_render_oracle_final_state(family):
    t = generate(family, 1)
    env = DawEnvironment()
    env.reset(task_id=t.task_id)
    from openenv.core.env_server.mcp_types import CallToolAction

    for st in t.oracle:
        env.step(CallToolAction(tool_name=st.tool, arguments=st.args))
    ep = env.episode
    html = render.session_html(ep.backend.session)
    for tr in ep.backend.session.tracks:
        assert tr.name.replace("&", "&amp;") in html
    assert "Episode over" in render.checks_html(
        ep.task, ep.initial, ep.backend.session, 0, ep.score.as_dict()
    )
    assert render.step_log_html(ep.step_log)


def test_names_are_escaped():
    s = Session()
    from openenv_daw.server.sim import ops

    ops.create_track(s, "<script>x</script>", "midi")
    assert "<script>" not in render.session_html(s)


def test_projections_match_score_episode():
    t = generate("mix_balance", 0)
    init = Session.restore(t.initial_state)
    p = render.project(t, init, init)
    assert p["done"] == score_episode(t, init, init, "done", 0, False).reward
    assert p["blocked"] == score_episode(t, init, init, "blocked", 0, False).reward


def test_studio_builds_with_independent_sessions(monkeypatch):
    import gradio as gr
    from openenv_daw.server.ui.studio import Session as UiSession
    from openenv_daw.server.ui.studio import build_daw_studio

    monkeypatch.delenv("SPACE_ID", raising=False)
    demo_local = build_daw_studio(None, [], None, False, "t", None, env_factory=DawEnvironment)
    assert isinstance(demo_local, gr.Blocks)

    monkeypatch.setenv("SPACE_ID", "gabguerin/openenv-daw")
    demo_space = build_daw_studio(None, [], None, False, "t", None, env_factory=DawEnvironment)
    assert isinstance(demo_space, gr.Blocks)

    a, b = UiSession(DawEnvironment()), UiSession(DawEnvironment())
    a.env.reset(task_id="melody_in_key-train-0")
    b.env.reset(task_id="reverb_send-train-0")
    assert a.env.episode.task.task_id != b.env.episode.task.task_id


def test_web_root_serves_with_interface(monkeypatch):
    monkeypatch.setenv("ENABLE_WEB_INTERFACE", "true")
    monkeypatch.delenv("SPACE_ID", raising=False)
    import importlib

    from openenv_daw.server import app as app_mod

    app_mod = importlib.reload(app_mod)
    try:
        r = TestClient(app_mod.app).get("/web/")
        assert r.status_code == 200
        monkeypatch.setenv("SPACE_ID", "gabguerin/openenv-daw")
        app_mod = importlib.reload(app_mod)
        client = TestClient(app_mod.app, follow_redirects=False)
        assert client.get("/web/").status_code == 200
        assert client.get("/login/huggingface").status_code == 307
        assert client.get("/logout").status_code == 307
    finally:
        monkeypatch.delenv("ENABLE_WEB_INTERFACE", raising=False)
        monkeypatch.delenv("SPACE_ID", raising=False)
        importlib.reload(app_mod)
        json, os  # noqa: B018


def _play(task_id):
    from openenv.core.env_server.mcp_types import CallToolAction

    env = DawEnvironment()
    env.reset(task_id=task_id)
    ep = env.episode
    snaps = [ep.initial.snapshot()]
    for st in ep.task.oracle:
        env.step(CallToolAction(tool_name=st.tool, arguments=st.args))
        snaps.append(ep.backend.session.snapshot())
    return env, ep, snaps


def test_track_diff_flags_new_and_changed():
    _, ep, snaps = _play("reverb_send-train-0")
    prev = Session.restore(snaps[1])  # right after create_track
    flags = render.track_diff(ep.initial, prev, ep.backend.session)
    assert any("new" in f for f in flags.values())  # the Reverb return track
    assert any("changed" in f for f in flags.values())  # Pad got a send
    assert any("touched" in f for f in flags.values())


def test_arrangement_markers_for_foreign_and_damage():
    from openenv_daw.server.sim.session import Track

    s = Session()
    s.tracks.append(Track(id="trk_1", name="Bus", kind="group"))
    s.tracks.append(Track(id="trk_2", name="Keys", kind="midi"))
    html = render.arrangement_html(s, s, None, "trk_2", ["extra clip on Keys"])
    assert "ds-foreign" in html and "🔒" in html
    assert html.count("ds-dmg") == 2 and 'data-track="trk_2"' in html


def test_studio_views_and_snapshots_align():
    from openenv_daw.server.ui.studio import Session as UiSession
    from openenv_daw.server.ui.studio import _views

    sess = UiSession(DawEnvironment())
    sess.env.reset(task_id="reverb_send-train-0")
    from openenv.core.env_server.mcp_types import CallToolAction

    for st in sess.env.episode.task.oracle:
        sess.env.step(CallToolAction(tool_name=st.tool, arguments=st.args))
    views = _views(sess)
    assert len(sess.snapshots) == sess.env.episode.step_count + 1
    assert len(views) == 5
    sess.view_step = 0
    assert "viewing step 0" in _views(sess)[4]


async def test_step_async_inside_running_loop():
    from openenv_daw.server.ui.studio import Session as UiSession
    from openenv_daw.server.ui.studio import _call

    sess = UiSession(DawEnvironment())
    sess.env.reset(task_id="reverb_send-train-0")
    await _call(sess, "get_session", {})
    assert sess.env.episode.step_log[-1]["ok"]


def test_reset_kwargs_heldout_family_and_wrap():
    from openenv_daw.server.ui.studio import reset_kwargs

    env = DawEnvironment()
    kw = reset_kwargs("heldout", "melody_in_key", 9999)
    env.reset(**kw)
    assert env.episode.task.family == "melody_in_key"
    assert reset_kwargs("train", "(any)", -3)["seed"] == 0
    assert "family" not in reset_kwargs("train", "(any)", 1)


def test_skeleton_uses_enum():
    import json as _json

    from openenv_daw.server.ui.studio import skeleton

    out = _json.loads(
        skeleton(
            {
                "properties": {"kind": {"type": "string", "enum": ["midi", "audio"]}},
                "required": ["kind"],
            }
        )
    )
    assert out["kind"] == "midi"


def test_episode_log_marks_current_step_and_checks_delta():
    _, ep, snaps = _play("reverb_send-train-0")
    html = render.episode_html(ep.step_log, 2, "step 2", True)
    assert html.count("data-step=") == len(ep.step_log) + 1  # plus the reset row
    assert html.count("ds-cur") == 1 and 'data-step="2" class="ds-cur"' in html
    prev = Session.restore(snaps[0])
    out = render.dark_checks_html(ep.task, ep.initial, ep.backend.session, 0, None, prev)
    assert "now passing" in out


def test_python_snippet_uses_last_call():
    _, ep, _ = _play("reverb_send-train-0")
    html = render.python_html({"split": "train", "seed": 3}, ep.step_log[0])
    assert "env.reset(split=&#x27;train&#x27;, seed=3)" in html
    assert f"env.call(&quot;{ep.step_log[0]['tool']}&quot;" in html


def test_agent_openai_tools_preserves_names_and_schemas():
    from openenv.core.env_server.mcp_types import ListToolsAction
    from openenv_daw.server.ui import agent

    env = DawEnvironment()
    tools = env.step(ListToolsAction()).tools  # type: ignore[attr-defined]
    env.close()
    oa = agent.openai_tools(tools)
    assert len(oa) == len(tools)
    assert [x["function"]["name"] for x in oa] == [t.name for t in tools]
    for spec, t in zip(oa, tools, strict=True):
        assert spec["type"] == "function"
        assert spec["function"]["parameters"] == t.input_schema


def test_agent_build_messages_replays_oracle_prefix():
    from openenv.core.env_server.mcp_types import CallToolAction
    from openenv_daw.server.ui import agent

    env = DawEnvironment()
    env.reset(task_id="reverb_send-train-0")
    prefix = env.episode.task.oracle[:2]
    for st in prefix:
        env.step(CallToolAction(tool_name=st.tool, arguments=st.args))
    msgs = agent.build_messages(env.episode.task.intent, env.episode.step_log)
    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "user", "content": env.episode.task.intent}
    tail = msgs[2:]
    assert len(tail) == 2 * len(prefix)
    for i in range(len(prefix)):
        asst, tool_msg = tail[2 * i], tail[2 * i + 1]
        assert asst["role"] == "assistant"
        assert tool_msg["role"] == "tool"
        call = asst["tool_calls"][0]
        assert call["id"] == tool_msg["tool_call_id"]
        assert call["function"]["name"] == prefix[i].tool
        assert json.loads(call["function"]["arguments"]) == prefix[i].args
        assert json.loads(tool_msg["content"])["ok"] is True


async def test_agent_next_call_variants():
    import httpx
    from openenv_daw.server.ui import agent

    cfg = agent.ModelConfig(base_url="http://mock.local/v1", model="test-model", api_key="secret")
    msgs = [{"role": "user", "content": "check session"}]
    tools = [{"type": "function", "function": {"name": "get_session", "parameters": {}}}]

    # 1. Native tool_calls reply
    def native_handler(req: httpx.Request) -> httpx.Response:
        assert req.url == httpx.URL("http://mock.local/v1/chat/completions")
        assert req.headers["authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Checking session.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_session",
                                        "arguments": '{"include_notes": true}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    r1 = await agent.next_call(cfg, msgs, tools, transport=httpx.MockTransport(native_handler))
    assert r1.error is None
    assert r1.tool == "get_session"
    assert r1.args == {"include_notes": True}
    assert r1.text == "Checking session."

    # 2. <tool_call> text fallback
    def tag_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                'Let me inspect.\n<tool_call>{"name": "get_session", '
                                '"arguments": {"track": "Pad"}}</tool_call>'
                            ),
                        }
                    }
                ]
            },
        )

    r2 = await agent.next_call(cfg, msgs, tools, transport=httpx.MockTransport(tag_handler))
    assert r2.error is None
    assert r2.tool == "get_session"
    assert r2.args == {"track": "Pad"}
    assert r2.text == "Let me inspect."

    # 3. No tool call
    def plain_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "All done."}}]},
        )

    r3 = await agent.next_call(cfg, msgs, tools, transport=httpx.MockTransport(plain_handler))
    assert r3.error is None
    assert r3.tool is None
    assert r3.text == "All done."

    # 4. HTTP 401 -> readable error
    def unauth_handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid token"}})

    r4 = await agent.next_call(cfg, msgs, tools, transport=httpx.MockTransport(unauth_handler))
    assert r4.tool is None
    assert r4.error is not None
    assert "401" in r4.error and "Invalid token" in r4.error


async def test_studio_model_step_executes_tool():
    from openenv_daw.server.ui import agent
    from openenv_daw.server.ui.studio import Session as UiSession
    from openenv_daw.server.ui.studio import step_with_model

    sess = UiSession(DawEnvironment())
    sess.env.reset(task_id="reverb_send-train-0")
    cfg = agent.ModelConfig(base_url="http://mock.local/v1", model="test-model")

    async def fake_caller(_cfg, _msgs, _tools):
        return agent.ModelReply(tool="get_session", args={}, text="Inspecting project.")

    ok, msg = await step_with_model(sess, cfg, [], caller=fake_caller)
    assert ok is True
    assert len(sess.env.episode.step_log) == 1
    assert sess.env.episode.step_log[0]["tool"] == "get_session"
    assert sess.env.episode.step_log[0]["ok"] is True
    assert "`get_session`" in msg
    assert "Inspecting project." in msg
