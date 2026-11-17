import json
import os

import pytest
from fastapi.testclient import TestClient

from openenv_daw.env import DawEnvironment
from openenv_daw.rewards.rubrics import score_episode
from openenv_daw.schema import Session
from openenv_daw.server.ui import render
from openenv_daw.tasks.generators import families, generate


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
    from openenv_daw.sim import ops

    ops.create_track(s, "<script>x</script>", "midi")
    assert "<script>" not in render.session_html(s)


def test_projections_match_score_episode():
    t = generate("mix_balance", 0)
    init = Session.restore(t.initial_state)
    p = render.project(t, init, init)
    assert p["done"] == score_episode(t, init, init, "done", 0, False).reward
    assert p["blocked"] == score_episode(t, init, init, "blocked", 0, False).reward


def test_studio_builds_with_independent_sessions():
    import gradio as gr

    from openenv_daw.server.ui.studio import Session as UiSession
    from openenv_daw.server.ui.studio import build_daw_studio

    demo = build_daw_studio(None, [], None, False, "t", None, env_factory=DawEnvironment)
    assert isinstance(demo, gr.Blocks)
    a, b = UiSession(DawEnvironment()), UiSession(DawEnvironment())
    a.env.reset(task_id="melody_in_key-train-0")
    b.env.reset(task_id="reverb_send-train-0")
    assert a.env.episode.task.task_id != b.env.episode.task.task_id


def test_web_root_serves_with_interface(monkeypatch):
    monkeypatch.setenv("ENABLE_WEB_INTERFACE", "true")
    import importlib

    from openenv_daw.server import app as app_mod

    app_mod = importlib.reload(app_mod)
    try:
        r = TestClient(app_mod.app).get("/web/")
        assert r.status_code == 200
    finally:
        monkeypatch.delenv("ENABLE_WEB_INTERFACE")
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
    from openenv_daw.schema import Track

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
