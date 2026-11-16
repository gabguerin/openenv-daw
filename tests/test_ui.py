import json
import os

import pytest
from fastapi.testclient import TestClient

from openenv_daw.env import DawEnvironment
from openenv_daw.rewards.rubrics import score_episode
from openenv_daw.runner.rollout import InProcessAdapter, iter_episode
from openenv_daw.schema import Session
from openenv_daw.server.ui import render
from openenv_daw.server.ui.quota import Quota
from openenv_daw.tasks.generators import families, generate

from .test_runner import OraclePolicy


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


def test_quota_limits():
    now = [0.0]
    q = Quota(runs_per_hour=2, max_turns=5, daily_runs=3, clock=lambda: now[0])
    assert q.acquire("a", False) is None and q.acquire("a", False) is None
    assert q.acquire("a", False)  # per-session cap
    assert q.acquire("b", False) is None
    assert q.acquire("c", False)  # daily cap
    assert q.acquire("c", True) is None  # OAuth exempt
    assert q.turns(False, 40) == 5 and q.turns(True, 40) == 40
    now[0] = 90000
    assert q.acquire("a", False) is None


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


async def test_iter_episode_events():
    from openenv_daw.runner.rollout import select_tasks

    tasks = {t.task_id: t for t in select_tasks("train", 1)}
    tid = next(iter(tasks))
    kinds = [e["type"] async for e in iter_episode(OraclePolicy(tasks), InProcessAdapter(), tid)]
    assert kinds[0] == "start" and kinds[-1] == "end" and "call" in kinds


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
