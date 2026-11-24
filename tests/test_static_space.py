"""The static Space: ``BrowserStudio`` and the bundle that runs in the browser (Pyodide)."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest
import yaml
from openenv.core.env_server.mcp_types import CallToolAction
from openenv_daw.server.daw_environment import DawEnvironment
from openenv_daw.server.tasks.generators import families
from openenv_daw.server.ui.static_bridge import BrowserStudio

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "push_static_space", ROOT / "scripts" / "push_static_space.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["push_static_space"] = mod
    spec.loader.exec_module(mod)
    return mod


script = _load_script()


@pytest.fixture(scope="module")
def tools():
    return script.tool_definitions()


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    return script.build_static_site(tmp_path_factory.mktemp("site") / "static-space")


def _oracle_reward_env(family: str, seed: int) -> tuple[str, float]:
    env = DawEnvironment()
    env.reset(split="train", family=family, seed=seed)
    assert env.episode
    for st in env.episode.task.oracle:
        env.step(CallToolAction(tool_name=st.tool, arguments=st.args))
    assert env.episode.score
    return env.episode.task.task_id, env.episode.score.reward


# --- BrowserStudio ---------------------------------------------------------------------------
def test_lifecycle_reset_manual_oracle_select(tools):
    s = BrowserStudio(tools)
    out = json.loads(s.init())
    assert out["state"]["started"] is False and set(out["views"]) >= {"task", "log"}

    out = json.loads(s.reset("train", "(any)", 4, False))
    assert out["state"]["started"] and out["seed"] == 4 and out["views"]["task"]

    out = json.loads(s.manual("get_session", "{}"))
    assert out["ok"] and out["state"]["step_count"] == 1

    out = json.loads(s.manual("get_session", "[1]"))
    assert not out["ok"] and "JSON object" in out["message"]
    assert not json.loads(s.manual("get_session", "{"))["ok"]

    out = json.loads(s.select_step(0))
    assert out["state"]["step_count"] == 1 and "viewing step 0" in out["views"]["log"]

    out = json.loads(s.manual("create_track", '{"name": 3}'))  # wrong argument types: tool error
    assert out["state"]["step_count"] == 2


def test_reset_family_filter_and_random_seed(tools):
    s = BrowserStudio(tools)
    fam = next(iter(families()))
    a = json.loads(s.reset("heldout", fam, 0, False))
    assert a["state"]["started"]
    r = json.loads(s.reset("train", "(any)", 0, True))
    assert isinstance(r["seed"], int) and r["seed"] >= 0


def test_actions_before_reset_and_after_done(tools):
    s = BrowserStudio(tools)
    assert json.loads(s.manual("get_session", "{}"))["message"] == "Reset first."
    assert json.loads(s.oracle_step())["ok"] is False
    s.reset("train", "(any)", 0, False)
    out = {}
    for _ in range(40):
        out = json.loads(s.oracle_step())
        if out["state"]["done"]:
            break
    assert out["state"]["done"]
    assert json.loads(s.manual("get_session", "{}"))["message"].startswith("The episode is over")


@pytest.mark.parametrize("family", list(families()))
def test_oracle_reward_matches_server_env(tools, family):
    s = BrowserStudio(tools)
    s.reset("train", family, 1, False)
    for _ in range(60):
        if json.loads(s.oracle_step())["state"]["done"]:
            break
    assert s.episode and s.episode.score
    assert (s.episode.task.task_id, s.episode.score.reward) == _oracle_reward_env(family, 1)


# --- the bundle ------------------------------------------------------------------------------
def test_site_files_and_card(site):
    assert {p.name for p in site.iterdir()} == {
        "README.md",
        "app.js",
        "bundle.zip",
        "index.html",
        "tools.json",
    }
    _, front, body = (site / "README.md").read_text().split("---", 2)
    card = yaml.safe_load(front)
    assert card["sdk"] == "static"
    assert not {"app_port", "base_path", "hf_oauth", "hf_oauth_scopes"} & set(card)
    assert "DAW Studio" in body

    pkg = yaml.safe_load((ROOT / "README.md").read_text().split("---", 2)[1])
    assert card["tags"] == pkg["tags"]
    assert {"openenv", "rl-environment", "audio", "ableton"} <= set(card["tags"])
    assert "reinforcement-learning" not in card["tags"]
    discovery = json.loads((ROOT / "discovery.json").read_text())
    assert {"openenv", "audio", "ableton"} <= set(discovery["tags"]) and "rl" in discovery["tags"]


def test_tools_json_matches_registry(site, tools):
    assert json.loads((site / "tools.json").read_text()) == tools
    assert "submit" in {t["name"] for t in tools}


def test_page_references_shipped_files(site):
    html = (site / "index.html").read_text()
    js = (site / "app.js").read_text()
    assert 'src="app.js"' in html
    for name in ("bundle.zip", "tools.json"):
        assert name in js and (site / name).exists()
    assert set(re.findall(r'\$\("([\w-]+)"\)', js)) <= set(re.findall(r'id="([\w-]+)"', html))


def test_bundle_runs_without_server_dependencies(site, tmp_path):
    """Unzip the bundle alone and drive an oracle episode with the heavy imports blocked."""
    with zipfile.ZipFile(site / "bundle.zip") as z:
        z.extractall(tmp_path)
    prog = textwrap.dedent(
        """
        import importlib.abc, json, sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in {"fastmcp", "mcp", "gradio", "httpx", "fastapi", "uvicorn"}:
                    raise ImportError(f"blocked: {name}")

        sys.meta_path.insert(0, Block())
        import openenv_daw, openenv
        assert openenv_daw.__file__.startswith(sys.argv[1]), openenv_daw.__file__
        assert openenv.__file__.startswith(sys.argv[1]), openenv.__file__
        from openenv_daw.server.ui.static_bridge import BrowserStudio

        s = BrowserStudio(json.load(open(sys.argv[2])))
        s.reset("heldout", "(any)", 2, False)
        out = {}
        for _ in range(60):
            out = json.loads(s.oracle_step())
            if out["state"]["done"]:
                break
        print(json.dumps({"done": out["state"]["done"], "reward": s.episode.score.reward,
                          "task": s.episode.task.task_id}))
        """
    )
    res = subprocess.run(
        [sys.executable, "-c", prog, str(tmp_path), str(site / "tools.json")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PATH": "", "PYTHONPATH": str(tmp_path), "PYTHONHASHSEED": "0"},
    )
    assert res.returncode == 0, res.stderr
    got = json.loads(res.stdout.strip().splitlines()[-1])
    assert got["done"] and got["reward"] == 1.0
