"""OpenEnv standard surface: /metadata."""

from fastapi.testclient import TestClient
from openenv_daw.server.app import app

client = TestClient(app)


def test_metadata_is_specific():
    m = client.get("/metadata").json()
    assert m["name"] == "openenv_daw"
    assert "DawEnvironment" not in m["description"]
    assert m["documentation_url"]


def test_runtime_plan_replays_to_full_reward():
    """`openenv validate` replays validation/runtime.json; it must load and solve its task."""
    from pathlib import Path

    import yaml
    from openenv.core.env_server.mcp_types import CallToolAction
    from openenv.validation.manifest import ExecutionDeclaration
    from openenv.validation.runtime.contracts import load_runtime_plan
    from openenv_daw.server.daw_environment import DawEnvironment

    root = Path(__file__).resolve().parents[1]
    manifest = yaml.safe_load((root / "openenv.yaml").read_text())
    execution = ExecutionDeclaration.model_validate(manifest["validation"]["execution"])
    plan = load_runtime_plan(root, execution)
    env = DawEnvironment()
    env.reset(seed=plan.reset.seed, episode_id=plan.reset.episode_id, **plan.reset.options)
    obs = None
    for a in plan.actions:
        obs = env.step(CallToolAction(tool_name=a["tool_name"], arguments=a["arguments"]))
    assert obs is not None and obs.done and obs.reward == 1.0
