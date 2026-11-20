"""OpenEnv standard surface: /metadata."""

from fastapi.testclient import TestClient
from openenv_daw.server.app import app

client = TestClient(app)


def test_metadata_is_specific():
    m = client.get("/metadata").json()
    assert m["name"] == "openenv_daw"
    assert "DawEnvironment" not in m["description"]
    assert m["documentation_url"]


def test_task_api_routes():
    assert client.get("/list_environments").json() == ["openenv_daw"]
    splits = [s["name"] for s in client.get("/openenv_daw/splits").json()]
    assert splits == ["train", "heldout"]
    n = client.post("/openenv_daw/num_tasks", json={"split": "heldout"}).json()["num_tasks"]
    assert n == 242
    body = {"split": "heldout", "start": 0, "stop": 3}
    tasks = client.post("/openenv_daw/task_range", json=body).json()["tasks"]
    assert len(tasks) == 3 and tasks[0]["intent"]
    bad = client.post("/openenv_daw/task", json={"split": "heldout", "index": 242})
    assert bad.status_code == 400


def test_task_api_matches_reset_index():
    from openenv_daw.server.daw_environment import DawEnvironment

    env = DawEnvironment()
    for split in ("train", "heldout"):
        spec = env.get_task(split, 5)
        assert env.reset(split=split, index=5).task_id == spec["id"]


def test_readme_quickstart_scores_one():
    """The steps shown in the README quickstart."""
    from openenv.core.env_server.mcp_types import CallToolAction
    from openenv_daw.server.daw_environment import DawEnvironment

    from openenv_daw.server.tasks.store import default_store

    env = DawEnvironment()
    task = default_store().get("mix_balance-train-0")  # the README's task; its wording varies
    assert env.reset(task_id="mix_balance-train-0").intent == task.intent

    def call(tool, **kw):
        return env.step(CallToolAction(tool_name=tool, arguments=kw))

    call("get_session")
    for st in task.oracle:
        r = call(st.tool, **st.args)
    assert r.done and r.reward == 1.0 and r.metadata["success"] is True


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
