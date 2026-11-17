"""OpenEnv standard surface: /metadata and the Task API."""

from fastapi.testclient import TestClient

from openenv_daw.server.app import app

client = TestClient(app)


def test_metadata_is_specific():
    m = client.get("/metadata").json()
    assert m["name"] == "openenv_daw"
    assert "DawEnvironment" not in m["description"]
    assert m["documentation_url"]


def test_splits_and_counts():
    splits = client.get("/openenv_daw/splits").json()
    assert {s["name"] for s in splits} == {"train", "heldout"}
    n = client.post("/openenv_daw/num_tasks", json={"split": "heldout"}).json()
    assert n["num_tasks"] == 212


def test_task_payload_has_no_oracle():
    r = client.post("/openenv_daw/task", json={"split": "heldout", "index": 0})
    assert r.status_code == 200
    task = r.json()["task"]
    assert task["intent"] and task["task_id"]
    assert not {"oracle", "checks", "initial_state"} & set(task)


def test_train_preview_is_bounded():
    r = client.post("/openenv_daw/tasks", json={"split": "train"})
    assert r.status_code == 200
    tasks = r.json()["tasks"]
    assert 0 < len(tasks) <= 20


def test_unknown_split_is_a_client_error():
    r = client.post("/openenv_daw/task", json={"split": "nope", "index": 0})
    assert r.status_code == 400
