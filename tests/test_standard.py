"""OpenEnv standard surface: /metadata."""

from fastapi.testclient import TestClient
from openenv_daw.server.app import app

client = TestClient(app)


def test_metadata_is_specific():
    m = client.get("/metadata").json()
    assert m["name"] == "openenv_daw"
    assert "DawEnvironment" not in m["description"]
    assert m["documentation_url"]
