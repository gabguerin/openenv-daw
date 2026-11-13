import pytest
from fastmcp import Client

from openenv_daw.backends import SimBackend
from openenv_daw.sim.profiles import PROFILES
from openenv_daw.tools import build_server


@pytest.fixture
def server():
    b = SimBackend()
    return build_server(lambda: b, on_submit=lambda s, r: {"ok": True, "status": s}), b


async def test_lists_portable_tools_plus_submit(server):
    mcp, _ = server
    async with Client(mcp) as c:
        names = {t.name for t in await c.list_tools()}
    assert names == set(PROFILES["portable"]["tools"]) | {"submit"}


async def test_call_roundtrip_and_error_result(server):
    mcp, b = server
    async with Client(mcp) as c:
        r = await c.call_tool("create_track", {"name": "Lead", "kind": "midi"})
        assert r.data["ok"] and r.data["track"]["id"] == "trk_1"
        await c.call_tool(
            "create_midi_clip", {"track": "Lead", "start_beat": 0, "length_beats": 4, "name": "m"}
        )
        r = await c.call_tool(
            "insert_midi_notes",
            {"clip": "m", "notes": [{"pitch": 60, "start_beat": 0, "duration_beats": 1}]},
        )
        assert r.data["note_count"] == 1
        bad = await c.call_tool("delete_track", {"track": "nope"})
        assert bad.data["ok"] is False and bad.data["error"]["code"] == "not_found"
    assert len(b.session.tracks) == 1


async def test_profile_filters_tools():
    b = SimBackend(profile="mixer_only")
    mcp = build_server(lambda: b, profile="mixer_only")
    async with Client(mcp) as c:
        names = {t.name for t in await c.list_tools()}
    assert names == set(PROFILES["mixer_only"]["tools"])


async def test_instructions_exposed(server):
    mcp, _ = server
    assert "get_session" in mcp.instructions
