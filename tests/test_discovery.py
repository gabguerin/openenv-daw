"""discovery.json is valid for ``openenv catalog`` and matches the real tool set."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from openenv.core.env_server.mcp_environment import get_server_tools
from openenv.discovery.metadata import DiscoveryDeclaration

from openenv_daw.backends import SimBackend
from openenv_daw.tools import build_server

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "packages/openenv-daw/"  # paths are repo-relative in the monorepo


def _declaration() -> DiscoveryDeclaration:
    return DiscoveryDeclaration.model_validate(json.loads((ROOT / "discovery.json").read_text()))


def _local(path: str) -> Path:
    return ROOT / path.removeprefix(PREFIX)


def test_declaration_is_valid():
    d = _declaration()
    assert d.agent_tools is not None and d.license == "Apache-2.0"
    assert 2 <= len(d.representative_queries) <= 5


def test_tools_match_registry_and_manifest():
    d = _declaration()
    server = build_server(lambda: SimBackend(), on_submit=lambda s, r: {})
    registry = set(get_server_tools(server))
    manifest = yaml.safe_load((ROOT / "openenv.yaml").read_text())
    declared = manifest["validation"]["capabilities"]["declared_tools"]
    assert set(d.agent_tools.names) == registry == set(declared)


def test_evidence_files_exist():
    d = _declaration()
    assert _local(d.agent_tools.source).is_file()
    assert _local(d.license_source).is_file()
