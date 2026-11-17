import importlib.metadata
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    m = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M)
    assert m
    return m.group(1)


def test_manifest_version_matches_pyproject():
    manifest = yaml.safe_load((ROOT / "openenv.yaml").read_text())
    assert str(manifest["version"]) == _pyproject_version()


def test_installed_version_matches_pyproject():
    assert importlib.metadata.version("openenv-daw") == _pyproject_version()


def test_changelog_top_heading_matches_version_stage():
    version = _pyproject_version()
    top = next(
        line for line in (ROOT / "CHANGELOG.md").read_text().splitlines() if line.startswith("## ")
    )
    if re.search(r"\.dev\d+$", version):
        assert top.startswith("## Unreleased"), "dev versions log changes under 'Unreleased'"
    else:
        assert top == f"## {version}"
