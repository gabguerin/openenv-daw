#!/usr/bin/env python3
"""Build (and optionally push) the Static Hugging Face Space for DAW Studio.

The static Space needs no server: Pyodide runs the real simulator, tasks and rubric in the
visitor's browser (``server/ui/static_bridge.py`` is the Python side, ``server/ui/static/`` the
page). It is free (Docker and Gradio Spaces on ``cpu-basic`` need a Hugging Face PRO account) and
supports Sign in with Hugging Face, so the model call can be billed to the visitor.

    python scripts/push_static_space.py --out dist/static-space                # build only
    python scripts/push_static_space.py --push --repo-id gabguerin/openenv-daw # build and push

The Docker Space (``openenv push``) stays available for the full OpenEnv server API.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "gabguerin/openenv-daw"
STATIC_DIR = PACKAGE_ROOT / "server" / "ui" / "static"

# Source files shipped to the browser, relative to the package root, under ``openenv_daw/``.
# Everything here must stay free of fastmcp, gradio, httpx and openenv imports.
BUNDLE_FILES = [
    "server/prompt.py",
    "server/episode.py",
    "server/backends.py",
    "server/music.py",
    "server/instructions.md",
    "server/ui/__init__.py",
    "server/ui/agent.py",
    "server/ui/render.py",
    "server/ui/static_bridge.py",
]
BUNDLE_DIRS = ["server/sim", "server/rewards", "server/tasks"]
BUNDLE_SUFFIXES = {".py", ".yaml", ".jsonl", ".md"}

# The only parts of ``openenv`` the rubric needs: the rubric base classes and the Observation
# type. Copied verbatim from the installed package (BSD-3-Clause, see openenv/NOTICE.txt).
VENDORED_OPENENV = {
    "openenv.core.rubrics.base": "openenv/core/rubrics/base.py",
    "openenv.core.rubrics.containers": "openenv/core/rubrics/containers.py",
    "openenv.core.env_server.types": "openenv/core/env_server/types.py",
}
OPENENV_SHIMS = {
    "openenv/__init__.py": "",
    "openenv/core/__init__.py": "",
    "openenv/core/env_server/__init__.py": "",
    "openenv/core/rubrics/__init__.py": (
        "from openenv.core.rubrics.base import Rubric\n"
        "from openenv.core.rubrics.containers import (\n"
        "    Gate,\n    RubricDict,\n    RubricList,\n    Sequential,\n    WeightedSum,\n)\n\n"
        '__all__ = ["Gate", "Rubric", "RubricDict", "RubricList", "Sequential", "WeightedSum"]\n'
    ),
    "openenv/NOTICE.txt": (
        "base.py, containers.py and types.py are copied unmodified from the openenv-core package\n"
        "(https://github.com/meta-pytorch/OpenEnv, BSD-3-Clause) so the rubric runs in the browser.\n"
    ),
}


def bundle_entries() -> dict[str, bytes]:
    """Archive path -> file content for the browser bundle."""
    entries: dict[str, bytes] = {
        "openenv_daw/__init__.py": b"",
        "openenv_daw/server/__init__.py": b"",
    }
    paths = [PACKAGE_ROOT / f for f in BUNDLE_FILES]
    for d in BUNDLE_DIRS:
        paths += sorted(
            p
            for p in (PACKAGE_ROOT / d).rglob("*")
            if p.is_file() and p.suffix in BUNDLE_SUFFIXES and "__pycache__" not in p.parts
        )
    for p in paths:
        entries[f"openenv_daw/{p.relative_to(PACKAGE_ROOT).as_posix()}"] = p.read_bytes()

    for module, arcname in VENDORED_OPENENV.items():
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"cannot locate {module}; install openenv-core")
        entries[arcname] = Path(spec.origin).read_bytes()
    for arcname, text in OPENENV_SHIMS.items():
        entries[arcname] = text.encode()
    return entries


def write_bundle(path: Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for arcname, data in sorted(bundle_entries().items()):
            z.writestr(zipfile.ZipInfo(arcname, date_time=(2020, 1, 1, 0, 0, 0)), data)


def tool_definitions() -> list[dict[str, Any]]:
    """The environment's tools as ``{name, description, input_schema}``, read from the real
    FastMCP registry so the browser never drifts from ``server/tools.py``."""
    from openenv.core.env_server.mcp_types import ListToolsAction
    from openenv_daw.server.daw_environment import DawEnvironment

    env = DawEnvironment()
    try:
        tools = env.step(ListToolsAction()).tools  # type: ignore[attr-defined]
        return [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in tools
        ]
    finally:
        env.close()


def space_readme() -> str:
    """The Space card: the package README's front matter with ``sdk: static``."""
    text = (PACKAGE_ROOT / "README.md").read_text()
    _, front, _ = text.split("---", 2)
    meta: dict[str, Any] = yaml.safe_load(front)
    for key in ("app_port", "base_path", "app_file"):
        meta.pop(key, None)
    meta["sdk"] = "static"
    card = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).rstrip()
    body = (
        "# DAW Studio\n\n"
        "Edit a music project with portable DAW tools, manually, with the oracle, or with a model,"
        " and get scored by the environment's rubric. The simulator, tasks and rewards of"
        " [openenv-daw](https://github.com/gabguerin/daw-agent) run in your browser through"
        " Pyodide; only the model call leaves the page.\n\n"
        "- Models: sign in with Hugging Face to use Inference Providers (billed to you), or point"
        " it at any OpenAI-compatible URL.\n"
        "- Trained model: [gabguerin/openenv-daw-lfm3-1.7b-sft](https://huggingface.co/gabguerin/openenv-daw-lfm3-1.7b-sft)\n"
        "- Tasks: [gabguerin/openenv-daw-tasks](https://huggingface.co/datasets/gabguerin/openenv-daw-tasks)\n"
    )
    return f"---\n{card}\n---\n\n{body}"


def build_static_site(out_dir: Path) -> Path:
    """Write the complete static site into ``out_dir`` (recreated) and return it."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copy(STATIC_DIR / "index.html", out_dir / "index.html")
    shutil.copy(STATIC_DIR / "app.js", out_dir / "app.js")
    write_bundle(out_dir / "bundle.zip")
    (out_dir / "tools.json").write_text(json.dumps(tool_definitions(), indent=1))
    (out_dir / "README.md").write_text(space_readme())
    return out_dir


def push_static_site(out_dir: Path, repo_id: str, token: str | None = None) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="space",
        delete_patterns="*",
        commit_message="Update static DAW Studio",
    )
    return f"https://huggingface.co/spaces/{repo_id}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=PACKAGE_ROOT / "dist" / "static-space")
    ap.add_argument("--push", action="store_true", help="upload the built site to the Hub")
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    args = ap.parse_args()
    site = build_static_site(args.out)
    print(f"built {site} ({sum(1 for p in site.iterdir())} files)")
    if args.push:
        print(f"pushed {push_static_site(site, args.repo_id)}")


if __name__ == "__main__":
    main()
