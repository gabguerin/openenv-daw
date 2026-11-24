"""The system prompt. Kept free of heavy imports so the browser (Pyodide) bundle can load it."""

from __future__ import annotations

from importlib import resources

from openenv_daw.server.sim.session import GRID_DIVISIONS


def load_instructions() -> str:
    """The system prompt. Training and deployment both call this function."""
    text = resources.files("openenv_daw.server").joinpath("instructions.md").read_text()
    return text.replace("{GRID}", str(GRID_DIVISIONS)).rstrip() + "\n"
