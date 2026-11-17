"""ASGI entry point: ``uvicorn openenv_daw.server.app:app``."""

from __future__ import annotations

import os
from functools import partial

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
from openenv_daw.models import DawResetObservation, DawResetRequest, DawState
from openenv_daw.server.backends import resolve_backend
from openenv_daw.server.daw_environment import DawEnvironment


def _factory() -> DawEnvironment:
    backend = os.getenv("OPENENV_DAW_BACKEND")
    return DawEnvironment(backend_factory=resolve_backend(backend) if backend else None)


def _ui_kwargs() -> dict:
    """The Studio tab is only built (and gradio only imported) when the web interface is on."""
    if os.getenv("ENABLE_WEB_INTERFACE", "false").lower() not in ("true", "1", "yes"):
        return {}
    from openenv_daw.server.ui.studio import build_daw_studio

    return {
        "gradio_builder": partial(build_daw_studio, env_factory=_factory),
        "show_default_tab": False,
        "title_override": "openenv-daw: DAW tool-use environment",
    }


app = create_app(
    _factory,
    CallToolAction,
    CallToolObservation,
    env_name="openenv_daw",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "64")),
    state_cls=DawState,
    reset_cls=DawResetRequest,
    reset_observation_cls=DawResetObservation,
    **_ui_kwargs(),
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
