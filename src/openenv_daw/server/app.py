"""ASGI entry point: ``uvicorn openenv_daw.server.app:app``."""

from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from openenv_daw.backends.registry import resolve_backend
from openenv_daw.env.environment import DawEnvironment
from openenv_daw.env.models import DawResetObservation, DawResetRequest, DawState
from openenv_daw.sim.profiles import DEFAULT_PROFILE


def _factory() -> DawEnvironment:
    backend = os.getenv("OPENENV_DAW_BACKEND")
    return DawEnvironment(
        profile=os.getenv("OPENENV_DAW_PROFILE", DEFAULT_PROFILE),
        backend_factory=resolve_backend(backend) if backend else None,
    )


app = create_app(
    _factory,
    CallToolAction,
    CallToolObservation,
    env_name="openenv_daw",
    max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "64")),
    state_cls=DawState,
    reset_cls=DawResetRequest,
    reset_observation_cls=DawResetObservation,
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
