"""openenv-daw: an OpenEnv environment for DAW tool use, and an SDK for DAW adapters."""

__version__ = "0.2.0"

from .errors import DawError
from .versions import SCHEMA_VERSION, TOOLSET_VERSION

__all__ = ["SCHEMA_VERSION", "TOOLSET_VERSION", "DawError", "__version__"]

_LAZY = {
    "Episode": "openenv_daw.env.episode",
    "TaskStore": "openenv_daw.tasks.store",
    "load_tasks": "openenv_daw.tasks.store",
    "register_check": "openenv_daw.rewards.checks",
    "register_profile": "openenv_daw.sim.profiles",
    "DawToolEnv": "openenv_daw.trl_env",
    "build_dataset": "openenv_daw.trl_env",
    "resolve_backend": "openenv_daw.backends.registry",
}
__all__ += sorted(_LAZY)


def __getattr__(name: str):  # lazy: keeps ``import openenv_daw`` light
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(name)
