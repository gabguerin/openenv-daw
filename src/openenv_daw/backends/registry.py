"""Find backends by name: entry points in the ``openenv_daw.backends`` group, or a
``module:factory`` path for adapters that are not packaged. A factory takes the profile name and
returns a ``Backend``."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from importlib.metadata import entry_points

from .base import Backend

GROUP = "openenv_daw.backends"


def available_backends() -> list[str]:
    return sorted(ep.name for ep in entry_points(group=GROUP))


def resolve_backend(name_or_path: str) -> Callable[[str], Backend]:
    """Return a ``factory(profile) -> Backend``."""
    if ":" in name_or_path:
        module, _, attr = name_or_path.partition(":")
        target = getattr(importlib.import_module(module), attr)
    else:
        eps = [ep for ep in entry_points(group=GROUP) if ep.name == name_or_path]
        if not eps:
            raise KeyError(f"unknown backend {name_or_path!r}; have {available_backends()}")
        target = eps[0].load()
    return lambda profile: target(profile=profile)
