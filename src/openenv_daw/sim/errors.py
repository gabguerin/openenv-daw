"""Re-export of the shared error types (kept for backwards compatibility)."""

from __future__ import annotations

from openenv_daw.errors import (
    DawError,
    ErrorCode,
    conflict,
    invalid,
    not_found,
    unsupported,
)

SimError = DawError

__all__ = [
    "DawError",
    "ErrorCode",
    "SimError",
    "conflict",
    "invalid",
    "not_found",
    "unsupported",
]
