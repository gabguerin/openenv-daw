"""Typed tool errors shared by the simulator and DAW adapters."""

from __future__ import annotations

from typing import Any, Literal

ErrorCode = Literal["not_found", "invalid_args", "conflict", "unsupported", "execution_error"]


class DawError(Exception):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code: ErrorCode = code
        self.message = message

    def to_result(self) -> dict[str, Any]:
        return {"ok": False, "error": {"code": self.code, "message": self.message}}


def not_found(msg: str) -> DawError:
    return DawError("not_found", msg)


def invalid(msg: str) -> DawError:
    return DawError("invalid_args", msg)


def conflict(msg: str) -> DawError:
    return DawError("conflict", msg)


def unsupported(msg: str) -> DawError:
    return DawError("unsupported", msg)


SimError = DawError
