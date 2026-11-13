"""Typed simulator errors. Adapters for real DAWs should raise the same codes."""

from __future__ import annotations

from typing import Any, Literal

ErrorCode = Literal["not_found", "invalid_args", "conflict", "unsupported"]


class SimError(Exception):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code: ErrorCode = code
        self.message = message

    def to_result(self) -> dict[str, Any]:
        return {"ok": False, "error": {"code": self.code, "message": self.message}}


def not_found(msg: str) -> SimError:
    return SimError("not_found", msg)


def invalid(msg: str) -> SimError:
    return SimError("invalid_args", msg)


def conflict(msg: str) -> SimError:
    return SimError("conflict", msg)


def unsupported(msg: str) -> SimError:
    return SimError("unsupported", msg)
