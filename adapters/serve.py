"""Serve any backend as a standalone MCP server over stdio: ``openenv-daw mcp --backend NAME``.
The portable tools are exactly the training tools (no ``submit``: the client ends the session)."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any

from fastmcp import FastMCP
from openenv_daw.server.backends import Backend, resolve_backend
from openenv_daw.server.tools import build_server


def make_server(backend: Backend, middleware: Sequence[Any] = ()) -> FastMCP:
    """``middleware``: FastMCP middleware (for example a safety gate with elicitation)."""
    mcp = build_server(lambda: backend)
    for m in middleware:
        mcp.add_middleware(m)
    return mcp


def main(
    argv: list[str] | None = None,
    middleware_factory: Callable[[Backend], Sequence[Any]] | None = None,
) -> None:
    ap = argparse.ArgumentParser(prog="openenv-daw mcp")
    ap.add_argument("--backend", default="sim", help="entry-point name or module:factory")
    a = ap.parse_args(argv)
    backend = resolve_backend(a.backend)()
    mws = middleware_factory(backend) if middleware_factory else ()
    make_server(backend, mws).run()
