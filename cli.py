"""``openenv-daw mcp --backend NAME``: serve any backend as a standalone MCP server over stdio.
The portable tools are exactly the training tools (no ``submit``: the client ends the session)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
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


def serve(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="openenv-daw mcp")
    ap.add_argument("--backend", default="sim", help="entry-point name or module:factory")
    make_server(resolve_backend(ap.parse_args(argv).backend)()).run()


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "mcp":
        sys.exit("usage: openenv-daw mcp --backend NAME")
    serve(sys.argv[2:])


if __name__ == "__main__":
    main()
