"""``openenv-daw`` command line: ``openenv-daw mcp --backend NAME --profile P``."""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] != "mcp":
        sys.exit("usage: openenv-daw mcp --backend NAME [--profile P]")
    from openenv_daw.adapters.serve import main as serve

    serve(sys.argv[2:])


if __name__ == "__main__":
    main()
