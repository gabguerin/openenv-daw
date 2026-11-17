# Contributing

`openenv-daw` is developed in a private monorepo and mirrored one way (fast-forward only) to this
repository. Pull requests here are welcome; a maintainer ports an accepted PR into the monorepo
(`git am --directory=packages/openenv-daw`, which keeps your authorship) and the next mirror push
closes it.

Before opening a PR: `uv sync --all-extras`, then `uv run ruff check .`, `scripts/typecheck.sh` and
`uv run pytest`. A change to the tools, the Session schema or the instructions needs a line in
`CHANGELOG.md`.
