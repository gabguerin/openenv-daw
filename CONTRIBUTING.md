# Contributing

`openenv-daw` is developed in a private monorepo and mirrored one way (fast-forward only) to this
repository. Pull requests here are welcome; a maintainer ports an accepted PR into the monorepo
(`git am --directory=packages/openenv-daw`, which keeps your authorship) and the next mirror push
closes it.

Catalog discovery: `discovery.json` feeds `openenv catalog build`. The 0.1-draft profile only lists
environments in direct child folders of its root, so the monorepo (`--root packages`) can be
catalogued but this mirror, with the env at the repository root, cannot. Paths in `discovery.json`
are repo-relative (`packages/openenv-daw/...`); update them if the env moves (for example under
`envs/`).

Before opening a PR: `uv sync --all-extras`, then `uv run ruff check .`, `uv run mypy src` and
`uv run pytest`. A change to the tools, the Session schema or the instructions must bump
`TOOLSET_VERSION` / `SCHEMA_VERSION` in `versions.py` and add a line to `CHANGELOG.md`.
