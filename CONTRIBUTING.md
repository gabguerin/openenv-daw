# Contributing

`openenv-daw` is developed in a private monorepo and mirrored one way (fast-forward only) to this
repository. Pull requests here are welcome; a maintainer ports an accepted PR into the monorepo
(`git am --directory=packages/openenv-daw`, which keeps your authorship) and the next mirror push
closes it.

Before opening a PR: `uv sync --all-extras`, then `uv run ruff check .`, `scripts/typecheck.sh` and
`uv run pytest`. A change to the tools, the Session schema or the instructions needs a line under
`## Unreleased` in `CHANGELOG.md`.

## Versions

Do not edit `version` in `pyproject.toml` or `openenv.yaml` in a feature change, and do not add
numbered changelog headings. The version stays `0.1.0.dev0` until the first release (0.1.0).

To release:

1. Rename the `Unreleased` heading in `CHANGELOG.md` to the version.
2. Drop the `.devN` suffix in `pyproject.toml` and `openenv.yaml`.
3. Run `scripts/lock_standalone.sh` (and `uv lock` at the repo root).
4. Tag `openenv-daw-v<version>` (PyPI) and run the `space-openenv-daw` workflow (Hugging Face).
5. In the release commit, or the next one, set the next `.dev0` version (for example
   `0.2.0.dev0`) and add a new `## Unreleased` heading.
