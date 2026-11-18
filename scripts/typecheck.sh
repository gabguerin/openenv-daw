#!/usr/bin/env bash
# mypy for the `openenv init` layout: the package root is this folder, whose name is not a valid
# module name, so expose it as `openenv_daw` through a temporary symlink.
set -euo pipefail
pkg="$(cd "$(dirname "$0")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
ln -s "$pkg" "$tmp/openenv_daw"
cd "$pkg" && MYPYPATH="$tmp" uv run mypy --explicit-package-bases -p openenv_daw.server -p openenv_daw.adapters -m openenv_daw.trl_env -m openenv_daw.client -m openenv_daw.models -m openenv_daw.cli
