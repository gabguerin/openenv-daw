#!/usr/bin/env bash
# Regenerate packages/openenv-daw/uv.lock for the Space image (`uv sync --frozen`), which builds the
# package on its own, outside the monorepo workspace. Every dependency is pinned to the version the
# monorepo uv.lock resolved, so the image runs what the tests ran.
# Run from anywhere; CI re-runs it and fails on a diff.
set -euo pipefail
pkg="$(cd "$(dirname "$0")/.." && pwd)"
root="$(cd "$pkg/../.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

cp "$pkg/pyproject.toml" "$pkg/README.md" "$pkg/LICENSE" "$tmp/"

pins="$(cd "$root" && uv export --package openenv-daw --frozen --all-extras --no-dev \
  --no-emit-workspace --no-hashes --no-annotate --no-header)"
{
  printf '\n[tool.uv]\nconstraint-dependencies = [\n'
  grep -v '^\s*$' <<<"$pins" | sed 's/\\/\\\\/g; s/"/\\"/g; s/^/  "/; s/$/",/'
  printf ']\n'
} >>"$tmp/pyproject.toml"

(cd "$tmp" && uv lock --no-progress -q)
cp "$tmp/uv.lock" "$pkg/uv.lock"
echo "wrote $pkg/uv.lock"
