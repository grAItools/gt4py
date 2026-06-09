#!/usr/bin/env bash
# Upstream-contribution preflight: assert that a change set intended for
# GridTools/gt4py carries NO graitools-fork-only content — i.e. nothing under
# graitools/ and not the graitools-only block in the root AGENTS.md.
#
# Run before opening a PR against GridTools/gt4py:
#   bash graitools/scripts/check-fork-boundary.sh gridtools/main <your-branch>
set -euo pipefail

BASE="${1:?usage: $0 <upstream-base-ref> [tip]   e.g. $0 gridtools/main HEAD}"
TIP="${2:-HEAD}"

if ! git rev-parse --verify --quiet "$BASE^{commit}" >/dev/null; then
  echo "fork-boundary: baseline ref '$BASE' not found." >&2
  echo "  Add a remote tracking GridTools/gt4py main, e.g.:" >&2
  echo "    git remote add gridtools https://github.com/GridTools/gt4py.git && git fetch gridtools main" >&2
  exit 2
fi

fail=0

forkfiles="$(git diff --name-only "$BASE" "$TIP" -- 'graitools/' '.github/workflows/fork-boundary.yml')"
if [ -n "$forkfiles" ]; then
  echo "fork-boundary: contribution includes fork-only paths:" >&2
  printf '  %s\n' $forkfiles >&2
  fail=1
fi

if git diff "$BASE" "$TIP" -- AGENTS.md | grep -q 'graitools-only:start'; then
  echo "fork-boundary: contribution touches the graitools-only block in AGENTS.md." >&2
  echo "  Strip the <!-- graitools-only:start ... :end --> section before upstreaming." >&2
  fail=1
fi

if [ "$fail" -ne 0 ]; then
  echo "fork-boundary: FAILED — strip fork-only content before opening a GridTools/gt4py PR." >&2
  exit 1
fi

echo "fork-boundary: OK — no fork-only content in $BASE..$TIP."
