# `graitools/` — fork-only experimental area

This directory belongs to the **`graitools/gt4py` experimental fork** only.
Everything under `graitools/` is **never** to be included in a pull request to
`GridTools/gt4py`. It exists so we can trial agentic-development workflows on
top of an otherwise upstream-identical tree.

## Design proposals

`graitools/proposals/` is a **low-barrier scratchpad for design ideas**. Drop a
markdown file here whenever you have an idea, a sketch, or a proposal worth
writing down — well before it is mature enough for a formal ADR. The point is
that anyone (and any agent) can quickly scan what already exists, cross-check a
new idea against it, and surface conflicts early.

This is deliberately **not** the ADR location. `docs/development/ADRs/` holds the
formal, historical, upstreamable decision record. Proposals here are informal
and fork-only; an accepted proposal that turns out to be a real architectural
decision gets **promoted to an ADR** (see `AGENTS.md`).

### File one

1. Copy `proposals/TEMPLATE.md` to `proposals/<short-kebab-slug>.md`.
2. Fill in the frontmatter (`status: idea` to start) and the body.
3. Commit. No numbering, no index, no approval gate — keep it cheap.

Before writing, skim the existing proposals (`ls proposals/`, grep `title:` /
`tags:`) and link related ones via `related:`. The agent workflow is in
[`AGENTS.md`](AGENTS.md).

## Keeping fork-only content out of GridTools

Git has no native "this path never goes upstream" mechanism, so the boundary is
maintained by convention + tooling:

- **Boundary:** all fork-only content lives under `graitools/`. The single
  exception is one clearly-fenced `graitools-only` block at the end of the root
  [`AGENTS.md`](../AGENTS.md) (so agents anywhere discover this area).

- **Preflight:** before opening a PR against `GridTools/gt4py`, run

  ```sh
  bash graitools/scripts/check-fork-boundary.sh gridtools/main <your-branch>
  ```

  It fails if the contribution carries any `graitools/` path or the
  `graitools-only` AGENTS.md block. (`gridtools/main` = a ref tracking
  `GridTools/gt4py` main; add it as a remote if you have not.)

- **CI:** `.github/workflows/fork-boundary.yml` runs the same preflight for
  upstream-staging branches (`upstream/**`). If your team opens GridTools PRs
  directly from feature branches instead, run the preflight by hand.

The hard guarantee can only live in GridTools' own CI, which is not ours to add
from the fork — the above keeps the leak surface tiny and obvious.
