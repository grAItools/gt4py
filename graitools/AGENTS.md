# `graitools/` — Agent Instructions (fork-only)

This directory is **fork-only** and must never appear in a PR to
`GridTools/gt4py`. See [`README.md`](README.md) for the boundary rules and the
upstream-contribution preflight.

## Design proposals (`graitools/proposals/`)

A low-barrier scratchpad for design ideas — not the ADR location.

### Before drafting a proposal

- Read what already exists: `ls graitools/proposals/`, then grep `title:` and
  `tags:` across the folder.
- If your idea overlaps or conflicts with an existing proposal, say so
  explicitly in the body and link it via `related:`. Surfacing conflicts is the
  whole point of this area.

### Writing one

- Copy `proposals/TEMPLATE.md` to `proposals/<short-kebab-slug>.md`.
- Filenames are free-form kebab-case slugs — **no sequence numbers**.
- Fill the frontmatter; start at `status: idea`.

### Status lifecycle

`idea → in-review → accepted | rejected`. An `accepted` proposal may later be
`superseded` (record the replacement's slug in the new file's `supersedes:`).

### Promote to an ADR

When an `accepted` proposal is a genuine architectural decision, graduate it to
a proper ADR under `docs/development/ADRs/<subsystem>/` (that record *is*
upstreamable). Then set the proposal's `status: accepted` and point its `adr:`
field at the new ADR path. The proposal stays here as the informal history; the
ADR is the formal record.
