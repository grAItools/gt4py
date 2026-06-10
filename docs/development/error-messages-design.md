# User-Friendly DSL Error Messages — Design Document

**Status:** prototype implemented (see "Prototype" below) · **Scope:** `gt4py.next`
**Related:** research notes on rustc/Elm-style diagnostics; Shape-Up cycle
"Enhancements to error reporting"; issue #1031 (missing AST locations).

## 1. Problem

GT4Py embeds a DSL in Python: users write what looks like Python, but only a
subset of it is accepted, and the semantics (whole-field operations, no
implicit type conversion) differ from NumPy. The errors a newcomer hits first
are therefore not bugs in their algorithm but collisions with the DSL's rules —
exactly the place where Elm's "syntax cliff" research showed users give up.

Before this prototype, a typical first-day session looked like this:

```
Undeclared symbol 'tmp_feild'.                      ← no suggestion
  File "/tmp/demo.py", line 11
        return tmp_feild
               ^^^^^^^^^

Unsupported Python syntax: 'ast.While'.             ← leaks compiler internals
  File "/tmp/demo.py", line 21
        while True:
                                                    ← broken multi-line snippet
            a = a + 1.0
        ^^^^^^^^^^^^^^^                             ← carets on the wrong line
```

The three failures: no *fix suggestion*, internal jargon (`ast.While`), and a
renderer that mishandles multi-line spans. None of these require new
infrastructure to fix — GT4Py already threads `eve.SourceLocation` through
FOAST/PAST nodes and has a central `DSLError(location, message)` — they
require the error *data model* to carry more than a string, and call sites to
use it.

## 2. Design principles (from the research)

Applied from the rustc/Elm/Guppy/Dynamo playbook, adapted to a Python-embedded
DSL:

1. **Diagnostics are structured data, not strings.** A message plus a primary
   span, optional caret label, secondary labeled spans, notes, hints, and a
   stable code. Rendering is a separate, single-owner concern.
2. **Point at the smallest offending span; explain contributing spans.**
   "this has type 'Field\[[IDim], bool\]'" under the operand, not a paragraph.
3. **Say what to do, not only what is wrong.** Every common error carries a
   `Hint:` naming the supported alternative (`where(...)`, `astype(...)`,
   `scan_operator`, `&`/`|`).
4. **Never leak compiler internals into the headline.** `ast.While` becomes
   "'while' loop"; internal names appear only as a fallback for constructs we
   haven't catalogued.
5. **Meet Python users where they are.** We keep the CPython-style
   `File "...", line N` header (terminals and IDEs linkify it) and add a
   rustc-style line-number gutter underneath, rather than switching to
   `--> file:line:col` wholesale.
6. **Message quality is regression-tested.** Bad programs with assertions on
   the rendered text (rustc UI-test pattern) live in
   `tests/next_tests/unit_tests/ffront_tests/test_diagnostic_messages.py`.

## 3. Architecture

### 3.1 Data model (`gt4py.next.errors.exceptions`)

`DSLError` is extended in place — no parallel hierarchy, fully
backward-compatible (`DSLError(location, message)` still works):

```python
class DSLError(GT4PyError):
    code: ClassVar[Optional[str]] = None  # stable id, set per subclass

    location: Optional[SourceLocation]  # primary span
    label: Optional[str]  # text after the carets
    related: list[tuple[SourceLocation, str]]  # secondary labeled spans
    notes: list[str]  # "Note: ..." — why it's an error
    hints: list[str]  # "Hint: ..." — what to do instead
```

The split between `notes` and `hints` is deliberate (mirrors rustc's
`note`/`help`): notes state facts ("GT4Py does not implicitly convert between
datatypes."), hints give commands ("Convert one operand explicitly, e.g.
'astype(<expr>, float64)'."). `code` is a class-level slug
(`"undefined-symbol"`, `"unsupported-syntax"`) rather than an instance string:
the category is a property of the error class, and a registry of codes can
later map to documentation pages (the Dynamo graph-break-site /
`rustc --explain` pattern) without touching call sites.

Subclasses encapsulate message construction so call sites stay declarative:
`UndefinedSymbolError(loc, name, candidates=...)` computes the
"Did you mean ...?" hint itself (via `difflib.get_close_matches`); call sites
only supply the candidate set they have at hand.

### 3.2 Rendering (`gt4py.next.errors.formatting`)

One function, `format_diagnostic_parts`, owns the on-screen shape; both
`DSLError.__str__` and the excepthook delegate to it. Rendering rules:

- Line-number gutter (`11 | ...`), carets under the first line of the span,
  optional label after the carets.
- Multi-line spans render at most 3 source lines, then a `...` gutter row
  (fixes the pre-existing doubled-blank-line bug for multi-line locations).
- A `related` span on the *same line* as the primary span merges into the
  primary snippet as a `-` underline row — the common case for binary
  operators. Spans on other lines (or files) get their own indented
  `Related:` block with a full snippet.
- `Note:`/`Hint:` lines wrap at 88 columns with hanging indent.
- If source text is unavailable (REPL, notebook cell GC'd), everything
  degrades to the plain `File "...", line N` header — never a crash, never a
  wrong snippet.

The renderer is deliberately in-house and small (~100 lines): no mature
"miette for Python" exists, and Guppy's alternative (PyO3 bindings to Rust's
miette) is a heavy dependency for what is, at this stage, string formatting.
If GT4Py later wants IDE/LSP output, the structured fields on `DSLError` are
already the JSON-serializable interface; only an emitter would be added.

### 3.3 The unsupported-subset catalogue (`ffront.dialect_parser`)

The "error productions" idea from parser theory, adapted to subset rejection:
a single table maps rejected `ast` node types to a user-facing name and hints:

```python
_UNSUPPORTED_FEATURE_HINTS: dict[type[ast.AST], tuple[str, tuple[str, ...]]] = {
    ast.While: ("'while' loop",
        ("GT4Py functions describe operations on whole fields without explicit "
         "loops. For sequential dependencies along a dimension, use a 'scan_operator'.",)),
    ast.Lambda: ("'lambda' expression",
        ("Define a separate function decorated with '@field_operator' instead.",)),
    ...
}
```

`DialectParser.generic_visit` consults the table; unlisted constructs fall
back to the qualified `ast` class name, so the table can grow incrementally
and nothing regresses when CPython adds node types. Every entry also gets the
uniform note "Only a subset of Python is valid inside GT4Py functions."

This is the maintainability core: **improving the message for one more
construct is one dict entry plus one UI test** — no renderer or exception
changes.

### 3.4 Call-site upgrades (`foast_passes.type_deduction`, `func_to_foast`)

Pass-level errors attach structure where the pass has it:

- `visit_Name`: undeclared symbols raise `UndefinedSymbolError` with the
  symbol table as candidate set. SSA-versioned internal names
  (`tmpᐞ0`) are mapped back to user spelling via
  `single_static_assign.original_name` before they reach the user.
- `_deduce_binop_type`: the incompatible operand gets the primary span and a
  type label; the *other* operand becomes a `related` span ("the other operand
  has type '...'"). Boolean-field arithmetic — the classic NumPy-porting
  mistake — additionally hints `where(mask, a, b)` / `astype`.
- dtype promotion failures (e.g. `float32 + float64`) label both operands and
  state the no-implicit-conversion rule as a note.
- `visit_BoolOp` (`and`/`or` on fields) explains *why* (fields hold one
  boolean per grid point) and hints `&`/`|`.

## 4. Prototype: before / after

All examples are real output of the prototype (`str(err)`; the excepthook
prints the same body plus the exception's qualified name).

**Typo in a variable name** — the most common error of all:

```
Undeclared symbol 'tmp_feild'.
  File "/tmp/demo_errors.py", line 11
    11 |         return tmp_feild
       |                ^^^^^^^^^ not defined at this point
  Hint: Did you mean 'tmp_field'?
```

**Unsupported construct** (was: `Unsupported Python syntax: 'ast.While'.` with
broken carets):

```
Unsupported Python syntax: 'while' loop.
  File "/tmp/demo_errors.py", line 21
    21 |         while True:
       |         ^^^^^^^^^^^
    22 |             a = a + 1.0
  Note: Only a subset of Python is valid inside GT4Py functions.
  Hint: GT4Py functions describe operations on whole fields without explicit loops. For
    sequential dependencies along a dimension, use a 'scan_operator'.
```

**Arithmetic with a boolean mask** (was: message only, caret on one operand):

```
Unsupported operand type(s) for +: 'Field[[IDim], float64]' and 'Field[[IDim], bool]'.
  File "/tmp/demo_errors.py", line 44
    44 |         return a + mask
       |                    ^^^^ '+' expects arithmetic operands, but this has type 'Field[[IDim], bool]'
       |                - the other operand has type 'Field[[IDim], float64]'
  Hint: To select values based on a boolean mask, use 'where(mask, a, b)'. To compute
    with a boolean field, convert it explicitly, e.g. 'astype(mask, int32)'.
```

**Mixed precision** (`float32 + float64`):

```
Could not promote 'Field[[IDim], float32]' and 'Field[[IDim], float64]' to common type in call to '+'.
  File "/tmp/demo_errors.py", line 55
    55 |         return a + b
       |                ^^^^^
       |                - this operand has type 'Field[[IDim], float32]'
       |                    - this operand has type 'Field[[IDim], float64]'
  Note: GT4Py does not implicitly convert between datatypes.
  Hint: Convert one operand explicitly, e.g. 'astype(<expr>, float64)', or make the
    datatypes of the inputs match.
```

**`and`/`or` on fields**:

```
Unsupported Python syntax: logical operators `and`, `or`.
  File "/tmp/demo_errors.py", line 65
    65 |         return a and b
       |                ^^^^^^^
  Note: `and` and `or` operate on whole truth values, but fields contain one boolean per
    grid point.
  Hint: Use the element-wise operators '&' and '|' instead.
```

## 5. How to add or improve a diagnostic (contributor guide)

1. **A newly rejected Python construct** → add one entry to
   `_UNSUPPORTED_FEATURE_HINTS` in `ffront/dialect_parser.py`. Name the
   construct as the user spells it (`"'match' statement"`), and give one hint
   naming the supported alternative.
2. **A richer message at an existing `raise errors.DSLError(...)`** → add
   `label=`, `related=`, `notes=`, `hints=` keyword arguments. Do not encode
   that information into the message string.
3. **A new error category** → subclass `DSLError`, set `code`, build the
   message and hints in `__init__` from semantic arguments (like
   `UndefinedSymbolError(loc, name, candidates=...)`), and export it from
   `errors/__init__.py`.
4. **Always** add a UI test in `test_diagnostic_messages.py` asserting on the
   rendered text. Tests are the spec; this is what keeps message quality from
   regressing.

Style (per `CODING_GUIDELINES.md`): messages, notes, and hints are sentences —
capital first letter, trailing period; code objects in single quotes. Labels
are sentence fragments (they continue the caret line) and have no trailing
period.

## 6. Alternatives considered

- **Adopt/bind an external renderer (miette via PyO3, like Quantinuum
  Guppy).** Rejected for now: adds a Rust build dependency for ~100 lines of
  formatting; revisit only if diagnostics gain features (multi-file spans,
  syntax highlighting) where parity with miette matters.
- **A parallel `Diagnostic` dataclass next to the exception hierarchy.**
  Rejected: GT4Py's errors *are* exceptions and flow through `raise`; a second
  representation would need conversion at every boundary. Extending `DSLError`
  keeps one type with both roles (rustc's `Diag` is similarly both).
- **rustc-style `--> file:line:col` headers.** Rejected in favor of Python's
  `File "...", line N`, which existing tools linkify and Python users parse on
  sight.
- **Numeric error codes (`GT0042`).** Deferred; slugs (`undefined-symbol`)
  are self-describing and don't need a registry to stay collision-free. If
  documentation pages per error land (Stage 3 below), slugs map directly to
  URL anchors.

## 7. Roadmap

The prototype is Stage 1 of the staged plan; the stages are independent.

- **Stage 0 — harden locations (parallel, ongoing).** Audit that every
  FOAST/PAST/IR node carries a populated location; replace the ~50
  user-reachable `AssertionError`s with `DSLError` or an internal-error type
  ("Please submit a bug report"). Benchmark: no `DSLError` without a usable
  location, no user-triggerable bare `AssertionError`.
- **Stage 1 — this prototype.** Structured `DSLError` + renderer + catalogue +
  the five highest-frequency errors. Extend coverage call site by call site
  (the ~40 remaining `raise errors.DSLError` sites in `type_deduction.py` are
  the backlog, prioritized by how often users hit them).
- **Stage 2 — explain pages.** Generate a documentation page per `code` with a
  triggering example and fix; append "see <docs-url>/errors/#undefined-symbol"
  as a final note when a page exists. Requires only a code→URL map in the
  renderer.
- **Stage 3 — lowering-stage attribution.** Errors raised past the frontend
  (GTIR transforms, backends) should re-attach the best-known frontend
  location at pass boundaries instead of surfacing internal tracebacks; the
  existing excepthook + `config.VERBOSE_EXCEPTIONS` already provide the
  JAX-style filtered/full toggle.
- **Stage 4 — presentation polish.** Color (gated on TTY and `NO_COLOR`),
  UTF-8-byte→character column conversion for non-ASCII sources, tab-aware
  caret alignment, and a JSON emitter for IDE/LSP integration — all contained
  in `formatting.py` by construction.

## 8. Open questions

- Should `notes`/`hints` ride on Python 3.11+ `BaseException.__notes__` so
  plain tracebacks (without the excepthook) also show them? Today they only
  appear via `str(err)`/excepthook. (Blocked on dropping 3.10.)
- Candidate sets for "did you mean": currently the local symbol table;
  including `fbuiltins` names (e.g. `minimun` → `minimum`) would catch another
  frequent class of typos but needs care to not suggest reserved names where
  they are invalid.
- The same-line `related` merge handles the binary-operator case; true
  multi-label single-row rendering (rustc's stacked labels) is more work and
  probably not worth it until a real case demands it.
