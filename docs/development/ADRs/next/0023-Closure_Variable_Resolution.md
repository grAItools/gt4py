---
tags: [frontend]
---

# Closure Variable Resolution

- **Status**: proposed
- **Authors**: Hannes Vogt (@havogt), drafted with Claude Code
- **Created**: 2026-06-11
- **Updated**: 2026-06-11

This document analyses the limitations of the current closure variable handling in the
`gt4py.next` frontend, defines which kinds of closure references *should* be part of the
GT4Py programming model, compares strategies to support them, and documents the design of
the prototype implemented alongside it.

## Context

Field operators and programs are written as Python functions and may refer to names that
are not parameters or locals — *closure variables*. The frontend captures them with
`inspect.getclosurevars()` into a flat `dict[str, Any]` mapping the **source-level name**
to the Python value (`ffront.source_utils.get_closure_vars_from_function`).

The crucial property of the current design is that *identity in all later stages is the
source-level name*:

- Builtin calls are recognized by comparing `foast.Name.id` against
  `fbuiltins.*_BUILTIN_NAMES` (in `foast_passes.type_deduction` and `foast_to_gtir`).
- Calls to other operators lower to `SymRef(<name used at the call site>)`, while the
  callee registers its IR function definition under its original `def` name
  (`GTCallable.__gt_gtir__()`), linked together at program level in `past_to_itir`.
- Offsets lower to `OffsetLiteral(<name used at the call site>)` and are looked up by
  that name in the `offset_provider` at execution time.
- All closure variables of all transitively referenced operators are merged into a
  single flat namespace (`transform_utils._get_closure_vars_recursively`), which raises
  if the same name is bound to different values anywhere in the call tree.

Any reference whose source-level name differs from the "canonical" name therefore breaks
— each case in a different stage, with a different (often internal) error. Only
attributes of `eve.utils.FrozenNamespace` and `enum` classes were resolved by value
(folded into constants by the former `ClosureVarFolding` pass).

A second, orthogonal property is *when* values are captured: at decoration time (parsing
happens in `FieldOperator.__post_init__`). Embedded execution, in contrast, executes the
original Python function and therefore sees the *live* values. Closing over a mutable
binding consequently has diverging semantics between embedded and compiled execution —
or, for plain scalars, simply fails in compiled mode.

## (a) Catalog of limitations

All cases below were verified empirically on `main` (2026-06). "Stage" names where the
failure surfaced; note how late and how obscure many of the errors are.

### Cases that should work (in scope of the GT4Py model)

| #   | Case                                                                                          | Behavior on `main`                                                        | Stage                |
| --- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | -------------------- |
| 1   | Module-prefixed builtin: `gtx.where(...)`                                                     | `DSLError: Functions can only be called directly.`                        | FOAST type deduction |
| 2   | Aliased builtin: `my_where = where; my_where(...)`                                            | `AssertionError: 'FOAST' expression is not fully typed.` (internal)       | FOAST type deduction |
| 3   | Module-prefixed type constructor: `gtx.float64(1)`                                            | `DSLError: Functions can only be called directly.`                        | FOAST type deduction |
| 4   | Module-prefixed operator call in a field operator: `helpers.helper(a)`                        | `DSLError: Functions can only be called directly.`                        | FOAST type deduction |
| 5   | Aliased operator in a field operator: `from helpers import helper as h2; h2(a)`               | `EveValueError: Symbols {SymbolRef('h2')} not found.` (internal)          | GTIR validation      |
| 6   | Module-prefixed operator call in a program                                                    | `DSLError: Functions must be referenced by their name in function calls.` | PAST parsing         |
| 7   | Aliased operator call in a program                                                            | `RuntimeError: ... symbols resolve to a function with a mismatching name` | PAST linter          |
| 8   | Two same-named operators from different modules, used (under distinct aliases) in one program | `EveValueError: Multiple definitions of symbol 'helper'` (internal)       | GTIR validation      |
| 9   | The same operator referenced under two names                                                  | `EveValueError: Multiple definitions of symbol 'helper'` (internal)       | GTIR validation      |
| 10  | Scalar constant captured from a module attribute: `np.pi`                                     | `AssertionError: Unreachable` (internal)                                  | FOAST→GTIR lowering  |
| 11  | Module-level scalar marked `typing.Final`                                                     | `EveValueError: Symbols {SymbolRef('CONST')} not found.` (internal)       | GTIR validation      |
| 12  | `Dimension` via module prefix (e.g. in `broadcast`)                                           | works (via `NamespaceProxy`)                                              | —                    |
| 13  | `FrozenNamespace` / `enum` attribute constants                                                | works (folded early)                                                      | —                    |

### Cases that should be rejected — but with a decoration/compile-time error, not an internal one

| #   | Case                                                    | Behavior on `main`                                            | Desired behavior                                                                                                                   |
| --- | ------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 14  | Plain (non-`Final`) global scalar in compiled execution | `EveValueError: Symbols ... not found.` at backend            | clear error: mark `Final` or use `FrozenNamespace`. In embedded execution it silently uses the *live* value — diverging semantics. |
| 15  | Nonlocal scalar (factory-function pattern)              | same as 14                                                    | same as 14 (annotations of function locals are not introspectable, so "final-ness" cannot be verified)                             |
| 16  | Closing over a `Field` value                            | `EveValueError: Symbols ... not found.` at backend            | clear error; see stretch goal (c) for how this could eventually be allowed                                                         |
| 17  | Callables in containers: `OPS["h"](a)`                  | `DSLTypeError: Unexpected object 'OPS' of type dict`          | out of model: a closure reference must be a statically resolvable constant expression (name or attribute chain)                    |
| 18  | Plain undecorated Python functions                      | `DSLTypeError: Unexpected object ... of type function`        | out of model (no DSL definition to lower); error message could suggest `@field_operator`                                           |
| 19  | Mutating a captured binding after decoration            | silently ignored in compiled mode, picked up in embedded mode | rejected by the model: only effectively-final bindings may be captured (the divergence becomes impossible)                         |

### The model: what may be captured

A closure reference must be a **constant reference expression**: a name, or a chain of
attribute accesses rooted at a name, that can be resolved at decoration time to a value
of a supported kind:

- a GT4Py builtin (function, type constructor),
- a `GTCallable` (field operator, scan operator, ...),
- a `Dimension` or `FieldOffset`,
- a namespace used as a prefix (module, `FrozenNamespace`, `enum` class),
- a scalar **constant**, where "constant" means the binding is immutable or promised to
  be: an attribute of a `FrozenNamespace`/`enum`, an attribute of a module, or a
  module-level global annotated `typing.Final`.

Identity is defined **by value, not by name**: aliases, re-exports and module-prefixed
references of the same value must be interchangeable. Mutable bindings (plain globals,
nonlocals) and mutable containers are not part of the model; data values (fields, arrays)
are not capturable (see (c) for a possible extension).

## (b) Strategies

### Strategy A — value-based resolution, still early ("resolve names to values at decoration, canonicalize")

Keep the existing capture-at-decoration-time pipeline, but immediately after parsing
resolve every closure reference (including attribute chains) to its Python *value* and
canonicalize the AST:

- constants → folded into `Constant` nodes,
- builtins → rewritten to the canonical builtin name,
- attribute-derived references to operators/dimensions/offsets → rewritten to a
  synthesized, value-derived name registered as an additional closure variable,
- direct references (including aliases) keep their source name; the program lowering
  registers IR function definitions under the *reference name* instead of the definition
  name, making aliases work without rewriting and keeping user-facing names in IR and
  error messages.

* **Pros**: small, local change (two frontend passes + the function registration in
  `past_to_itir`); fixes all catalogued cases; errors move to decoration time with source
  locations; no change to stages, caching, embedded execution, or backends; IR stays
  readable.
* **Cons**: keeps snapshot-at-decoration semantics (mitigated by the `Final` rule);
  constants are baked into the AST (a recompile requires redefining the operator —
  already the case before); the flat merged namespace remains, so two *directly*
  referenced operators with the same source name but different values still collide
  (aliasing is now a documented workaround); the AST is no longer purely syntactic.

### Strategy B — late binding via an explicit closure environment ("linker")

Make the captured environment a first-class object: references in FOAST/PAST point to
opaque, unique reference ids; a `ClosureEnvironment` maps ids to values. Binding happens
at program-assembly time ("link step" in `past_to_itir`): function definitions get final
names (preferring readable ones, mangling only on actual collision) and `SymRef`s are
remapped (`RemapSymbolRefs` exists); constants are substituted at lowering rather than
into the AST.

- **Pros**: fully value-based identity — even the same-source-name collision (case 8
  without aliases) disappears; the AST stays syntactic (better for tooling/printing);
  enables re-binding (`with_closure(...)`), staleness checks, and is the natural
  foundation for capturing fields (stretch goal c) since the environment can carry
  run-time data to be turned into implicit parameters.
- **Cons**: a cross-cutting refactor: stages (`FOASTOperatorDef.closure_vars` and its
  fingerprinting), `GTCallable.__gt_closure_vars__`, both type-deduction passes, both
  lowerings and the DaCe program path all assume the flat name-keyed dict; higher review
  and migration cost; harder to land incrementally.

### Strategy C — minimal name-mangling patches

Keep name-based identity but patch the symptoms: deduplicate identical function
definitions, rename `GTCallable` closure variables to their definition name when
unambiguous, fold module attributes like `FrozenNamespace` attributes.

- **Pros**: smallest diff.
- **Cons**: does not fix the underlying name-identity problem; module-prefixed *calls*
  still need parser/type-deduction changes anyway; ambiguity errors remain late and
  internal; piles up special cases that strategy A subsumes.

### Strategy D — trace-based capture (à la JAX)

Execute the function with tracer values; closures resolve themselves by ordinary Python
semantics.

- **Pros**: closures, aliases, containers, helper functions — everything Python allows —
  resolve for free, always against live values.
- **Cons**: a different frontend architecture, deliberately not chosen for GT4Py
  (ADR 0001): tracing loses source locations and type annotations, restricts control flow
  (no data-dependent `if` on fields without special handling), and would change the
  semantics of the existing source-based dialects. Out of scope.

### Decision

**Strategy A now, with B as the evolution path.** A resolves every catalogued limitation
except the no-alias same-name collision (8'), is implementable and reviewable
incrementally, and does not paint us into a corner: the canonicalization pass is exactly
the place where strategy B's reference ids would be introduced later, and the
key-based function registration in `past_to_itir` is the seed of B's link step. The
prototype on this branch implements A.

## Prototype (implemented on this branch)

- `ffront.foast_passes.closure_var_resolution.ClosureVarResolution` replaces
  `ClosureVarFolding`: resolves attribute chains through modules/namespaces by value,
  folds constants (including `np.pi`-style module attributes and `typing.Final` globals),
  canonicalizes builtin references, and synthesizes value-derived names
  (`transform_utils.mangled_function_name`) for attribute-derived operator / dimension /
  offset references. Unresolvable-but-embedded-valid references (non-`Final` scalars,
  fields) are left untouched.
- `ffront.past_passes.closure_var_canonicalization.ClosureVarCanonicalization` is the
  PAST counterpart; `past.Call.func` now also accepts attribute expressions
  (`program_ast`, `func_to_past`).
- `past_to_itir.past_to_gtir` registers each `GTCallable`'s IR function definition under
  the name it is *referenced* by (the closure-variable key) instead of the definition
  name; the same callable may be registered under several names. The now-impossible
  "misnamed function" lint was removed.
- `foast_to_gtir` rejects remaining data-typed closure variables (non-`Final` scalars,
  fields, tuples) with an explanatory error at lowering time — embedded execution is
  unaffected.
- `source_utils.get_final_closure_var_names` detects `typing.Final` annotations of
  module-level globals; `dialect_parser` threads them into the parsers.

Known gaps / follow-ups:

- Case 8 without aliases (two *directly* referenced same-named operators) still raises;
  the error message now suggests aliasing as the workaround. Fully solving it requires
  strategy B's link step.
- Local variables shadowing a canonical builtin name (e.g. a local named `where` together
  with `gtx.where`) are detected only when the local is visible in the symbol table.
- `typing.Final` cannot be verified for nonlocals (function-local annotations are not
  retained at runtime); the factory pattern therefore requires a `FrozenNamespace` (or a
  future explicit opt-in, e.g. `gtx.constant(...)`).
- The `compile`d-program fingerprint already contained the closure values; folded
  constants do not change cache-correctness.

## (c) Stretch goal: closing over read-only fields

Capturing field *data* is different in kind from everything above: the value cannot be
folded into the IR, it must reach the backend at execution time. A clean design needs to
answer three questions:

1. **Marking.** Capture must be explicit, since the field's buffer lifetime and mutation
   become part of the operator's contract. Options: require the binding to be annotated
   `typing.Final[gtx.Field[...]]`; or an explicit wrapper `gtx.constant_field(field)`
   (preferred: it is introspectable at runtime, can freeze/copy-on-capture the buffer,
   and can expose `__gt_type__`). Plain field bindings stay rejected (as in the
   prototype).

2. **Mechanism: implicit parameters.** During FOAST canonicalization, a captured
   read-only field becomes a *hidden parameter* of the operator (type from the value);
   `GTCallable` grows a `__gt_implicit_args__() -> dict[str, Field]` so that callers —
   transitively up to the program — append the captured values to the argument list at
   call/compile time. This reuses the entire existing argument machinery (descriptors,
   static domains, allocation checks) instead of inventing a second data path. Embedded
   execution needs no change at all (Python closure semantics already do this), which
   keeps both modes consistent *if* binding is by reference.

3. **Binding time and caching.** Bind the *binding*, not the buffer: at each call the
   current value of the captured wrapper is passed, so rebinding
   (`op.with_closure(field=...)` or reassigning through the wrapper) behaves like
   embedded execution. The compiled-program cache key must include only the field's
   *type/domain descriptor* (as for normal arguments), not the buffer identity —
   captured fields then have zero recompilation cost. Mutation through GT4Py is prevented
   by the wrapper (read-only views); mutation of the underlying buffer from outside
   remains observable, same as for ordinary arguments.

   An alternative — freezing the buffer into the compiled program as a true constant
   (e.g. baked into generated code or a DaCe constant) — is attractive for lookup tables
   (enables constant-folding in the backend) but couples cache identity to buffer
   *content*; it should be a separate opt-in (`gtx.constant_field(field, freeze=True)`)
   layered on `StaticArg`-style argument descriptors (ADR 0021).

This fits strategy A (the canonicalization pass is where the hidden parameter is
introduced) but becomes more natural with strategy B, where the closure environment
already travels with the stages.

## References

- ADR 0001 — Field View Frontend Design
- ADR 0021 — Argument Descriptors (mechanism for the `freeze=True` variant of (c))
- Prototype: this branch (`claude/elegant-cannon-13ew06`)
