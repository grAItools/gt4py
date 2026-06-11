---
tags: [backend, otf, workflows, toolchain, proposal]
---

# Proposal: Redesigning the OTF Compilation Pipelines

- **Status**: draft / request for comments
- **Created**: 2026-06-11
- **Companion document**:
  [Prior Art: Compilation Pipelines in Python-Embedded Compiler Frameworks](otf-pipeline-redesign-prior-art.md)
- **Supersedes (if accepted)**: parts of
  [ADR 0011](../ADRs/next/0011-On_The_Fly_Compilation.md),
  [ADR 0016](../ADRs/next/0016-Multiple-Backends-and-Build-Systems.md),
  [ADR 0017](../ADRs/next/0017-Toolchain-Configuration.md)

## 1. Summary

GT4Py's compiled backends are driven by the workflows of `gt4py.next.otf`: a
generic, statically-typed function-composition framework
(`Workflow[StartT, EndT]`, `NamedStepSequence`, `CachedStep`, …) through which
a program definition is piped from DSL AST to a compiled Python extension.
The design (ADR 0011, 2022) achieved its main goal — backends share the
binding, build-system, and import stages instead of being monoliths — but a
review of the code and of how every backend actually uses the framework shows
that the abstractions have not held up: the promised static type safety is
largely illusory, configuration goes through a second framework
(`factory_boy`) with stringly-typed override paths, caching is spread over
four uncoordinated layers with three different key derivations, and there is
no sanctioned way to inspect or re-run intermediate stages — consumers resort
to duck-typed unwrapping of workflow internals.

We surveyed how JAX, PyTorch 2, MLIR, TVM, Numba, DaCe, and Triton organize
the same problem (details in the [companion document](otf-pipeline-redesign-prior-art.md)).
The consistent lesson: successful systems expose a *small, fixed set of named
stages as first-class objects* (JAX's `trace → lower → compile`), extend via
*narrow registries* rather than recomposable generic pipelines, key caches
*once* on a canonical artifact hash, and treat *inspectability as a feature
of the stage boundary*, not as something bolted on.

This document proposes both:

- **Incremental improvements** (§6) that remove the worst friction within the
  current architecture, each independently landable; and
- **A redesign** (§7): replace the generic workflow framework with an
  explicit *staged compilation API* — typed stage objects
  (`Program → LoweredProgram → CompiledArtifact → Executable`), a backend
  protocol with two methods instead of a four-slot recipe, one fingerprint
  function, one artifact store, and instrumentation hooks at every stage
  boundary.

## 2. Scope and method

In scope: `src/gt4py/next/otf/` (workflow framework, stages, arguments,
compilation, caching, `CompiledProgramsPool`), its consumers in
`src/gt4py/next/backend.py` and `src/gt4py/next/program_processors/runners/`
(gtfn, dace, roundtrip), and the configuration pattern of ADR 0017. Out of
scope: the IRs themselves (FOAST/PAST/GTIR), the transformations inside the
translation steps, and `gt4py.cartesian` (referenced only for comparison).

The analysis is based on a full read of the otf package and its consumers,
the relevant ADRs (0009, 0011, 0012, 0016, 0017, 0021), and a survey of
seven external frameworks. Every weakness claim below cites the code.

## 3. The current architecture

### 3.1 The two pipelines

There are really two pipelines glued together by `backend.Backend`
(`backend.py:147`):

```
                         (1) frontend transforms                (2) "executor" / OTF workflow
DSL def + args ──► Transforms (MultiWorkflow) ──► GTIR + CompileTimeArgs ──► OTFCompileWorkflow ──► callable
                   func_to_foast / foast_to_past /                           translation
                   past_lint / past_to_itir, …                               bindings
                   (step order chosen by isinstance                          compilation
                    dispatch on the input)                                   decoration
```

Above both sits the dispatch layer: `decorator.Program.__call__` delegates to
`CompiledProgramsPool` (`otf/compiled_program.py:319`), which manages
per-variant compiled programs (static arguments, offset providers, async
compilation) and calls `Backend.compile()` on a miss.

### 3.2 The framework pieces

- `otf/workflow.py` — generic combinators: the `Workflow[StartT, EndT]`
  protocol (a single-argument callable), `StepSequence` /
  `ChainableWorkflowMixin` (`.chain()`), `NamedStepSequence` (steps =
  dataclass fields, order discovered by reflection over type hints),
  `MultiWorkflow` (input-dependent step order), `CachedStep`,
  `SkippableStep`, `.replace()`.
- `otf/toolchain.py` — `ConcreteArtifact[DefT, ArgsT]` (a data+args pair) and
  three adapters (`DataOnlyAdapter`, `ArgsOnlyAdapter`, `StripArgsAdapter`)
  needed because workflows take exactly one argument.
- `otf/stages.py`, `otf/definitions.py` — the stage vocabulary:
  `CompilableProgramDef` (GTIR + `CompileTimeArgs`) → `ProgramSource` →
  `CompilableProject` → `ExecutableProgram`, plus protocols for the steps
  between them.
- `otf/recipes.py` — `OTFCompileWorkflow`, the canonical four-slot
  `NamedStepSequence`: `translation`, `bindings`, `compilation`,
  `decoration`.
- `otf/compilation/` — shared build tail: build-system projects
  (CMake/compiledb), a file cache keyed on a hash of the generated source,
  `BuildData` JSON progress tracking, locking, importer.
- `otf/arguments.py`, `otf/compiled_program.py` — compile-time argument
  modeling (`CompileTimeArgs`, `ArgStaticDescriptor`) and the
  `CompiledProgramsPool` with code-generated cache-key extractors.
- Backend assembly — per-backend `factory_boy` factories
  (`GTFNBackendFactory`, `DaCeBackendFactory`) that wire the four slots and
  expose configuration as factory params and traits.

## 4. Strengths

The current design got important things right, and any redesign should
preserve them:

1. **A shared stage vocabulary.** `ProgramSource` / `CompilableProject` /
   `ExecutableProgram` give backends a common language, and the boundaries
   are in the right places: "generated source", "buildable project",
   "imported callable" are exactly the artifact types that Triton, DaCe, and
   PyTorch also use as stage boundaries. The DaCe and GTFN backends genuinely
   share the nanobind/CMake/compiledb/import tail — the original goal of ADR
   0011 (escaping cartesian's per-backend monoliths) was achieved.
2. **The build tail is solid engineering.** Build-progress tracking via
   `BuildData` JSON, inter-process locking for concurrent MPI ranks
   (`compilation/compiler.py:73`), session vs. persistent cache lifetimes,
   and the compiledb optimization (reusing CMake configuration across
   programs) are all features the surveyed frameworks also converged on.
3. **`CompiledProgramsPool` is ahead of most prior art.** Per-variant
   compilation keyed on static argument *values* (descriptor mechanism, ADR
   0021), asynchronous background compilation, code-generated hot-path cache
   keys, and an explicit AOT `compile()` API are features PyTorch and JAX
   acquired only late (guards, AOT API) — GT4Py has them, just under a layer
   of `eval`-based metaprogramming that obscures them.
4. **Steps are pure and individually testable.** Most steps are frozen
   dataclasses mapping immutable inputs to immutable outputs; unit tests for
   the combinators are simple.
5. **Decisions are documented.** ADRs 0009–0021 record the rationale; this
   proposal can engage with explicit "Revise If" clauses rather than
   reverse-engineering intent.

## 5. Weaknesses

### 5.1 The type-safety promise does not hold

ADR 0011's central design decision was "Statically Typed Workflows". In
practice:

- `NamedStepSequence.__call__` is `getattr`-in-a-loop with `Any`
  intermediates (`workflow.py:135-140`); nothing checks step N's output type
  against step N+1's input type — neither statically (mypy sees field types
  individually, not the composition) nor at runtime. The ADR itself concedes
  this: "making sure that the steps are compatible in the order specified is
  up to the implementer".
- Step discovery is reflection: a dataclass field is a step iff its *type
  annotation* passes `issubclass(field_type, Workflow)` against a
  `runtime_checkable` protocol (`workflow.py:150-157`) — i.e. structurally,
  anything annotated as callable. Whether something is a pipeline stage is
  decided by a typing detail invisible at the definition site.
- The combinators themselves need escape hatches: `typing.cast` in
  `StepSequence.chain` (`workflow.py:216`), `# type: ignore[return-value]`
  in `SkippableStep` ("up to the implementer to make sure StartT == EndT",
  `workflow.py:279`), `# type: ignore[assignment]` on `CachedStep`'s default
  hash (`workflow.py:257`). Seven `type: ignore`s in otf + backends are the
  price of generics that don't fit.
- `MultiWorkflow.step_order` (`backend.py:98-137`) chooses the step list by
  `match` on the runtime input type — the static input/output types of the
  whole workflow are a fiction.

The framework pays the full complexity cost of generic typed composition
(`StartT`, `StartT_contra`, `EndT`, `EndT_co`, `NewEndT`, `IntermediateT`,
`HashT`, `DataT`, `ArgT` — `workflow.py:22-30`) while delivering, in effect,
runtime duck typing. Numba's "expert-only" untyped `state` pipeline (see
[prior art §5](otf-pipeline-redesign-prior-art.md#5-numba)) is the honest
version of the same trade-off.

### 5.2 The single-argument constraint forces artifact gymnastics

Because a `Workflow` takes exactly one input, the program definition and its
arguments must travel as one value, producing `ConcreteArtifact[DefT, ArgsT]`
plus three adapter combinators whose only job is to route a step to the
`.data` or `.args` half (`toolchain.py:30-66`), and factory wrappers like
`adapted_jit_to_aot_args_factory` (`backend.py:36-41`) for each of them. The
"Pipeline Architecture" that ADR 0011 explicitly rejected — one data type
with many states flowing through everything — has effectively crept back in
through this side door, just with generics on top.

### 5.3 Configuration through `factory_boy` is the wrong tool

Backends are assembled via `factory_boy` — an ORM *test-fixture* library —
with parameters addressed by double-underscore paths:

```python
# program_processors/runners/dace/workflow/backend.py:126-136
return DaCeBackendFactory(
    gpu=gpu,
    cached=cached,
    otf_workflow__cached_translation=cached,
    otf_workflow__bare_translation__async_sdfg_call=...,
    otf_workflow__bare_translation__auto_optimize_args=optimization_args,
    ...
)
```

Consequences:

- Override paths are strings resolved at runtime; a typo
  (`otf_workflow__bare_translaton__...`) is silently accepted as a new
  parameter or fails far from the call site. mypy cannot check any of it,
  and `factory_boy`'s own annotations force `# type: ignore[assignment]` at
  every `LazyFunction`/`LazyAttribute` (`gtfn.py:115`, `dace/workflow/factory.py:42`).
- The factory graph (`Params`, `Trait`, `SubFactory`, `SelfAttribute("..x")`)
  is a second program that *describes* the pipeline, with its own evaluation
  order, that contributors must learn in addition to the pipeline itself.
  The `cached_translation` trait is duplicated verbatim in the GTFN and DaCe
  workflow factories (`gtfn.py:122-132`, `dace/workflow/factory.py:46-59`).
- Defaults bind to `gt4py.next.config` at import time (ADR 0017 documents
  this and its known inconsistency and testability problems as accepted
  costs of a "minimal implementation").

None of the surveyed frameworks configures pipelines this way. The pattern
they converge on is *plain constructor arguments for stage-local options +
a scoped context object for cross-cutting options* (TVM `PassContext`, MLIR
pass options, `jax.jit(...)` kwargs; prior art §9.3).

### 5.4 No sanctioned access to intermediate stages

There is no API to ask "give me the GTIR / generated C++ / SDFG for this
program with this backend". The cost is visible wherever someone needed it:

- `dace/program.py:79-99` casts `self.backend.executor` to
  `OTFCompileWorkflow`, peels off a possible `CachedStep` by
  `hasattr(executor, "step")` (twice — the comment admits "we don't know if
  the compile workflow is cached"), then `.replace()`s private translation
  flags and re-runs the step to extract an SDFG. It also mutates a frozen
  `CompileTimeArgs` via `object.__setattr__` (`dace/program.py:62-72`).
- The roundtrip backend's debugging story is a `debug` flag that prints a
  temp-file path; gtfn's is "know where the build cache is".
- Nothing exists like MLIR's `-mlir-print-ir-after-all`, Triton's
  `kernel.asm['ttgir']`, or JAX's `lowered.as_text()` (prior art §9.5).

ADR 0011's stated goal "keep components easy to reason about" is defeated at
exactly the moment a human needs to reason about a concrete compilation.

### 5.5 Caching: four layers, three key functions, no shared design

| Layer                 | Where                                                          | Key                                                                                                        | Store                 |
| --------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | --------------------- |
| Backend executor memo | `CachedStep` around the whole OTF workflow (`gtfn.py:168-172`) | `stages.compilation_hash`: hash of GTIR + content-hash of arg specs + `id()`-based offset-provider hash    | in-memory dict        |
| Translation cache     | `CachedStep` around translation (`gtfn.py:122-132`)            | `stages.fingerprint_compilable_program`: content hash of GTIR fingerprint + offset providers + column axis | `FileCache` on disk   |
| Build cache           | `compilation/cache.py:45-72`                                   | SHA-256 of *generated source* + entry-point + deps                                                         | per-program build dir |
| Variant pool          | `CompiledProgramsPool` (`compiled_program.py:412-416`)         | code-generated tuple of static arg values + `id()`-based offset-provider hash + specialization hash        | in-memory dict        |

Problems beyond sheer multiplicity:

- **Keys are derived independently at each layer** — the precise failure
  mode that produced JAX's cross-layer cache-poisoning bug
  ([PR #24828](https://github.com/jax-ml/jax/pull/24828); prior art §1.3).
- **The build cache key is computed from the *output* of translation**, so
  even a fully warm disk cache must re-run the entire frontend + translation
  (typically the slowest pure-Python part) just to find the cache folder.
  The translation `FileCache` mitigates this for two backends, as an opt-in
  trait, with a different key function.
- **`id()`-based hashes** of offset providers
  (`common.hash_offset_provider_items_by_id`) are fast but tie cache
  validity to object identity — correct only under conventions that are
  nowhere enforced, and they leak into the ordering assumptions of the gtfn
  bindings (`gtfn.py:91-94` "Any modification to the hashing of offset
  providers may break this assumption").
- **No diagnostics.** Nothing like JAX's `jax_explain_cache_misses` exists;
  a user observing recompilation has no tool to learn which layer missed or
  why.
- **The caching design contradicts the ADR**: ADR 0011 decided "caching is
  … the concern of other steps (currently only the build step)"; reality
  added three more layers without revisiting the decision.

### 5.6 Backend extension is wide, not narrow

To add a compiled backend today one must: implement a translation step
(reasonable), pick/write a bindings step, wire a `Compiler` with a
builder factory, write a decoration step (a bare `functools.partial`, which
silently loses `replace()`-ability and factory configurability —
`gtfn.py:149-151`), then write *two* `factory_boy` factories
(workflow + backend) duplicating the caching traits, naming logic
(`name_device`, `name_cached`, `name_postfix` string assembly), and
device/allocator switching of the existing ones. The gtfn and dace factory
pairs differ by little besides class names — ~150 lines of copied wiring
each. Compare PyTorch's backend contract — one callable, one registration
call (prior art §2) — or Triton's `add_stages`.

Meanwhile backends that *don't* fit the four-slot recipe (roundtrip — no
bindings/build; embedded execution — no compilation at all) simply bypass
`OTFCompileWorkflow`, so the "uniform" recipe describes exactly two of the
five execution paths.

### 5.7 Smaller frictions (evidence in code)

- **Naming drift**: ADR 0011's names (`ProgramCall`, `CompilableSource`,
  `CompiledProgram`, `otf.step_types`) no longer match the code
  (`CompilableProgramDef`, `CompilableProject`, `ExecutableProgram`,
  `otf.definitions`); docstrings still use the old names
  (`definitions.py:40,47,64`). `Backend.executor` is not an executor; the
  rename TODO sits in `backend.py:143-146`.
- **`CompileTimeArgs` is not compile-time**: it carries a *runtime*
  `OffsetProvider` because a GTIR pass needs the tables
  (`arguments.py:148` TODO; circumvented by frozen-dataclass mutation in
  `dace/program.py:67-72`); `column_axis` is threaded through but unused
  (`compiled_program.py:627` TODO).
- **`eval`/`exec` metaprogramming** for descriptor extraction and cache keys
  (`arguments.py:64-66,279-280`, `compiled_program.py:244,291,501`) is
  performance-justified on the hot path but used even in cold paths, and
  makes the argument-descriptor machinery the hardest-to-follow code in the
  package.
- **Test coverage pins the combinators, not the contracts**: there are unit
  tests for `StepSequence`/`CachedStep` toys, but none for stage-boundary
  compatibility, `replace()` patterns, or cache-key stability.

## 6. Incremental improvements

These are worth doing even if the redesign (§7) is rejected, and none of
them blocks it. Ordered by value/effort.

1. **Single fingerprint function** *(small, high value)*. Define one
   canonical `fingerprint(program_def) -> str` (content-based, covering GTIR
   fingerprint, arg types, offset-provider *types*, backend-relevant config,
   gt4py version) and use it for the translation cache, the executor
   `CachedStep`, and — via a `fingerprint → build dir` index — the build
   cache, making warm starts skip translation entirely. Replace `id()`-based
   offset-provider hashing with content hashing of the type + an identity
   fast path. (Triton/JAX key model; prior art §§1.3, 7.)
2. **`inspect`/dump API + stage instrumentation** *(small, high value)*.
   Add an optional observer to `NamedStepSequence.__call__`/`MultiWorkflow`
   (`on_stage(name, artifact)`) plus a `GT4PY_DUMP_STAGES=<dir>` config that
   writes each artifact (GTIR text, generated source, SDFG JSON) per
   program, MLIR-`print-ir-tree-dir`-style. Expose
   `Backend.lower(definition, compile_time_args) -> stage artifacts` as a
   supported method so `dace/program.py` can stop unwrapping workflow
   internals.
3. **`explain_cache_misses` logging** *(small)*. Each cache layer logs key,
   hit/miss, and the first differing key component under a debug flag.
4. **Replace `factory_boy` with plain code** *(medium, high value)*. A
   backend becomes a frozen dataclass / `make_*_backend()` function with
   keyword arguments — the `make_dace_backend` wrapper already proves the
   need; this removes the trait duplication, the `__`-path strings, and the
   `type: ignore`s. Deduplicate the shared "cached translation", naming, and
   device wiring into one helper used by gtfn and dace. (Supersedes part of
   ADR 0017; config remains in `gt4py.next.config`.)
5. **Honest typing for the combinators** *(small)*. Drop the unused
   combinator generality (`SkippableStep`, `StepSequence.chain` chains are
   barely used outside tests); document `NamedStepSequence` as runtime-typed;
   declare step order explicitly (a class-level tuple) instead of
   type-annotation reflection.
6. **Fix the argument model edges** *(medium)*. Split `CompileTimeArgs` into
   a true compile-time part (types, descriptors, offset-provider *types*)
   and an explicitly-named bridge for the passes that still need runtime
   tables; remove dead `column_axis`; restrict `eval`/`exec` codegen to the
   measured hot path (`_argument_descriptor_cache_key_from_args`) and use
   ordinary code elsewhere.
7. **Align names with the ADR or the ADR with the names** *(small)*. One
   rename pass (`Backend` → `Toolchain`, `executor` → `compile_pipeline`,
   docstring stage names), plus a short successor ADR recording the changes.

Items 1–3 directly remove the two pains that triggered this proposal
(hard to follow, hard to extend) at low risk. Items 4–7 reduce the concept
count contributors must hold.

## 7. Redesign: a staged compilation API

### 7.1 Design principles (from the prior-art survey)

1. **Stages are few, named, and first-class** — not a generic combinator
   algebra. Every surveyed system that crosses artifact-type boundaries uses
   a small fixed chain at the top level (JAX `trace/lower/compile`, PyTorch
   capture/aot/backend, Triton's stage dict) and pass-manager-style
   sub-pipelines *inside* IR stages (prior art §9.1). GT4Py already has the
   right boundary types; they should become the API instead of being hidden
   inside workflow plumbing.
2. **The backend contract is narrow.** One registrable object with `lower`
   and `compile`, not a four-slot recipe plus two factories (prior art §9.2).
3. **One fingerprint, one artifact store.** Cache keys derived once from the
   canonical lowered artifact + options + toolchain version; all layers are
   indices into one content-addressed store; misses are explainable (prior
   art §§1.3, 9.4).
4. **Inspectability at every boundary.** Each stage object renders itself
   (`as_text()`/`dump()`); an instrumentation context observes transitions;
   debug payloads are explicitly unstable while the stage API is stable
   (JAX's split, prior art §1.1).
5. **Configuration = constructor args + scoped context**, never a parallel
   factory language (prior art §9.3).
6. **Compile-time argument info is one explicit type.** The aval/
   `ShapeDtypeStruct` lesson: a single `ArgSpec` describes exactly what
   compilation specializes on (type, static value, domain descriptor, …) and
   is what both JIT dispatch and AOT `compile()` construct — no
   `JITArgs`-vs-`CompileTimeArgs` adapters (prior art §1.2).

### 7.2 The stage model

```
   Program (definition + ProgramSpec)
      │  frontend: parse / lower / GTIR passes        [pass-list, instrumentable]
      ▼
   LoweredProgram          GTIR + ArgSpecs + offset-provider types; .fingerprint(), .as_text()
      │  backend.lower()                              [per-backend codegen]
      ▼
   CompiledArtifact        sources + bindings + build recipe, or SDFG; .dump(dir)
      │  backend.compile()                            [build systems / DaCe build; artifact store]
      ▼
   Executable              imported callable + calling convention metadata
      │  bind(runtime conversions)
      ▼
   BoundProgram            what CompiledProgramsPool stores and dispatches to
```

Sketch of the user/backend-facing API (names illustrative):

```python
class Backend(Protocol):  # the whole extension contract
    name: str
    device: core_defs.DeviceType

    def lower(self, prog: LoweredProgram, ctx: CompileContext) -> CompiledArtifact: ...
    def compile(self, art: CompiledArtifact, ctx: CompileContext) -> Executable: ...


@dataclasses.dataclass(frozen=True)
class CompileContext:  # scoped, cross-cutting; replaces factory params
    build_cache: ArtifactStore
    options: CompileOptions  # cmake build type, jobs, debug, ...
    instruments: tuple[Instrument, ...] = ()  # on_stage(name, artifact), timing, dumping


# explicit AOT path (JAX-style), also used internally by JIT dispatch:
lowered = program.lower(argspecs, offset_provider_type)  # frontend only
artifact = backend.lower(lowered, ctx)  # inspect: artifact.dump(path)
exe = backend.compile(artifact, ctx)
```

Key differences from today:

- **No generic `Workflow` protocol.** The frontend transform chain becomes
  an ordinary function (or small pass list) `definition -> LoweredProgram`;
  `MultiWorkflow`'s isinstance dispatch becomes overloads per definition
  type. Inside the frontend and inside backends, IR-to-IR passes can use a
  simple pass-list with instrumentation (the family-(a)/(b) hybrid; prior
  art §9.1) — but that is an implementation detail, not the public
  architecture.
- **`bindings`, `compilation`, `decoration` stop being configurable slots.**
  They are implementation details of each backend's `lower`/`compile`,
  composed in plain Python; the *shared* implementations (nanobind binding
  generator, CMake/compiledb projects, importer) remain as a library that
  backends call — which is how they are actually reused today anyway.
- **`Executable` carries its calling convention** (parameter order, expected
  connectivity buffers, device) as data, so the argument-conversion
  "decoration" can be generated/checked against it instead of relying on
  cross-module ordering comments (`gtfn.py:91-94`).
- **`ArgSpec` unifies the argument story**: it is the element type of both
  `lowered.argspecs` and the pool's cache key; static values and domain
  descriptors (ADR 0021) become `ArgSpec` fields rather than a parallel
  descriptor-context machinery. `CompiledProgramsPool` remains (it is a
  strength) but keys on `(lowered.fingerprint(), argspec values)` — the same
  fingerprint as the artifact store.
- **One `ArtifactStore`** (content-addressed by fingerprint, session or
  persistent) replaces `CachedStep`+`FileCache`+build-dir hashing; an entry
  stores *all* stage artifacts side by side (Triton/DaCe model), which makes
  the store itself the debug dump and enables warm starts without re-running
  translation.
- **Backend registration via entry points** (`gt4py.next.backends` group),
  so DaCe-style backends can live out-of-tree (PJRT/PyTorch model). String
  lookup `gtx.backend("gtfn.gpu")` becomes possible alongside direct
  construction.

### 7.3 What this fixes, explicitly

| Weakness (§5)                 | Redesign answer                                                                       |
| ----------------------------- | ------------------------------------------------------------------------------------- |
| 5.1 illusory typing           | Concrete types per stage; no generics, no reflection; mypy checks real signatures     |
| 5.2 artifact gymnastics       | Stages take the arguments they need; no single-argument constraint, no adapters       |
| 5.3 factory_boy               | Plain dataclasses + `CompileContext`; checked keyword arguments                       |
| 5.4 no introspection          | Stage objects with `as_text()`/`dump()`; instruments; supported `lower()` entry point |
| 5.5 cache sprawl              | One fingerprint, one store, explainable misses                                        |
| 5.6 wide extension surface    | Two-method `Backend` protocol + entry-point registration                              |
| 5.7 naming drift / args model | New names are the code; `ArgSpec` replaces the JIT/AOT split                          |

### 7.4 Migration strategy

The redesign can be executed incrementally because the stage *boundaries*
already exist; what changes is the connective tissue:

1. Land incremental items 1–3 (§6) — they build the fingerprint, store, and
   instrumentation the redesign needs, while immediately useful.
2. Introduce `LoweredProgram`/`CompiledArtifact`/`Executable` as thin
   wrappers over today's `CompilableProgramDef`/`CompilableProject`/callable,
   and implement the `Backend` protocol for gtfn by *delegating* to the
   existing steps. The old `Backend` dataclass becomes a shim.
3. Port DaCe; delete `dace/program.py`'s unwrapping in favor of the
   supported stage API. Port roundtrip (trivially: `lower` = codegen to
   Python source, `compile` = exec) — for the first time all five execution
   paths share one architecture.
4. Move `CompiledProgramsPool` keys onto the shared fingerprint; collapse
   `JITArgs`/`CompileTimeArgs` into `ArgSpec`.
5. Remove `otf/workflow.py`, `otf/toolchain.py`, `otf/recipes.py`, the
   factories, and the obsolete adapters; write the superseding ADR.

Steps 2–3 are the risky middle; the test-exclusion matrix (ADR 0015) plus
the existing integration suites give per-backend coverage, and the shim
keeps external code (`gtx.gtfn_cpu` style references) working until a
deprecation cycle completes.

### 7.5 Risks and open questions

- **Losing flexibility we actually use?** A survey of in-tree uses found no
  workflow recomposition that the two-method backend + library-of-parts
  cannot express; but downstream users (e.g. icon4py) may compose workflows
  directly — needs a check and a deprecation window for `otf.workflow`.
- **DaCe-orchestrated programs** (`DaCeProgram`) intentionally reach *into*
  the pipeline; the redesign must expose lowering-with-options
  (`backend.lower(lowered, ctx.with_options(...))`) richly enough to cover
  that use case — the current workaround is the best requirements list.
- **Async compilation and process pools**: making `CompiledArtifact`
  picklable (it is mostly strings + metadata) would unlock the
  `ProcessPoolExecutor` TODO (`compiled_program.py:157`); worth designing in
  from the start.
- **How much pass-manager?** This proposal deliberately does *not* import a
  full MLIR-style pass manager for GTIR transforms; a plain instrumented
  pass list suffices today. Revisit if pass dependencies/conditional passes
  materialize (DaCe's `Modifies`/`depends_on` model is the lightweight next
  step).
- **Stability tiers**: adopt JAX's three-ring honesty — stable user API
  (decorator, `compile`, `lower`), documented-but-unstable backend-author
  surface, internals — and say so in the docs (prior art §1.5).

## 8. Recommendation

Do §6 items 1–3 now; they are small, independently valuable, and build the
foundation. Then pursue the redesign via the shim path (§7.4) rather than
continuing to invest in the generic workflow framework: the survey shows the
industry converged on explicit staged APIs with narrow extension seams, the
current framework's flexibility is unused where it isn't actively harmful,
and the migration can ride the existing stage boundaries without a
big-bang rewrite.

## References

- Code: `src/gt4py/next/otf/`, `src/gt4py/next/backend.py`,
  `src/gt4py/next/program_processors/runners/{gtfn.py,dace/,roundtrip.py}`
  (all citations as of the commit this document was added in).
- ADRs: [0009](../ADRs/next/0009-Compiled-Backend-Integration.md),
  [0011](../ADRs/next/0011-On_The_Fly_Compilation.md),
  [0012](../ADRs/next/0012-GridTools_Cpp_OTF_Steps.md),
  [0016](../ADRs/next/0016-Multiple-Backends-and-Build-Systems.md),
  [0017](../ADRs/next/0017-Toolchain-Configuration.md),
  [0021](../ADRs/next/0021-Argument-Descriptors.md).
- External: see the
  [prior-art companion document](otf-pipeline-redesign-prior-art.md) for the
  full survey with citations (JAX, PyTorch 2, MLIR, TVM, Numba, DaCe,
  Triton, build-system prior art).
