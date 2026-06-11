---
tags: [backend, otf, workflows, toolchain, research]
---

# Prior Art: Compilation Pipelines in Python-Embedded Compiler Frameworks

- **Status**: research notes (companion to
  [OTF Pipeline Redesign](otf-pipeline-redesign.md))
- **Created**: 2026-06-11

This document collects an extended summary of how other Python-embedded
compiler frameworks organize their tracing / lowering / compilation /
execution pipelines. It is the evidence base for the proposal in
[`otf-pipeline-redesign.md`](otf-pipeline-redesign.md); the proposal only
references the conclusions, this document carries the details and citations.

Frameworks covered: JAX (in depth, as the closest analogue), PyTorch 2.x
(`torch.compile`), MLIR's pass manager, TVM, Numba, DaCe, Triton, and a short
section on build-system/import-stage prior art (cppimport, scikit-build-core,
Halide). A cross-cutting synthesis closes the document.

______________________________________________________________________

## 1. JAX

### 1.1 The staged pipeline and the explicit AOT API (`jax.stages`)

JAX decomposes "run this Python function fast" into four explicitly named,
individually accessible stages ([AOT docs](https://docs.jax.dev/en/latest/aot.html)):

1. **Trace** — specialize the Python function on the *types*
   (shape/dtype/pytree structure) of its arguments, producing a **jaxpr**
   (JAX's canonical functional IR).
2. **Lower** — translate the jaxpr to **StableHLO** (an MLIR dialect), the
   contract IR with the compiler.
3. **Compile** — hand StableHLO to XLA, which optimizes for a concrete device
   and returns a loaded executable.
4. **Execute** — call the executable with concrete arrays.

The default `jit` path runs all four lazily and caches; the **AOT API**
exposes each stage as a method chain producing a typed stage object:

```python
jitted = jax.jit(f)  # jax.stages.Wrapped (protocol)
traced = jitted.trace(*args, **kwargs)  # jax.stages.Traced
lowered = traced.lower()  # jax.stages.Lowered
compiled = lowered.compile()  # jax.stages.Compiled
out = compiled(x)  # strict execution
```

Per the [`jax.stages` reference](https://docs.jax.dev/en/latest/jax.stages.html),
each stage object exposes inspection hooks:

- **`Lowered`**: `as_text(dialect=None, debug_info=False)` (human-readable
  StableHLO/HLO), `compiler_ir(dialect=None)` (the actual MLIR module
  object), `cost_analysis()`, and an `in_tree` property recording the pytree
  structure of arguments.
- **`Compiled`**: `__call__`, `as_text()` (post-optimization HLO),
  `cost_analysis()` (FLOP/bytes estimates), `memory_analysis()`,
  `runtime_executable()` (the raw PJRT executable), `in_tree`.

Design decisions worth noting:

- *Stage objects are the API, not functions with mode flags.* Each stage is a
  first-class value you can hold, inspect, and pass around. The contract on
  what stage transitions exist is stable; the contract on debug *outputs* is
  explicitly disclaimed — `as_text()` / `cost_analysis()` are "for
  debugging", with "no guarantee of consistency across invocations" or across
  JAX versions/backends/platforms. This split — *stable workflow API,
  unstable introspection payloads* — lets JAX evolve its IRs freely.
- *Strictness increases monotonically down the pipeline.* A `Compiled` object
  rejects argument shapes/dtypes that differ from those it was specialized on
  (a `TypeError`, never silent re-specialization), and cannot be re-entered
  by transformations: wrapping a `Compiled` in `vmap`/`grad`/`jit` raises.
  Staging out is a one-way door, by design.
- *In-process vs. cross-process is a separate API.* `Lowered`/`Compiled` are
  deliberately process-local; serialization is delegated to **`jax.export`**,
  which wraps StableHLO + metadata (in/out types, platforms,
  calling-convention version) in an `Exported` object with explicit
  compatibility windows (6 months backward, 3 weeks forward) — guarantees
  that only hold if you go through the official API rather than grabbing the
  StableHLO yourself ([export docs](https://docs.jax.dev/en/latest/export/export.html)).
  A persistent compiled-program cache and a "ship this artifact" feature are
  different products with different stability needs.
- There is no numbered JEP for `jax.stages`; the design grew out of
  [issue #7733](https://github.com/jax-ml/jax/issues/7733) and is documented
  as the [AOT guide](https://docs.jax.dev/en/latest/aot.html) plus the module
  reference.

### 1.2 Modeling compile-time vs. runtime argument information

JAX's central abstraction is the **abstract value (aval)** — chiefly
`ShapedArray(shape, dtype, weak_type)` — which is *exactly* the information
the compiler specializes on, no more. The jaxpr IR is typed with avals
(`a:f32[8]`), is functionally pure (no free variables; constants are hoisted
to `constvars`), and is first-order with a handful of higher-order primitives
(`cond`, `while`, `scan`, `pjit`) carrying sub-jaxprs
([jaxpr docs](https://docs.jax.dev/en/latest/jaxpr.html)).

The user-facing mirror of an aval is
**`jax.ShapeDtypeStruct(shape, dtype, *, sharding=None, weak_type=False)`** —
"a container for the shape, dtype, and other static attributes of an array"
([reference](https://docs.jax.dev/en/latest/_autosummary/jax.ShapeDtypeStruct.html)).
Any AOT stage accepts these "hollow" arguments in place of real arrays, so
one can trace/lower/compile with zero data allocated. This duality (concrete
arg → aval ← explicit spec object) is the key enabler of the AOT API: *the
JIT path and the AOT path share one specialization mechanism.*

Argument classification happens in `jax.jit`'s signature
([jax.jit reference](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html)):

- **Traced args** (default): only shape/dtype enter the cache key; values are
  runtime inputs.
- **Static args** (`static_argnums`/`static_argnames`): the *value* is baked
  into the trace and becomes part of the cache key; must be hashable; every
  new value triggers retrace + recompile
  ([jit docs](https://docs.jax.dev/en/latest/jit-compilation.html)).
- **Donated args** (`donate_argnums`/`donate_argnames`): a third, orthogonal
  category — runtime info for the *runtime*, not the compiler: the buffer may
  be aliased to outputs and must not be reused by the caller.
- `keep_unused=False`: the pipeline may *drop* unused arguments from the
  executable — the compiled calling convention can differ from the Python
  signature, and the stage objects track this internally.

**Pytrees** decouple user-facing argument structure from the flat
tuple-of-arrays calling convention: every transformation flattens args into
`(leaves, treedef)` via `jax.tree.flatten`, operates on leaves, and
unflattens outputs; custom container types are supported via node
registration ([pytrees docs](https://docs.jax.dev/en/latest/pytrees.html)).
The `treedef` participates in the cache key, and `Lowered.in_tree` /
`Compiled.in_tree` expose it so callers can reconstruct the calling
convention.

**Cache key = (function identity, treedef, avals of dynamic leaves, static
arg values, relevant config/context).** A notable consequence: functions
redefined in a loop (lambdas, `functools.partial`) hash differently each time
and defeat the cache entirely — a documented top pitfall.

### 1.3 Caching and dispatch: in-memory layers, C++ fast path, persistent cache

JAX has (at least) three cache layers:

**(a) In-memory dispatch cache with a C++ fast path.** Jitted functions are
`PjitFunction` objects implemented in C++
([PR #4051](https://github.com/google/jax/pull/4051)). On call, C++ code
computes the argument signature (splitting dynamic from static args) and
looks up the executable; on a hit, dispatch never touches the Python tracing
machinery — overhead is single-digit microseconds, deliberately small
relative to a GPU kernel launch so Python can "run ahead" of the device via
async dispatch ([discussion #8281](https://github.com/jax-ml/jax/discussions/8281)).
On a miss it falls back to Python (trace → lower → compile). Historically
there were two stacked executable caches with subtly different keys; this
produced a real bug where an inner-cache hit could poison the outer cache
when functions shared a jaxpr but differed in compile parameters — fixed by
having the outer miss path lower/compile directly
([PR #24828](https://github.com/jax-ml/jax/pull/24828)). *Lesson: multiple
cache layers with independently derived keys are a bug factory; derive keys
once, at one level.* Tracing itself is cached and interned via weakref-LRU
caches keyed on (function, avals, treedef), so repeated `lower()` calls reuse
the jaxpr.

**(b) Persistent (disk) compilation cache**
([docs](https://docs.jax.dev/en/latest/persistent_compilation_cache.html)).
Opt-in via `JAX_COMPILATION_CACHE_DIR`. Key properties:

- **Cache key** = hash of the *non-optimized* HLO of the function, plus
  jaxlib version, relevant XLA compilation flags, device configuration, and
  an overridable `custom_hook()` for user salt. Keying on the
  pre-optimization IR (not Python source) means the cache survives cosmetic
  refactors but correctly misses on compiler-relevant changes.
- **Thresholds** avoid caching trivia:
  `jax_persistent_cache_min_compile_time_secs` (default 1.0 s) and
  `jax_persistent_cache_min_entry_size_bytes`.
- **Multi-process**: only rank 0 writes; remote filesystems supported.
- **Debuggability is a feature**: `jax_explain_cache_misses` logs *why* a key
  missed.
- **Documented failure modes**: host callbacks and `custom_partitioning`
  embed pointers in HLO, breaking key stability. *Lesson: anything in the IR
  that isn't value-semantic poisons content-addressed caching.*
- **Security note**: a shared writable cache is arbitrary-code-execution; the
  docs say "do not share a compilation cache with users you do not trust."
  Directly relevant to GT4Py's on-disk build caches.

### 1.4 Multiple backends through one lowering path: PJRT

JAX has **one** trace → jaxpr → StableHLO pipeline for all platforms;
per-platform divergence is pushed *below* the IR contract into the
compiler/runtime, accessed through **PJRT**, a C API that packages "compiler

- runtime" behind a uniform device interface
  ([PJRT integration](https://openxla.org/xla/pjrt/pjrt_integration),
  [RFC](https://github.com/openxla/community/blob/main/rfcs/20230123-pjrt-plugin.md)).
  Key mechanics:

* A plugin is a shared library exposing `GetPjrtApi()` returning a struct of
  function pointers (client creation, device enumeration, compile, execute,
  buffer management). Device internals are opaque to the framework.
* **Discovery is pip-installable**: plugins live in the `jax_plugins`
  namespace package or advertise a `jax_plugins` entry point, and implement
  `initialize()` which calls JAX's `register_plugin`
  ([integration guide](https://github.com/openxla/xla/blob/main/xla/pjrt/c/docs/pjrt_integration_guide.md)).
  This is how Apple METAL, Intel, and Google TPU support ship out-of-tree.
* Platform selection appears in the *API* only as strings:
  `jit(..., backend='gpu')`, `Traced.lower(lowering_platforms=('tpu','cuda'))`;
  `jax.export` can produce one multi-platform artifact.

Caveat for GT4Py: JAX can do this because XLA absorbs *all* codegen; GT4Py's
backends differ at the IR level, so the analogous move is standardizing the
*stage interfaces and artifact metadata*, not the IR itself. Small
per-platform IR differences are handled in JAX by platform-conditional
lowering rules for individual primitives, not by forking the pipeline.

### 1.5 Effects, pain points, and the extension-API philosophy

**Effects.** Because jaxprs are pure and execution is asynchronous, side
effects (debug prints, callbacks) required a dedicated design:
[JEP 10657](https://docs.jax.dev/en/latest/jep/10657-sequencing-effects.html)
threads dummy **token** values through jaxprs and through the runtime
dispatch path. Effects are tracked in the jaxpr type and surface in the
export calling convention. *Lesson: if your IR is pure, effects must be
modeled as explicit values from day one or they break caching and ordering
later* (note the persistent-cache failure with host callbacks above).

**Recurring user pain points:**

- **Recompilation on shape change / no dynamic shapes**: every new shape is a
  new cache entry; variably-shaped data forces padding/bucketing workarounds
  ([issue #2521](https://github.com/jax-ml/jax/issues/2521)). `jax.export`
  mitigates via symbolic dimension variables (shape polymorphism), but the
  core JIT path remains shape-monomorphic.
- **Silent cache defeat**: closures/lambdas redefined per call, unhashable or
  wrongly-classified static args.
- **Cache-layer mismatch confusion**: a persistent-cache hit but
  tracing-cache miss still costs trace time
  ([issue #22281](https://github.com/jax-ml/jax/issues/22281)) — users
  perceive "the cache" as one thing; staged caches need staged diagnostics.

**`jax.extend` philosophy
([JEP 15856](https://docs.jax.dev/en/latest/jep/15856-jex.html)).** JAX
acknowledged that downstream projects depend on internals (primitives,
jaxpr, lowering rules) and created a *second-tier public API* —
`jax.extend.core`, `.interpreters`, `.mlir` — that is discoverable and
changelogged but **explicitly promises no deprecation windows**. Rationale:
better to give extenders a named, honest surface than have them import
`jax._src`. This maps to a three-ring structure: (1) stable user workflow
API (the stage objects), (2) a documented-but-unstable extension surface for
backend authors, (3) true internals.

______________________________________________________________________

## 2. PyTorch 2.x — `torch.compile`

**Pipeline structure.** A *fixed sequence of heterogeneous stages*, each with
its own IR and contract, rather than a uniform pass manager:

1. **TorchDynamo** — graph capture. Hooks CPython's frame-evaluation API
   (PEP 523) to rewrite Python bytecode just before execution, extracting
   sequences of PyTorch ops into an **FX `GraphModule`**; unsupported
   constructs cause *graph breaks*, splitting execution between compiled
   graphs and fallback Python
   ([Dynamo overview](https://docs.pytorch.org/docs/stable/torch.compiler_dynamo_overview.html)).
2. **AOTAutograd** — traces through the autograd engine to produce joint
   forward+backward FX graphs ahead of time, decomposed into the small *core
   ATen* opset.
3. **TorchInductor** — the default backend: define-by-run loop-level IR,
   scheduling/fusion, then codegen to **Triton** kernels on GPU and
   **C++/OpenMP** on CPU
   ([ASPLOS '24 paper](https://dl.acm.org/doi/10.1145/3620665.3640366)).

**Configuration/extension.** The stage boundary between capture and backend
is a deliberately minimal typed contract: a backend is any callable
`(gm: torch.fx.GraphModule, example_inputs: list[Tensor]) -> Callable`.
Registration is by *name in a string-keyed registry*
(`torch._dynamo.register_backend`), discoverable via `list_backends()` and
composable via `lookup_backend("inductor")` inside a custom backend;
entry-point-based registration lets third-party packages add backends
([custom backends doc](https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html)).
Inside Inductor, extension is via *config-slot hooks*
(`post_grad_custom_post_pass` etc.) plus a `PatternMatcherPass` container for
declarative graph rewrites; an open RFC
([#153532](https://github.com/pytorch/pytorch/issues/153532)) proposes a
proper ordered registration interface because config slots don't compose.

**Caching.** Layered, per-stage, on-disk caches that mirror the stage
structure ([caching tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)):

- **FXGraphCache** — Inductor-level cache of compiled graphs, keyed on a hash
  of the FX graph + input metadata + relevant configs.
- **AOTAutogradCache** — sits *above* FXGraphCache, caching joint-graph
  artifacts so a hit skips AOTAutograd entirely
  ([#128234](https://github.com/pytorch/pytorch/issues/128234)).
- **TritonCache**, **AutotuningCache**, **PGO cache** (dynamic-shape
  decisions).
- **Mega-Cache**: `torch.compiler.save_cache_artifacts()` /
  `load_cache_artifacts()` bundles everything into a portable blob; remote
  Redis-backed caches exist. Validation pins PyTorch version, Triton version,
  and GPU model.

**Recompilation/guards.** Each compiled graph is protected by **guards** —
predicates on input properties (tensor shapes/dtypes, Python constants).
Guard failure triggers recompilation, bounded by
`torch._dynamo.config.cache_size_limit` (default 8); beyond that the frame
falls back to eager. This is the key design difference from "hash everything
up front": correctness of cache reuse is *checked at call time by cheap
predicates*, not encoded purely in the key
([recompilation doc](https://docs.pytorch.org/docs/stable/compile/programming_model.recompilation.html)).

**Inspectability.** `TORCH_LOGS=dynamo,aot,inductor` for per-stage IR dumps,
`gm.print_readable()` on any FX graph, `TORCH_COMPILE_DEBUG=1` writes
per-graph debug directories with each IR stage and generated Triton/C++;
`depyf` decompiles Dynamo's transformed bytecode.

**Trade-offs.** Stages are heterogeneous and not user-recomposable (you
can't reorder Dynamo/AOTAutograd); extensibility is concentrated at *two
sanctioned seams* (backend callable, FX pass hooks). This keeps the common
path robust but makes deep customization awkward (hence the RFC).

______________________________________________________________________

## 3. MLIR PassManager — the canonical pass-pipeline design

([Pass Infrastructure docs](https://mlir.llvm.org/docs/PassManagement/))

**Pipeline structure.** `PassManager` is the top-level entry; `OpPassManager`
instances nest to match the IR's region structure
(`pm.nest<func::FuncOp>()`). Passes are anchored on operation types; anchor
ops must be `IsolatedFromAbove`, which is what makes parallel execution over
siblings safe; passes must be copy-constructible and stateless across runs.

**Textual pipelines.** A pipeline *is data*: a round-trippable textual syntax

```
builtin.module(func.func(cse,canonicalize),convert-func-to-llvm{use-bare-ptr-memref-call-conv=1})
```

including nested pipelines and per-pass `{key=value}` options. This enables
reproducers, CLI-driven experimentation, and serializing the exact pipeline
alongside crashing IR.

**Registration/extension.** `PassRegistration<MyPass>()` for passes;
`PassPipelineRegistration<Options>("arg", "desc", builderFn)` registers
named, parameterized *pipelines* as first-class reusable units. Pass options
use declarative `Option<T>`/`ListOption<T>`; passes expose `Pass::Statistic`
counters.

**Instrumentation/debugging.** `PassInstrumentation` hooks
(`runBeforePass/AfterPass/AfterPassFailed`, before/after analysis) registered
via `PassManager::addInstrumentation()`. IR printing is just an
instrumentation: `-mlir-print-ir-before/after(-all)`,
`-mlir-print-ir-after-change`, `-mlir-print-ir-tree-dir=<path>` (one file per
pass into a directory tree). Crash reproducers emit the IR plus the textual
pipeline so any failure is replayable.

**Trade-offs/rationale.** A uniform IR (everything is an operation) buys a
*single* generic pass manager with nesting, parallelism, and analysis
caching/invalidation — but requires all stages to live in one IR data model.
This is the opposite end of the spectrum from GT4Py's typed
`Workflow[StartT, EndT]` chain, where each stage may have a different
artifact type.

______________________________________________________________________

## 4. TVM (Relax) pass infrastructure

([Pass Infrastructure — TVM docs](https://tvm.apache.org/docs/arch/pass_infra.html))

**Pipeline structure.** All passes are `IRModule -> IRModule` transforms.
Granularity is encoded as pass *kinds* — module-level (inter-procedural) vs
function-level. `tvm.transform.Sequential` (explicitly modeled on
`torch.nn.Sequential`) composes an ordered list and is itself a pass, so
sequences nest.

**Configuration.** `PassContext` is a *scoped, thread-local context object*
(a Python context manager, `PassContext::Current()`): `opt_level`,
`required_pass`, `disabled_pass`, `instruments`, free-form `config` dict.
Each pass carries `PassInfo` metadata (name, `opt_level`, `required`
prerequisites); a pass runs only if its opt_level ≤ the context's, unless
explicitly required/disabled. Dependencies are *declared* and resolved by
`Sequential` rather than encoded in call order.

**Registration/extension.** Decorators
`@tvm.transform.module_pass(opt_level=...)` /
`@relax.transform.function_pass` turn Python callables/classes into passes;
C++ passes are exposed via FFI, so Python and C++ passes interleave freely in
one `Sequential` — an explicit design goal.

**Instrumentation/inspectability.** `PassInstrument` with hooks
`enter_pass_ctx`, `exit_pass_ctx`, `should_run` (can veto a pass),
`run_before_pass`, `run_after_pass`; instances are passed in
`PassContext(instruments=[...])`. Built-ins: `PassTimingInstrument`,
`PrintBeforeAll`/`PrintAfterAll`. Since IRModule is printable and
round-trippable as TVMScript, any intermediate stage can be dumped and
re-parsed
([pass instrument how-to](https://tvm.apache.org/docs/v0.8.0/how_to/extend_tvm/use_pass_instrument.html)).

Caching is not part of the pass infra proper; the notable idea for GT4Py here
is the *context + metadata-driven pass gating* rather than caching.

______________________________________________________________________

## 5. Numba

**Pipeline structure.** `@jit` triggers: bytecode analysis → Numba IR →
rewrites/analysis → type inference → lowering to LLVM IR via llvmlite → LLVM
optimization → native code
([architecture doc](https://numba.readthedocs.io/en/stable/developer/architecture.html)).

**Customization API.** Unusually formalized for a JIT: the compiler is
`numba.core.compiler.CompilerBase`; subclass and override
`define_pipelines()` returning `PassManager` instances.
`DefaultPassBuilder.define_nopython_pipeline(state)` yields the stock
pipeline; you then surgically modify it
(`pm.add_pass_after(MyPass, IRProcessing)`). Custom passes subclass
`FunctionPass` with `run_pass(state)` mutating a shared `state` object (a
mutable compilation-state record — IR, type map, flags). Activation is *per
call site*: `@jit(pipeline_class=MyCompiler)`. Docs explicitly flag it "for
expert use only" since passes share invariants through the untyped `state`
([customizing the compiler](https://numba.readthedocs.io/en/stable/developer/custom_pipeline.html)).

**Caching.** `@jit(cache=True)`: file-based, per-function index file (`.nbi`)
listing compiled overload signatures + one `.nbc` payload per overload,
pickle-serialized. The effective key = function source location + type
signature of the overload + interpreter/architecture magic. Invalidation is
by *source-file timestamp/content*: editing anything in the file invalidates
every cached function in it, and changes in symbols defined in *other* files
are not detected — a known correctness/granularity weakness
([caching notes](https://numba.readthedocs.io/en/stable/developer/caching.html),
[#9316](https://github.com/numba/numba/issues/9316)). Cache
locator/backing-store classes are pluggable.

**Inspectability.** Per-artifact accessors on the dispatcher object:
`inspect_types()`, `inspect_llvm()`, `inspect_asm()`;
`NUMBA_DEBUG_PRINT_AFTER=all` for per-pass IR printing.

**Trade-off.** The `state`-blob pass manager is flexible but
stringly/dynamically typed — exactly the failure mode GT4Py's typed
`Workflow[StartT, EndT]` protocol is designed to prevent; Numba shows the
cost (expert-only, invariant breakage) of giving up typed stage boundaries.

______________________________________________________________________

## 6. DaCe (also one of GT4Py's backends)

([Passes and Pipelines](https://spcldace.readthedocs.io/en/latest/sdfg/passes.html),
[Code Generation](https://spcldace.readthedocs.io/en/latest/codegen/codegen.html))

**Pass/Pipeline API.** `dace.transformation.pass_pipeline.Pass` defines
`apply_pass(sdfg, pipeline_results: dict[str, Any])` — results of earlier
passes are passed *by pass-class name* in a dict, making analyses shareable.
Contract: return `None` iff nothing changed. Passes declare:

- `modifies() -> Modifies` (bitflags: which graph element classes they
  touch),
- `should_reapply(modified: Modifies) -> bool`,
- `depends_on() -> set[type[Pass]]`.

`Pipeline` is itself a `Pass` (arbitrary nesting); it topologically orders
dependencies and *skips* re-running a pass when nothing it cares about was
modified — a lightweight, declarative invalidation scheme;
`FixedPointPipeline` iterates to convergence. Even interactive
pattern-matching transformations are unified as passes.

**Codegen/build/caching.** Code generation emits C++/CUDA into a per-program
folder **`.dacecache/<program>/`** (`src/`, `build/`, `include/`, plus
`program.sdfg` snapshots and `map_codegen.json` source maps for IDE
navigation). The build stage invokes CMake in `.dacecache/<program>/build`,
producing a shared library loaded via a `CompiledSDFG` ctypes wrapper.
Cache-folder naming policy is configurable (`name`/`hash`/`unique`), with
hash mode keying the folder on the SDFG content hash. This — a
*human-readable on-disk project per program, with the IR snapshot stored next
to the generated sources* — is the closest existing analogue to GT4Py otf's
source → project → compiled-module tail, and DaCe's debuggability (open the
folder, read the C++, reload the SDFG) is a direct consequence.

______________________________________________________________________

## 7. Triton

**Pipeline structure.** Single-target, linear lowering: Python AST → **TTIR**
(machine-independent MLIR dialect) → **TTGIR** (TritonGPU dialect: layout
encodings, pipelining) → **LLVM IR** → **PTX** → **cubin**
([PyTorch blog: Triton kernel compilation stages](https://pytorch.org/blog/triton-kernel-compilation-stages/)).
Internally `triton.compile()` drives an ordered dict of stage functions which
each *backend* (a `BaseBackend` discovered from `triton/backends/`) registers
via `add_stages()` — `make_ttir`, `make_ttgir`, `make_llir`, `make_ptx`,
`make_cubin`; each stage is essentially "run this MLIR pass pipeline, return
the next artifact"
([internals deep dive](http://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/)).

**Caching.** Two layers: in-memory per-`JITFunction` dict, and on-disk
`~/.triton/cache` managed by `FileCacheManager` (pluggable via
`TRITON_CACHE_MANAGER`). The key hashes: kernel source, type signature,
`constexpr` values, *runtime specialization* (argument values like
divisibility-by-16 alignment, unless `do_not_specialize`), compiler options,
backend/arch (`sm_89`), and the Triton version. A cache directory holds
**every intermediate artifact side by side**: `kernel.json` metadata +
`.ttir`, `.ttgir`, `.llir`, `.ptx`, `.cubin` — caching and inspectability are
the same mechanism. At runtime the `CompiledKernel.asm` dict exposes all
stages (`kernel.asm['ttgir']` etc.).

**Trade-off.** Because there is exactly one pipeline shape per backend, the
"framework" is just a list of named stage functions plus a hash — radically
simpler than MLIR-style generality, viable because the domain (one kernel,
one target) is narrow. A good calibration point for how much machinery otf
actually needs per backend.

______________________________________________________________________

## 8. Build-system / import-stage prior art (short notes)

- **cppimport** — `import foo` triggers compilation of `foo.cpp` via an
  import hook; build config is embedded in the source file as a Mako block;
  the rebuild decision uses a *content checksum of the source + declared
  dependency files* appended to the extension; `CPPIMPORT_RELEASE_MODE` skips
  checks in production. Built on setuptools, hence no incremental builds
  ([GitHub](https://github.com/tbenthompson/cppimport)).
- **pybind11 build docs** survey the JIT-ish options (cppimport for
  experiments, CMake for production)
  ([pybind11 compiling](https://pybind11.readthedocs.io/en/stable/compiling.html)).
- **scikit-build-core** — a PEP 517 build backend driving CMake, configured
  statically in `pyproject.toml`
  ([SciPy proceedings paper](https://proceedings.scipy.org/articles/FMKR8387)).
  Relevant as the modern "build-system project" abstraction: source tree +
  CMakeLists + declarative config in, wheel/extension out.
- **Halide** — the classic *algorithm/schedule separation*: the same
  functional algorithm composed with an independently specified schedule,
  lowered by a fixed pipeline
  ([SIGGRAPH 2012](https://people.csail.mit.edu/jrk/halide12/)). The
  transferable idea: keep "what" (stencil semantics) and "how"
  (backend/layout/parallelization decisions) as separate inputs to the
  pipeline rather than baked into stages.
- Numba, Triton, and DaCe all converge on: per-program cache directory,
  key = hash(IR/source + options + toolchain version + target arch).

______________________________________________________________________

## 9. Cross-cutting patterns

1. **Two architectural families.**
   (a) *Uniform-IR pass managers* (MLIR, TVM, DaCe passes, Numba's
   PassManager): every stage is `IR -> IR` over one data model; composition
   is a runtime-assembled list with metadata-driven scheduling (opt_level,
   depends_on, Modifies).
   (b) *Typed stage chains* (Dynamo → AOTAutograd → Inductor; Triton's
   make_ttir → … → make_cubin; JAX's trace → lower → compile; GT4Py otf
   today): each stage maps one artifact type to another; the pipeline shape
   is mostly fixed and extension happens at designated seams.
   Every surveyed system that crosses *artifact-type boundaries* (graph →
   kernel source → binary) uses family (b) at the top level and family (a)
   *inside* IR-level stages — supporting a hybrid design for otf: a small
   fixed set of typed macro-stages, with pass-manager-style sub-pipelines
   within IR stages.
2. **Extension via named registries + a minimal callable contract** beats
   subclass-the-pipeline: `register_backend` +
   `(GraphModule, example_inputs) -> Callable` (PyTorch), `add_stages`
   (Triton), decorator registration (TVM), pip-installable PJRT plugins
   (JAX). Numba's subclass-`CompilerBase` approach is the least ergonomic and
   is officially "expert-only".
3. **Context objects instead of constructor wiring.** MLIR/TVM thread a
   scoped context (`PassContext`) carrying opt level, an options dict, and
   *instruments*, instead of baking configuration into each stage instance.
   GT4Py's `NamedStepSequence` dataclasses are constructor-wired; the survey
   suggests adding an orthogonal context/instrumentation channel rather than
   more constructor parameters.
4. **Artifact caches mirror pipeline stages and are keyed on hash(input
   artifact + config + toolchain version + target).** PyTorch layers
   AOTAutogradCache over FXGraphCache over TritonCache with a portable
   Mega-cache bundle; Triton keys on
   source+signature+constexprs+options+arch+version and stores *all*
   intermediates in the entry; Numba shows the anti-pattern (file-mtime
   invalidation, cross-file blindness); JAX keys its persistent cache on the
   canonical pre-optimization IR and can *explain* misses. PyTorch adds the
   distinct idea of *guards*: cheap runtime predicates validating reuse
   instead of over-hashing.
5. **Inspectability = instrumentation hooks + dump-everything cache
   entries.** Either before/after-pass hooks with IR printing (MLIR
   `-mlir-print-ir-after-all`, TVM `PrintAfterAll`) or an
   artifact-per-stage directory on disk (Triton cache dir, DaCe
   `.dacecache`, `TORCH_COMPILE_DEBUG`). Both reduce to: every stage boundary
   must be serializable and addressable by name.
6. **Pipelines as data.** MLIR's textual pipeline syntax and registered named
   pipelines make the pipeline itself loggable, diffable, and embeddable in
   crash reproducers — a capability purely code-composed workflows lack.
7. **Stable workflow API, unstable payloads, honest extension tier.** JAX's
   `jax.stages` (stable stage transitions, explicitly-unstable debug output)
   plus `jax.extend` (named, changelogged, no-compatibility-promise extension
   surface) is the most deliberate answer to "how do we let people build on
   our pipeline without freezing our internals".

## References

All sources are linked inline. The primary design documents are:

- JAX: [AOT lowering & compilation](https://docs.jax.dev/en/latest/aot.html),
  [jax.stages](https://docs.jax.dev/en/latest/jax.stages.html),
  [persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html),
  [JEP 15856 `jax.extend`](https://docs.jax.dev/en/latest/jep/15856-jex.html),
  [JEP 10657 effects](https://docs.jax.dev/en/latest/jep/10657-sequencing-effects.html),
  [PJRT integration](https://openxla.org/xla/pjrt/pjrt_integration)
- PyTorch 2: [ASPLOS '24 paper](https://dl.acm.org/doi/10.1145/3620665.3640366),
  [custom backends](https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html),
  [compile caching](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)
- MLIR: [Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)
- TVM: [Pass Infrastructure](https://tvm.apache.org/docs/arch/pass_infra.html)
- Numba: [architecture](https://numba.readthedocs.io/en/stable/developer/architecture.html),
  [custom pipelines](https://numba.readthedocs.io/en/stable/developer/custom_pipeline.html),
  [caching](https://numba.readthedocs.io/en/stable/developer/caching.html)
- DaCe: [passes](https://spcldace.readthedocs.io/en/latest/sdfg/passes.html),
  [codegen](https://spcldace.readthedocs.io/en/latest/codegen/codegen.html)
- Triton: [compilation stages](https://pytorch.org/blog/triton-kernel-compilation-stages/),
  [internals deep dive](http://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/)
