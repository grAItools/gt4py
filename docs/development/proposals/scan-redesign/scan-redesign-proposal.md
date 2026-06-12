# Proposal: Redesign of the vertical `scan` in `gt4py.next`

- **Status**: draft, for discussion
- **Authors**: Hannes Vogt (@havogt), drafted with Claude Code
- **Created**: 2026-06-11
- **Updated**: 2026-06-12 (K-local views, inclusive/exclusive scans,
  icon4py muphys evidence, scan fusion)
- **Companion document**: [scan-redesign-research.md](scan-redesign-research.md)
  (detailed analysis of the current implementation, prior art with sources, and
  the scan + sub-reduce case study);
  [scan-redesign-examples.md](scan-redesign-examples.md) (worked before/after
  examples, including the muphys decomposition and the sedimentation
  reference implementations)

Why-statement: In the context of vertical recurrences in weather and climate
models (vertical integrals, tridiagonal solvers, sedimentation), facing an
implicit scan range, a scalar element-function that is slow in embedded
execution and needs type promotion magic, and clumsy boundary-condition
handling, we propose to replace `scan_operator` with a *level-slice* scan
primitive with `jax.lax.scan`-compatible semantics — explicit `init`
(field-valued, supplied at the call site), separation of carry and per-level
output, scan range deduced from *inputs* with explicit override — with
scanned inputs presented as *K-local views* (relative level access without
delay-line carries), plus a *vertical window* construct that covers
scan + sub-reduce algorithms, and scan fusion as a toolchain obligation. We
considered keeping the scalar body, whole-column kernels, and GTScript-style
interval dispatch, and we accept that data-dependent per-column control flow
inside the scan body must be expressed with `where` instead of Python `if`.

## 1. Background: the current scan and its problems

The current API is a scalar per-element function with carry threading
(`src/gt4py/next/ffront/decorator.py`):

```python
@gtx.scan_operator(axis=KDim, forward=True, init=0.0)
def cumsum(carry: float, val: float) -> float:
    return carry + val
```

The problems, with evidence (details and file/line references in the
companion document, §A):

- **P1 — Implicit scan range.** The vertical range is deduced from the
  *output* domain and threaded through a context variable
  (`closure_column_range` in `src/gt4py/next/embedded/context.py`, consumed in
  `embedded/operators.py` and `iterator/embedded.py`). This is the opposite of
  how every other field operation works (domains propagate from inputs), it is
  fragile (a placeholder `NamedRange` is patched at call time), and it is
  unclear what the semantics *should* be when the scan output is consumed by
  an operation on a different domain.

- **P2 — Scalar element body.** The scan pass maps scalars to scalars while
  call sites pass and receive fields. Pro: Python `if` works per column in
  embedded execution. Cons: embedded execution is a Python loop over *every
  horizontal position times every level*
  (`embedded/operators.py::ScanOperator.__call__`, O(n_horizontal · n_k)
  interpreter iterations); the type system needs dedicated promotion/demotion
  machinery (`ffront/type_info.py::_scan_param_promotion`); and the construct
  cannot be mapped to `jax.lax.scan` or any array-level scan.

- **P3 — Carry is the output.** Whatever is carried is emitted at every
  level and vice versa. The ICON-like tridiagonal test
  (`tests/next_tests/integration_tests/multi_feature_tests/ffront_tests/test_icon_like_scan.py`)
  must return a *dummy bool field* because the `first_level` flag lives in the
  carry; conversely, a final carry (e.g. accumulated surface precipitation
  flux) is not retrievable without materializing a full 3D field.

- **P4 — Boundary conditions are clumsy.** Top/bottom special cases are
  encoded as flags in the carry plus a branch executed *at every level*, and
  the result must be repositioned by slicing the output at the program level:

  ```python
  class State(NamedTuple):
      z_q_new: float
      w_new: float
      first_level: bool


  @gtx.scan_operator(axis=KDim, forward=True, init=State(z_q_new=0.0, w_new=0.0, first_level=True))
  def _scan(state: State, w: float, z_q: float, z_a: float, z_b: float, z_c: float) -> State:
      z_g = z_b + z_a * state.z_q_new
      z_q_new = (0.0 - z_c) * z_g
      w_new = z_a * state.w_new * z_g
      return (
          State(z_q_new=z_q, w_new=w, first_level=False)
          if state.first_level
          else State(z_q_new=z_q_new, w_new=w_new, first_level=False)
      )


  # ... at the program level:
  _solve_nonhydro_stencil_52_like(..., out=(z_q[:, 1:], w[:, 1:], dummy[:, 1:]))
  ```

- **P5 — No support for scan + sub-reduce patterns.** Algorithms like
  explicit sedimentation in the Seifert–Beheng two-moment scheme, PPM flux
  integration, or ICON two-moment microphysics need, at each level, a
  *reduction over a bounded window of neighbouring levels* — sometimes inside
  a recurrence. Today this can only be emulated with hand-rolled ring buffers
  in the carry tuple (see companion document §C).

- **P6 — Production scans degenerate into one mega-operator.** The icon4py
  muphys graupel scheme (companion document §A.7) fuses *five* logically
  independent recurrences (rain/snow/ice/graupel sedimentation + temperature)
  into one `scan_operator` with a 21-component carry, because separate scans
  can neither share per-level intermediates nor be relied upon to fuse. Of
  those 21 components, 2 are pass-through output channels and one (`rho`) is
  a *delay line* — carried only because the body cannot read an input at
  k−1; a lookahead input is precomputed outside as
  `t_kp1 = concat_where(KDim < last_lev, t(Koff[1]), t)`. The scalar body
  also forces duplicated `*_scalar` twins of ordinary field operators, Python
  recursion-limit workarounds when tracing, and ~650 lines of hand-written
  DaCe hooks to clean up the generated code. The older icon4py
  `_icon_graupel_scan` shows the end state: a 28-tuple carry with an
  in-carry level counter, `sys.setrecursionlimit(350000)`, and 13 of 28
  outputs discarded at the call site.

## 2. Goals and non-goals

Goals:

1. Explicit, well-defined scan range, consistent with field-view domain
   propagation.
2. Fast embedded execution (array ops per level, not Python per element) and
   a direct mapping to `jax.lax.scan` for JAX-backed fields.
3. Boundary conditions at model top/bottom that are declarative, branch-free
   in the steady-state loop, and don't contaminate the carry.
4. Carry and per-level output as separate concepts; final carry retrievable.
5. A construct that covers bounded scan + sub-reduce algorithms
   (sedimentation, PPM) declaratively.
6. Direct access to neighbouring levels of *inputs* (k−1, k+1, …) without
   wiring delay lines through the carry or pre-shifting fields outside.
7. Small scans must be cheap: composable, sharing inputs, with fusion as a
   toolchain obligation — so nobody is forced to write a 21-component carry
   again.
8. A migration path: the old `scan_operator` expressible as sugar on top.

Non-goals (explicitly out of scope for the first iteration, see §4.7–4.8):

- Parallel-in-k execution (`associative_scan`-style lowering). The
  sequential-in-k / parallel-over-columns model is the production-proven one
  (GridTools, ICON-ACC); k-extents are ~80–137 while horizontal parallelism is
  abundant.
- Data-dependent trip counts (`while`-scans, early exit). Bounded, masked
  windows cover the known use cases on GPU-friendly terms.
- Distributed scans. The vertical axis is never distributed in our models.

## 3. Proposed design

### 3.1 Core primitive: a level-slice scan

The scan body is a function from *level slices* to *level slices*. A level
slice is a field over the horizontal domain of the call site (the K dimension
removed). The body takes the carry and one slice per scanned input, and
returns the new carry and the per-level output:

```
scan body:  (carry, x_k, ...) -> (carry', y_k)
scan call:  (xs..., init)     -> (final_carry, ys)
```

This is `jax.lax.scan` semantics (the de-facto ecosystem standard, also
adopted by PyTorch's higher-order `scan` and PyTorch/XLA — research §B.1,
§B.3), with the scan axis being the declared vertical dimension instead of the
leading array axis. Strawman syntax:

```python
@gtx.scan(axis=KDim, forward=True)
def cumsum_step(carry: float, x: float) -> tuple[float, float]:
    s = carry + x
    return s, s  # (new carry, output at this level)


@gtx.field_operator
def vertical_integral(
    x: gtx.Field[gtx.Dims[Cell, KDim], float],
) -> gtx.Field[gtx.Dims[Cell, KDim], float]:
    total, integral = cumsum_step(x, init=0.0)
    # `total`: Field[[Cell], float] — the final carry, e.g. a surface flux
    # `integral`: Field[[Cell, KDim], float]
    return integral
```

Semantics and type rules:

- **Slice typing.** Body arguments are level slices (or K-local views of the
  inputs, see below); the body is evaluated once per level on whole slices,
  not once per element. The annotation question is discussed below.
- **Carry structure** must be stable across iterations (same as JAX): the
  returned carry must have the type/structure of `init` after broadcasting.
- **`init` is a call-site argument** and may be a scalar, a tuple, or a
  *field* (a level slice, or anything broadcastable to one). A decorator
  default (`init=0.0`) remains possible for closed algorithms like `cumsum`.
- **Control flow** in the body: data-dependent branching is written with
  `where` (per-element) — Python `if` on field values is rejected at type
  checking, consistent with `field_operator` rules. Conditions on the *scan
  index only* get special treatment (§3.3).
- **Backward scans** keep the `forward=` flag; outputs stay index-aligned
  with inputs regardless of direction (as in `lax.scan(reverse=True)`).
- **Simple form (sugar).** A body that returns only the carry
  (`(carry, x) -> carry`) is interpreted as `y = carry'`, and the call returns
  just `ys`. This is the migration target for today's `scan_operator` and
  covers the dominant carry-is-output case without boilerplate.

#### Typing of body arguments, and access to neighbouring levels

Two coupled questions: what do the annotations *say*, and how does the body
read an input at k−1 or k+1? Today the answer to the second is "you don't":
muphys carries `rho` one level in the carry as a delay line and precomputes
`t_kp1` outside the scan with a clamped `Koff[1]` shift (P6) — previous-level
access wired through state is exactly the pain to remove. Options considered
for the first question:

- *(a) Element-type-as-slice*: annotate `x: float`, meaning "level slice of
  whatever horizontal type the caller passes". Terse and polymorphic, but the
  annotations are not the types the body actually computes with — a milder
  re-run of today's promotion dishonesty (P2), and a break with field-view
  style where annotations are real types.
- *(b) Honest field annotations*: `x: gtx.Field[Dims[Cell], float]` (or,
  with dimension variables, `gtx.Field[Dims[*Hs], float]`). Honest, but
  either loses horizontal polymorphism or requires new variadic-generic
  type machinery; and it still answers only the "what is x at level k"
  question, not level access.
- *(c) **K-local views** (proposed)*: scanned inputs arrive as views of the
  input column positioned at the current level, indexable by *relative*
  K-offset; indexing yields a level slice. Carry and outputs remain plain
  values.

```python
@gtx.scan(axis=KDim, forward=True)
def step(carry: float, t: gtx.KView[float], rho: gtx.KView[float]) -> float:
    # t[0]: slice at the current level; t[+1]: lookahead; rho[-1]: previous level
    flux = f(carry, t[0], t[+1], rho[0], rho[-1])
    return flux
```

This is, deliberately, *iterator/local-view style along K only* — "field
view horizontally, local view vertically". The justification: a scan is the
one construct with an inherent current position, and the local view is the
natural type at such a position. It is not a new concept in the stack: the
ITIR scan pass *already* receives iterators (shiftable and dereferenceable —
see e.g. the scan in
`tests/next_tests/unit_tests/iterator_tests/transforms_tests/test_domain_inference.py`,
which shifts its argument before deref); only the ffront `scan_operator`
flattens them to scalars. GTScript's `field[0, 0, -1]`, Halide's update
definitions, and GridTools fill-type k-caches are the same idea (research
§B.6, §B.8). Semantics and lowering:

- Offsets are static; the toolchain infers the *vertical extent* per input
  (cartesian-style extent analysis) and the scan range shrinks (or masks) at
  the column ends accordingly — same rules as `Koff` shifts.
- Inputs are read-only, so **lookahead (`t[+1]`) in a forward scan is
  legal** — muphys's externally precomputed `t_kp1` becomes `t[+1]` in the
  body. Reading *outputs* at an offset stays a carry concern (§4.6).
- Compiled backends map views to fill-type k-caches / registers; embedded
  maps `x[o]` at level k to the slice at k+o (still one array op per level);
  the JAX lowering passes one extra pre-shifted `xs` leaf per distinct
  offset — mechanical, `lax.scan` semantics preserved.
- Sugar: a parameter annotated with an element type (`x: float`) is an
  auto-dereferenced view, i.e. shorthand for `x[0]`-only access. Simple
  bodies (`cumsum_step` above) stay as terse as today, and option (a)
  becomes a *documented projection* of (c) rather than a lie: the annotation
  names what `x[0]` yields.

We recommend (c) with the (a)-style sugar. The honest-field-types option (b)
remains compatible (an author may pin concrete slice types) but is not
required.

A real example — the ICON-like tridiagonal forward sweep from P3/P4,
rewritten (boundary handling moved out, see §3.3):

```python
@gtx.scan(axis=KDim, forward=True)
def w_fwd_sweep(
    carry: tuple[float, float],
    z_q: float,
    w: float,
    z_a: float,
    z_b: float,
    z_c: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    z_q_km1, w_km1 = carry
    z_g = z_b + z_a * z_q_km1
    z_q_new = -z_c * z_g
    w_new = z_a * w_km1 * z_g
    return (z_q_new, w_new), (z_q_new, w_new)
```

No `first_level` flag, no dummy output, no per-iteration branch.

### 3.2 Scan range: deduced from inputs, explicitly overridable

The scan range is the intersection of the K-ranges of the scanned inputs —
the same rule as for any other field expression — optionally overridden:

```python
_, (z_q_res, w_res) = w_fwd_sweep(
    z_q, w, z_a, z_b, z_c, init=(z_q0, w0), k_range=KDim[1:]
)  # strawman override syntax
```

- If all inputs carry K, no annotation is needed: the range is where the
  inputs are defined. This removes the context-variable plumbing and the
  output-domain back-inference entirely.
- A scan with no K-bearing input (pure recurrence, `xs=None` in JAX terms)
  *must* specify the range explicitly.
- The program-level `domain=` argument restricts the *output* as today; the
  scan range itself no longer depends on it. Restricting the output below the
  scan range is allowed (compute, then write a sub-range); requesting output
  outside the scan range is an error.

This is a semantic change: today's "scan exactly over the requested output"
becomes "scan over the defined inputs, write the requested output". For the
practically relevant cases (output range == input range, or output is a
sub-range as in the `[:, 1:]` slicing idiom) the observable behaviour is
unchanged, but it is now *predictable from the call site*.

### 3.3 Boundary conditions

Three complementary mechanisms, ordered by how often we expect them:

1. **Peelable index conditionals** — the workhorse, covering boundary values
   computed from the scanned fields themselves (the ICON-like case).
   `concat_where` on the scan index is allowed *inside* the body:

   ```python
   @gtx.scan(axis=KDim, forward=True)
   def step(carry: float, x: float) -> float:
       return concat_where(KDim == 0, bc_expr(x), recurrence(carry, x))
   ```

   Because the condition depends only on the loop index, the toolchain
   guarantees *loop peeling*: compiled backends split the k-loop (the
   GridTools/Dawn vertical-interval model, Halide loop partitioning —
   research §B.5, §B.6), and embedded execution hoists the branch out of the
   level loop. The user writes one function; no branch survives in the
   steady-state loop; no level-slicing of inputs and no output stitching is
   needed — the peeled first iteration both seeds the carry and writes the
   first output level. This subsumes the cartesian `interval(...)` mechanism
   in field-view clothing and reuses an existing builtin. (An earlier draft
   added an `at_level(f, KDim, 0)` accessor to feed first-level values into
   `init`; it is unnecessary — this mechanism covers that case.)

2. **Field-valued `init`, optionally emitted — inclusive/exclusive scans.**
   When the boundary value originates *outside* the scanned fields (surface
   pressure from another component, a prescribed model-top flux), it enters
   as the `init` argument: a scalar or a k-less field slice, evaluated
   exactly once. `include_initial=True` additionally *emits* the init,
   extending the output range by one level upstream of the scan range —
   the array-API `cumulative_sum(include_initial=...)` design (research
   §B.2):

   ```python
   # hydrostatic-like integration: n cell increments -> n+1 interface values
   _, p = pressure_step(dz_rho_g, init=p_top, include_initial=True)
   ```

   This is where inclusive vs exclusive scans enter. With
   `y[k] = f(y[k-1], x[k])` and `y[k0] = init`, the "BC at the first level"
   pattern *is* an inclusive scan over `K[k0+1:]` whose init is emitted at
   `k0` — i.e. `include_initial` is the general form of mechanism 1's
   output, and the *exclusive* scan (`y[k] = carry before consuming x[k]`)
   is the same primitive with the last level dropped. Both ship as library
   wrappers (§3.5: `exclusive_cumsum`, `scan_with_initial`) rather than as
   core variants. The emitted level is also the natural producer of
   *staggered* outputs — fluxes on interfaces (n+1) from cell quantities
   (n), where "the flux at the model top is the boundary condition" — and
   should compose with however field view ends up modelling half-level
   dimensions (open question §5).

3. **Composition outside the scan** with `concat_where` over K regions —
   possible because the scan range is explicit (§3.2). This replaces the
   program-level `out=(z_q[:, 1:], ...)` slicing idiom and remains the
   escape hatch for boundary regions thicker than one level with genuinely
   different algorithms.

### 3.4 Vertical windows and scan + sub-reduce

The second new construct targets algorithms of the form

```
out[k] = reduce_{m in window(k)} f(inputs at k, k-m, k-m±1, ...)
```

— explicit sedimentation (COSMO/ICON two-moment), PPM flux integration,
ICON two-moment microphysics. The key insight from the case study (companion
document §C): when the window extent is bounded by a compile-time constant
`M` (in sedimentation: the maximum vertical Courant number; in PPM: the
maximum departure-cell count), the pattern is **not a scan at all** — it is a
vertical stencil with a static window plus a reduction, and the recurrence
that *does* appear (cumulative geometric height) is a separate, ordinary
`cumsum`. Moreover the physics formulas typically self-mask: contributions
from levels out of reach evaluate to exactly zero.

We therefore propose **vertical window offsets**, reusing the existing
unstructured machinery (local dimensions, sparse fields, `neighbor_sum`,
skip-value masking) rather than inventing a new looping construct. A window
offset is a compile-time "connectivity" along K:

```python
KWin = gtx.WindowOffset("KWin", source=KDim, offsets=range(0, -4, -1))  # k, k-1, k-2, k-3
KWinUp = gtx.WindowOffset(
    "KWinUp", source=KDim, offsets=range(-1, -5, -1), local_dim=KWin.local_dim
)  # k-1 ... k-4
```

`phi(KWin)` yields a field with an extra local dimension of static size 4,
entry `m` holding `phi[k + offsets[m]]` — exactly like a sparse neighbor
field. The COSMO "new explicit scheme" (research §C.1, eq. 12) then reads:

```python
@gtx.field_operator
def box_sedim_flux(v: CellKF, z: CellKF, phi: CellKF, dt: float) -> CellKF:
    z_up = minimum(z(KWinUp), z + abs(v(KWin)) * dt)  # z broadcasts over the window dim
    z_low = minimum(z(KWin), z_up)
    return -neighbor_sum(phi(KWin) * (z_up - z_low), axis=KWin.local_dim)
```

which is a direct, vectorized transcription of the reference implementation
(`box_sedim_naive_func` in the gist analysed in §C.2) — no ring buffer, no
carry, no index arithmetic.

Design points:

- **Boundaries.** Window entries reaching outside the field's K-range are
  handled like connectivity skip values: masked out of `neighbor_sum`
  (neutral element). Alternatively the output domain shrinks by intersection,
  as for ordinary `Koff` shifts; masking is the right default here because
  the physics wants "no contribution from above the model top".
- **Inside scans.** Access to *individual* offsets inside a scan body is
  already covered by K-local views (§3.1: `x[-2]`, `x[+1]`); the window
  construct adds the *reduction over the window* (`neighbor_sum` over the
  local dim) and is allowed on view arguments too (`x(KWin)`), for
  algorithms where the windowed contribution interacts with running state.
  Backends lower both to fill-type k-caches / ring buffers (the GridTools
  k-cache model, research §B.6); embedded lowers to shifted array views.
  This is precisely the rolling-buffer transformation done by hand in the
  gist (`box_sedim_scanlike`), now performed by the toolchain.
- **Data-dependent extents** are expressed as a static bound plus masking
  (`where`/self-masking formulas), the standard GPU-friendly form. The
  work-efficient scatter formulation of the COSMO paper is intentionally
  *not* supported: it writes to multiple output levels per step, which
  breaks the one-output-slice-per-level model and any future parallel-in-k
  option (analysis in §C.3–C.4).
- **Reading the in-progress output** (`ys(Koff[-1])`, GTScript/Halide style)
  is *not* part of the core primitive: an order-m recurrence is expressible
  with an m-tuple (or windowed) carry. We keep offset-output-reads open as
  later sugar that desugars to a window carry (§4.6).

### 3.5 Library layer

With the primitive in place, ship the common cases as library functions so
users rarely write raw scans:

- `gtx.lib.cumsum(x, axis=KDim, forward=True, include_initial=False)` and
  `gtx.lib.exclusive_cumsum(...)` — lowered to `xp.cumulative_sum` in
  embedded/array backends (array API 2023.12), to a trivial scan elsewhere.
- `gtx.lib.scan_with_initial(step, xs, init)` — the §3.3.2 pattern
  (inclusive scan with emitted init) as a named wrapper.
- `gtx.lib.tridiagonal_solve(a, b, c, d)` — the Thomas algorithm (forward
  sweep + backward substitution) as *one named operation*. Naive backends
  lower it to two scans; optimized GPU backends may use vendor libraries
  (cuSPARSE `gtsv2`-family) or per-column Thomas in registers. Precedent:
  `jax.lax.linalg.tridiagonal_solve`, LFRic's columnwise (CMA) operators
  (research §B.7). This removes the single most awkward scan use case from
  user code and pins down its numerics.
- `gtx.lib.windowed_scan(...)` — convenience wrapper combining §3.1 + §3.4
  for bounded-history recurrences.

### 3.6 Execution model per backend

| Backend             | Lowering                                                                                                                                                                                                                                                    |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Embedded NumPy/CuPy | Python loop over k (~100 iterations), one vectorized slice expression per level. Replaces O(n_h · n_k) scalar calls with O(n_k) array calls.                                                                                                                |
| Embedded JAX        | Direct `jax.lax.scan` over the K axis (body is already slice-level). Makes scans `jit`/`vmap`/`grad`-compatible — vertical solvers become differentiable for free.                                                                                          |
| GTFN                | Unchanged execution model: the slice body is scalarized per column inside the existing `vertical_executor` k-loop (a slice function applied per column *is* today's column scan). Peelable conditionals map to interval splitting; windows map to k-caches. |
| DaCe                | As today: sequential loop region nested in horizontal maps (`gtir_to_sdfg_scan.py`); carry/output separation removes the dummy-output temporaries.                                                                                                          |

The frontend simplifies: `_scan_param_promotion` and the scalar↔field
special-casing in `ffront/type_info.py` disappear; the ITIR `scan` builtin
gains an explicit init/range and a carry/output pair instead of the
carry-is-output convention.

### 3.7 Migration

- `@gtx.scan_operator(axis=..., forward=..., init=...)` is kept for one
  deprecation cycle as sugar: scalar body interpreted as simple-form slice
  body (Python `if` on data values rewritten to `where` where possible,
  otherwise a deprecation error with a fix-it hint).
- The `out=field[:, 1:]` slicing idiom keeps working (it is just an output
  restriction under §3.2 semantics).
- Tests in `tests/next_tests/**/ffront_tests/` that exercise scans
  (`test_icon_like_scan.py`, `test_execution.py::*scan*`) are the migration
  spec; each gets a rewritten twin before the old path is removed.

### 3.8 Composability and scan fusion

The muphys evidence (P6, research §A.7) shows what happens when small scans
are not viable: users fuse by hand. The design response has four parts:

1. **Small scans are cheap to write.** Slice-level bodies call ordinary
   field operators directly — the duplicated `*_scalar` helper twins in
   icon4py (`_fall_speed_scalar`, `_T_from_internal_energy_scalar`, manually
   inlined "to avoid scan_operator -> field_operator") disappear. Carry
   size shrinks to the genuinely recurrent state: K-local views remove the
   delay-line slots (`rho` at k−1, the `*_kup` caches of the old
   microphysics), carry/output separation removes the pass-through slots,
   and index conditionals remove in-carry level counters and `activated`
   first-iteration flags.
2. **Fusion is the toolchain's obligation, not the user's.** Scans over the
   same axis, direction, and range whose data dependencies are at the same
   level or carried (the muphys temperature scan consumes the species
   scans' same-level flux outputs) are fusable into one k-loop. GTFN
   already merges scans at the stencil level; DaCe fuses loop regions. The
   redesign turns this from incidental into specified: *the muphys
   decomposition — four species scans plus one temperature scan, written
   separately — must compile to a single vertical loop*. That statement
   belongs in the acceptance tests of the rewrite.
3. **Surface outputs are final carries.** muphys extracts `pr, ps, pi, pg, pre` by slicing the last level of full-column outputs at the program
   level; with §3.1 these are simply components of the returned final carry
   (k-less fields), allocated as such.
4. **Work-skipping becomes a transformation concern.** The muphys
   `if current_level_activated:` (skip all physics where nothing is active,
   with an identity else-branch) is a per-column, data-dependent
   optimization that today leaks into user code *and* requires ~650 lines
   of hand-written DaCe hooks to remove the resulting self-copies. In the
   slice world it is written as `where`-masking; sparsity/masking
   optimizations (including "else-branch is identity" detection) move into
   the toolchain, where the muphys TODO already wants them ("Can be made
   unnecessary with future transformations").

## 4. Discussion: decisions, pros and cons, alternatives

### 4.1 Body granularity: scalar vs level slice vs whole column

|                 | scalar element (status quo)                  | **level slice (proposed)**        | whole column (LFRic-style)         |
| --------------- | -------------------------------------------- | --------------------------------- | ---------------------------------- |
| embedded speed  | Python per element — unusable for real sizes | array op per level                | array op per column batch          |
| JAX mapping     | none                                         | `lax.scan` exactly                | possible but manual                |
| per-column `if` | natural                                      | `where` only                      | natural                            |
| type system     | promotion/demotion magic                     | plain broadcasting                | needs column types                 |
| backend codegen | per-column loop (natural for GTFN)           | scalarize per column (mechanical) | compiler must rediscover structure |

The slice body loses scalar `if` — the one real regression. We accept it:
(a) per-column data-dependent branching expresses as `where`; (b) every
array ecosystem (JAX, PyTorch, Futhark on GPU) made the same trade; (c) the
scalar form survives as migration sugar. The muphys
`if current_level_activated:` work-skipping branch is the strongest
counter-example — but note that on GPUs the scalar skip only pays when a
whole warp of columns is inactive, and its identity else-branch is what the
hand-written DaCe cleanup hooks exist to repair (§3.8): the branch is better
treated as a masking *transformation* than as user-level control flow.
Whole-column kernels (write the k-loop yourself) were rejected: maximal
freedom, but the toolchain loses the recurrence structure (no interval
splitting, no k-caching, no `lax.scan` mapping, no future associative
lowering).

### 4.2 Carry/output separation

Pro: removes dummy outputs (P3), gives final carries (surface fluxes),
matches JAX/PyTorch signatures, lets the Thomas sweep carry `(c', d')` while
emitting what the back-substitution needs. Con: one more tuple level in the
general form; mitigated by the simple form (§3.1). Rejected alternative:
keep carry-is-output and add a separate "final carry" accessor — fixes only
half of P3 and diverges from the ecosystem signature.

### 4.3 Range from inputs + override

Pro: consistent with all other field-view ops; call-site-predictable;
removes context-variable plumbing; makes scans composable with
`concat_where`. Con: pure recurrences need an explicit range (acceptable:
they have no other source of truth); behaviour change for exotic cases where
the output domain exceeded input domains (today: error-prone implicit;
proposed: explicit error). Rejected alternative: keep output-domain deduction
— it is exactly the "not clear how this should be done" problem.

### 4.4 Boundary conditions: init + peelable conditionals vs interval dispatch

We considered GTScript-style interval dispatch as API, e.g.
`@gtx.scan(intervals={KDim[0:1]: body0, KDim[1:]: body})`. Pro: explicit,
proven in cartesian. Con: multiplies function definitions for what is usually
one differing expression; interacts badly with carry typing (every body must
agree); and `concat_where`-in-body plus field `init` express the same thing
with existing vocabulary. The peeling *guarantee* (not just optimization) is
the load-bearing part — it must be stated in the spec and tested, otherwise
users are back to per-iteration branches.

### 4.5 Windows via local dimensions vs dedicated loop construct

Reusing sparse-field machinery means: no new IR concepts, embedded gets it
almost for free (stacked shifted views), masking semantics inherited from
skip values, and the construct works *outside* scans too (where most uses
actually live — §C insight). Con: window size is compile-time static; two
window offsets sharing a local dimension (the `KWin`/`KWinUp` pair) need a
small extension to offset declarations. Rejected alternative: a bounded
`for m in range(M)` loop in the body — more familiar, but introduces general
loops into the DSL with all their analysis burden, for no added expressivity.

### 4.6 Offset reads of the in-progress output

K-local views (§3.1) cover offset reads of *inputs*. For the *in-progress
output* — GTScript's `field[0,0,-1]` in FORWARD computations, Halide's
update definitions — we keep offset reads out of the core: the carry covers
them, and explicit state is easier to type, lower, and differentiate. They
remain reserved as sugar desugaring to a window carry. Revisit when
migrating cartesian users.

### 4.7 Parallel-in-k: deliberately not in v1

Linear recurrences and tridiagonal sweeps are associative-reformulable
(Blelloch; `lax.associative_scan`; PyTorch `associative_scan` — research
§B.1, §B.3, §B.8), but: NWP k-extents are tiny (~100), horizontal parallelism
saturates GPUs (the ICON/GridTools production model is sequential-k), the
reformulation changes floating-point results (reproducibility concerns), and
PyTorch's still-prototype status after 2+ years shows the compilation cost.
The design keeps the door open: the primitive's semantics are sequential; an
`associative=True` hint or pattern recognition (cumsum, linear recurrence)
can later unlock parallel lowering without user-visible change — the
Futhark `scan`/`loop` split and Halide `rfactor` are the precedents.

### 4.8 `while`-scans: rejected

Data-dependent trip counts (the COSMO paper's work-efficient inner `while`)
don't vectorize over columns, lose `vmap`/AD in JAX, and complicate every
backend. The masked bounded window is the vectorized form production codes
use anyway (the paper's own NEC-vectorized variant effectively does this).
Users with genuinely unbounded extents pre-chunk or fall back to multiple
passes.

## 5. Open questions

1. **K-view annotation spelling and honesty.** `gtx.KView[float]` strawman;
   whether the `x: float` auto-deref sugar (§3.1) keeps annotations honest
   enough for frontend style, or whether views should be mandatory in
   signatures whenever offsets are used.
2. **Staggered outputs of `include_initial`.** The emitted-init level
   produces n+1 outputs from n inputs — interface vs cell-center fields.
   How this interacts with half-level dimensions (icon4py's `KHalfDim`
   practice) and with §3.2 range deduction needs a dedicated design.
3. **Range override syntax.** `k_range=KDim[1:]` vs reusing the domain
   expression API from `concat_where` (`KDim > 0`) vs program-style named
   ranges. Should align with whatever domain-slicing syntax field view
   converges on.
4. **Simple-form detection.** Return-annotation-based (`-> float` vs
   `-> tuple[carry, y]`) is implementable in the traced frontend but subtle
   for tuple carries; an explicit flag (`emit="carry"`) may be safer.
5. **Fusion guarantees.** §3.8 makes scan fusion an obligation — but spec
   or best-effort? Which compositions are *guaranteed* (same range/direction,
   same-level coupling) vs merely attempted (different ranges, opposing
   directions as in Thomas forward+backward)? Needs to be pinned before
   users rely on decomposed scans being performance-neutral.
6. **Window declaration ergonomics.** Final spelling of `WindowOffset`,
   shared local dims, and interaction with `offset_provider` (these are
   compile-time, not runtime, connectivities).
7. **Naming.** `gtx.scan` vs keeping `scan_operator`; `KView` vs
   `ColumnView`; `WindowOffset` vs `KWindow`; `gtx.lib` vs `gtx.stdlib`.
8. **NamedCollection carries.** Today's `NamedTuple` carries (muphys's
   nested `IntegrationState`) should keep working in the slice world
   (collections of slices); verify against `arguments.extract` handling.

## 6. Consequences

Easier: writing and reading vertical solvers (no flag-in-carry idioms, no
delay lines, no `*_scalar` helper twins); embedded debugging at realistic
sizes; JAX differentiation of vertical physics; expressing
sedimentation/PPM declaratively; reasoning about scan domains; decomposing
muphys-scale schemes into per-process scans without a performance penalty
(given §3.8). Tracing also shallows out: slice bodies calling field
operators replace the deeply recursive scalar expansion behind icon4py's
`sys.setrecursionlimit` workarounds. Harder/cost: a frontend+IR+all-backends
change with a deprecation cycle; per-column `if` users must move to `where`
(work-skipping becomes a transformation, §4.1); the peeling guarantee,
K-view extents, window lowering, and the fusion obligation are new backend
responsibilities (GTFN intervals, scan merging, and k-caches already exist;
DaCe needs loop-region splitting and fusion).

If accepted, the decisions in §3.1–§3.4 and §3.8 should each land as a
numbered ADR under `docs/development/ADRs/next/` referencing this proposal;
this document stays as the design rationale.
