# gt4py.next test-structure cleanup

## Context

`tests/next_tests/` has grown organically. The top-level taxonomy (unit / feature /
multi-feature / regression / benchmark) is sound, but several things have drifted:

- **`integration_tests/feature_tests/ffront_tests/test_execution.py` (1469 lines, 68 tests)**
  is a catch-all. It is the *residue* of feature tests whose feature never got its own
  filename — scans, tuples, astype, domain, scalar args, shifts all live there mixed
  together, while siblings like `test_where.py`, `test_concat_where.py`, `test_scalar_if.py`
  already split their feature out. The file encodes nothing on the feature axis.
- Several backend-independent tests in it (`test_docstring`, `test_undefined_symbols`,
  `test_constant_closure_vars_*`) take `cartesian_case` and therefore run across the **full
  ~12-backend matrix** for no reason.
- Tree hygiene smells: a directory literally named `ir_utils_test.py/`; a bare `dace/` dir
  (vs the `*_tests/` convention); a tracked `artifacts/dummy_package/__init__ .py` with a
  **space** in the name (the package only resolves today via implicit-namespace-package
  semantics); `regression_tests/` has no `__init__.py` at any level; stray ignored
  `dummy_package/` and `.mypy_cache/` dirs in the tree.
- Shared helpers (`cases.py`, `definitions.py`, `ffront_test_utils.py`,
  `math_builtin_test_data.py`, …) sit at inconsistent levels with no documented convention.

Goal: give the tests a principled structure along clear axes, starting with the catch-all,
without losing any test or breaking the ADR-15 backend-exclusion matrix.

## The axis framework (the organizing principle)

Orthogonal axes should use orthogonal mechanisms. **Folders + filenames encode the static
"what is this test about" (scope → module/feature → aspect). Markers + fixtures encode the
dynamic matrix a single test is run across (backend, grid, capability, hardware).**

| Axis | Mechanism (correct home) | Current state |
|---|---|---|
| Test scope/level (unit / feature / multi-feature / regression / benchmark) | top-level **folder** | ✓ good — keep |
| Source module (ffront, iterator, otf, type_system, embedded, errors, program_processors, instrumentation) | **subfolder** mirroring `src/gt4py/next/` (unit tests) | ✓ mostly; bug: `ir_utils_test.py/` dir |
| **DSL feature** (scan, tuples, where, concat_where, reductions, math, astype, domain, shifts, if/ternary, broadcast) | **filename** in feature_tests **+** `uses_*` **marker** | ✗ **inconsistent — the `test_execution.py` problem** |
| Backend / exec target (embedded numpy/cupy/jax, gtfn cpu/gpu/imperative, dace cpu/gpu, roundtrip, gtir, formatters) | **fixture parametrization** (`exec_alloc_descriptor`/`program_processor`) + nox `-m` filters | ✓ good — never make folders for this |
| Backend-capability gating | **`uses_*` marker** → ADR-15 `BACKEND_SKIP_TEST_MATRIX` | ✓ good |
| Grid type (cartesian / unstructured) | **fixture** (`cartesian_case`/`unstructured_case`); filename only where the mechanism truly differs (shifts) | ✓ mostly |
| Mesh variant (simple / skip-value) | **fixture param** (`mesh_descriptor`) | ✓ good |
| Frontend stage (foast / past / itir / gtir / codegen) | subfolder + filename (unit tests) | ✓ good |
| Test aspect (execution-correctness / error-raising / type-deduction / introspection) | folder (unit vs feature) + filename | ✗ partly — error/introspection tests leak into the execution file |
| Hardware / optional deps (gpu, jax, dace, atlas) | **`requires_*` marker** | ✓ good |

**Placement rule for the feature axis (decided):**
1. One file per *basic* feature.
2. Interaction tests are placed by their **subject under test** — the capability whose
   regression the test is designed to catch (the supporting features are just the vehicle).
   Tie-breakers, in order: feature in the test's own name → the more advanced/less-mature
   feature (scan > reduction > tuple > astype > basic) → the feature whose marker is the
   backend-gating one.
3. **Every moved test keeps the full `uses_*` marker set** for the features it exercises.
   Markers are the cross-cutting index: `pytest -m "uses_scan and uses_tuple_returns"`
   finds scan×tuple tests regardless of file. This makes step 2 low-stakes.
4. Exception where grid type beats feature: **shifts/reductions** — cartesian shift and
   unstructured shift+neighbor-reduction are near-disjoint mechanisms, so grid type is the
   natural file boundary there. Do **not** split cartesian/unstructured globally.

---

## Phase 1 — Carve up `test_execution.py` (the core change)

Target file set in `tests/next_tests/integration_tests/feature_tests/ffront_tests/`.
New files created by moving test functions out of `test_execution.py` (apply the placement
rule above; representative assignments below, not an exhaustive per-test list):

- **`test_scan.py`** (NEW, ~11) — subject = scan. `test_scalar_scan`,
  `test_tuple_scalar_scan`, `test_scalar_scan_vertical_offset`, `test_fieldop_from_scan`,
  `test_solve_triag`, `test_ternary_scan`, `test_scan_nested_tuple_{output,input}`,
  `test_scan_different_domain_in_tuple`, `test_scan_tuple_field_scalar_mixed`,
  `test_scan_unused_parameter`. (tuple/ternary here are the vehicle → stay in scan.)
- **`test_tuples.py`** (NEW, ~14) — tuple args/returns/unpacking: `test_tuples`,
  `test_multicopy`, `test_*_tuple_arg*`, `test_tuple_return_2`, `test_nested_tuple_return`,
  `test_tuple_unpacking*`.
- **`test_astype.py`** (NEW, ~8) — `test_astype_*`, `test_type_constructor_alias`,
  `test_astype_int_local_field` (astype×unstructured → subject is astype).
- **`test_domain.py`** (NEW, ~6) — `test_domain*`, `test_domain_tuple`,
  `test_single_value_field`, `test_scalar_in_domain_spec_and_fo_call`.
- **`test_scalar_args.py`** (NEW, ~5) — `test_scalar_arg`, `test_np_bool_scalar_arg`,
  `test_nested_scalar_arg`, `test_double_use_scalar`, `test_scalar_arg_with_field`.
- **`test_cartesian_shifts.py`** (NEW, ~3) — `test_cartesian_shift`, `test_fold_shifts`,
  `test_offset_field` (dynamic offsets / `as_offset`).
- **Unstructured shift + neighbor reductions** → fold into existing **`test_gt4py_builtins.py`**
  (already holds `neighbor_sum`/`max_over`/`min_over`), or rename it `test_reductions.py` and
  move its `broadcast` test to basics. Moves: `test_unstructured_shift*`,
  `test_composed_unstructured_shift`, `test_horizontal_only_with_3d_mesh`,
  `test_neighbor_sum_with_non_zero_origin`, `test_nested_reduction*`,
  `test_tuple_with_local_field_in_reduction_shifted`, `test_ternary_builtin_neighbor_sum`,
  `test_local_index_premapped_field`.
- **Ternary** → fold `test_ternary_operator`, `test_ternary_operator_tuple` into
  `test_scalar_if.py`, renamed **`test_conditionals.py`** (if-statements + ternary together).
- **`test_basic.py`** (rename the shrunken `test_execution.py`) — genuine basics only:
  `test_copy`, `test_infinity`, `test_nan`, `test_zero_dims_fields`,
  `test_implicit_broadcast_mixed_dim`.
- **Move OUT of the feature matrix** (backend-independent — they take `cartesian_case` and
  needlessly fan out over ~12 backends): `test_docstring`, `test_undefined_symbols`,
  `test_constant_closure_vars_with_frozen_namespace`, `test_constant_closure_vars_with_enums`
  → `tests/next_tests/unit_tests/ffront_tests/` (decorator/closure-capture + error aspect).
  Verify each truly doesn't need a backend (e.g. `test_undefined_symbols` may want one
  backend to raise — if so, pin to roundtrip, don't run the matrix).

Mechanics: use `git mv`-free moves (cut/paste test bodies; `git` tracks content), keep the
license header + shared import block in each new file, prune now-unused imports from the
shrunken file. Do not alter test bodies or markers — only relocate.

## Phase 2 — Tree hygiene (renames / packaging)

- `unit_tests/iterator_tests/ir_utils_test.py/` → **`ir_utils_tests/`** (a dir whose name
  ends in `.py`); add `__init__.py`. Mirrors `src/.../iterator/ir_utils/`. Contains
  `test_domain_utils.py`, `test_misc.py`.
- `integration_tests/feature_tests/dace/` → **`dace_tests/`** (matches the `*_tests/`
  convention; nothing imports it by path, confirmed — safe).
- `artifacts/dummy_package/__init__ .py` → **`__init__.py`** (remove the space; it is a
  tracked typo, the package works only via namespace-package fallback today). Imported by
  `unit_tests/type_system_tests/test_type_translation.py` and
  `feature_tests/ffront_tests/test_import_from_mod.py` — verify both still pass.
- `regression_tests/` — add `__init__.py` at `regression_tests/`,
  `regression_tests/ffront_tests/`, `regression_tests/embedded_tests/` (every other tier is
  a package). Keep the tier (it pins specific past-bug repros), just make it consistent.
- Stray ignored dirs: remove the empty `tests/next_tests/dummy_package/` and the
  `multi_feature_tests/ffront_tests/.mypy_cache/`; confirm both are covered by `.gitignore`
  (`norecursedirs` already excludes `.*`). No test writes `dummy_package/`, so it is a stale
  local artifact.

## Phase 3 — Helper convention + taxonomy + docs

Conservative on churn: do **not** relocate the two heavily-imported framework modules
(`next_tests.definitions`, `next_tests.integration_tests.cases`) — moving them rewrites
imports in ~100 files for no functional gain. Instead, establish and document the convention,
and move only genuinely-misfiled loose helpers:

- **Convention** (document in `src/gt4py/next/AGENTS.md` testing section):
  - `fixtures/` → pytest fixture modules.
  - `artifacts/` → static test data and sample packages.
  - `definitions.py` (root) and `integration_tests/cases.py` → the two canonical framework
    modules, kept at their established import paths.
  - Single-consumer reference setups (`fvm_nabla_setup.py`, `hdiff_reference.py`) → stay
    co-located with their one test.
- **Move** `integration_tests/feature_tests/math_builtin_test_data.py` → `artifacts/`
  (it is static data, currently loose in the feature dir); update its one importer.
- **Audit** the unit-vs-feature `instrumentation_tests/` and `otf_tests/` overlap. The two
  `test_gpu_profiler.py` files differ (confirmed) → keep both but ensure scopes don't
  overlap; merge only if redundant.
- **Taxonomy verdict:** keep `unit / feature / multi_feature / regression / benchmark`. It is
  the scope axis and works; the only fix is making `regression_tests/` a proper package
  (Phase 2). No top-level restructuring.

## Verification

Run after **each** phase (fast loop first, matrix last):

1. **No test lost / renamed away:** capture `uv run pytest tests/next_tests --collect-only -q
   | tail -1` (total count) before Phase 1 and after each phase — counts must match (moves)
   or change only by the intended `cartesian_case`→unit reduction (those 3–4 tests drop from
   ~12× to 1×).
2. **Markers still index the moves:** `uv run pytest tests/next_tests -m "uses_scan"
   --collect-only -q` and `-m "uses_tuple_returns"` list the relocated tests.
3. **Targeted runs** on a cheap backend, e.g.
   `uv run pytest tests/next_tests/integration_tests/feature_tests/ffront_tests/test_scan.py -q`
   and the moved unit tests in `unit_tests/ffront_tests/`.
4. **Renames resolve:** `uv run pytest --collect-only
   tests/next_tests/unit_tests/iterator_tests/ir_utils_tests/` and the renamed `dace_tests/`;
   run `test_type_translation.py` + `test_import_from_mod.py` after the `__init__.py` fix.
5. **Boundaries / QA:** `uv run tach check`; `uv run pre-commit run` on staged files.
6. **Matrix confidence (final):** `uv run nox -s "test_next-<py>(...)"` (gpu/dace sessions
   may skip locally).

Land as **one PR per phase** (Phase 1 is the high-value core; Phases 2–3 are independent and
low-risk) to keep each diff reviewable. Suggested titles:
`refactor[next]: split ffront test_execution.py into feature files`,
`refactor[next]: test-tree hygiene (renames, packaging)`,
`refactor[next]: document test-helper convention`.

## Out of scope / deferred

- No changes to `BACKEND_SKIP_TEST_MATRIX`, fixtures, or `cases.py` framework semantics.
- No new `common_tests/` / `experimental_tests/` (separate coverage effort, not a cleanup).
- No relocation of `definitions.py` / `cases.py` (import-churn not worth it).
