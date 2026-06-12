# Worked examples: vertical scan redesign

- **Status**: companion document to
  [scan-redesign-proposal.md](scan-redesign-proposal.md) and
  [scan-redesign-research.md](scan-redesign-research.md)
- **Created**: 2026-06-12

Before/after code for the main scan use cases under the proposed design.
All "proposed" code uses the strawman syntax of the proposal (§3.1–§3.4,
§3.8) and is **illustrative, not validated**: names, the `k_range=` spelling,
and `KView` indexing are open questions (§5 of the proposal). The physics in
the muphys sketch is transcribed for structure, not correctness.

## 1. Cumulative sum, and the final carry as a surface diagnostic

Today:

```python
@gtx.scan_operator(axis=KDim, forward=True, init=0.0)
def cumsum(carry: float, val: float) -> float:
    return carry + val
```

Proposed (simple form — body returns only the carry, call returns only `ys`):

```python
@gtx.scan(axis=KDim, forward=True)
def cumsum(carry: float, val: float) -> float:
    return carry + val


integral = cumsum(x, init=0.0)
```

General form, when the column total is itself a result (e.g. accumulated
surface flux) — impossible today without reading the last level of a full 3D
field:

```python
@gtx.scan(axis=KDim, forward=True)
def cumsum_step(carry: float, val: float) -> tuple[float, float]:
    s = carry + val
    return s, s


total, integral = cumsum_step(x, init=0.0)
# total:    Field[[Cell], float]      — the final carry
# integral: Field[[Cell, KDim], float]
```

## 2. ICON-like tridiagonal forward sweep (boundary condition at level 0)

Today (`tests/next_tests/integration_tests/multi_feature_tests/ffront_tests/test_icon_like_scan.py`):

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


# program level: dummy bool output and output slicing
_solve_nonhydro_stencil_52_like(..., out=(z_q[:, 1:], w[:, 1:], dummy[:, 1:]))
```

Proposed (mechanism §3.3.1 — peelable index conditional; no flag, no dummy
output, no slicing; the toolchain peels `k == 0` out of the loop):

```python
class State(NamedTuple):
    z_q: float
    w: float


@gtx.scan(axis=KDim, forward=True)
def w_fwd_sweep(carry: State, z_q: float, w: float, z_a: float, z_b: float, z_c: float) -> State:
    z_g = z_b + z_a * carry.z_q
    step = State(z_q=(0.0 - z_c) * z_g, w=z_a * carry.w * z_g)
    return concat_where(KDim == 0, State(z_q=z_q, w=w), step)


z_q_res, w_res = w_fwd_sweep(z_q, w, z_a, z_b, z_c, init=State(z_q=0.0, w=0.0))
```

(`init` is still required to fix the carry structure; it is never *used*
because the peeled first iteration overwrites it — backends drop it.)

## 3. Backward scan with a first-level flag → `include_initial`

Today (icon4py `diagnose_pressure.py`, paraphrased): a backward scan with
carry `(pressure, pressure_interface, is_first_level: bool)` and
`init=(0.0, 0.0, True)`; the body selects `surface_pressure` instead of the
carried interface pressure when the flag is set, then carries `False`
forever; the `pressure` slot is never read back (output channel only).

Proposed (mechanism §3.3.2 — the boundary value comes from *outside* the
scanned fields, so it is the `init`; the flag and the dead slot disappear):

```python
@gtx.scan(axis=KDim, forward=False)
def pressure_step(p_interface: float, dz_rho_g: float) -> tuple[float, tuple[float, float]]:
    p_interface_above = p_interface * exp(...)  # hydrostatic increment
    p_full = sqrt(p_interface * p_interface_above)  # e.g. interface -> full level
    return p_interface_above, (p_full, p_interface_above)


_, (pressure, pressure_ifc) = pressure_step(dz_rho_g, init=surface_pressure)
```

With `include_initial=True` the surface value itself appears in the output —
n cell inputs, n+1 interface outputs, "the value at the boundary *is* the
init" (the exclusive/inclusive duality, research §B.2):

```python
_, p_ifc = pressure_step(dz_rho_g, init=surface_pressure, include_initial=True)
```

## 4. Thomas algorithm: don't write it at all

```python
w = gtx.lib.tridiagonal_solve(a, b, c, d)  # proposal §3.5
```

Today this is two hand-written scans (icon4py
`solve_tridiagonal_matrix_for_w_forward_sweep.py` /
`..._back_substitution.py`).

## 5. Sedimentation (scan + sub-reduce, regime 1): vertical window

The COSMO "boxtracking" flux (research §C.1, eq. 12), today inexpressible
without hand-rolled ring buffers in a carry tuple. Proposed (§3.4), with the
geometric height composed in front as a `cumsum` when only `dz` is given:

```python
KWin = gtx.WindowOffset("KWin", source=KDim, offsets=range(0, -4, -1))  # k .. k-3
KWinUp = gtx.WindowOffset(
    "KWinUp", source=KDim, offsets=range(-1, -5, -1), local_dim=KWin.local_dim
)  # k-1 .. k-4


@gtx.field_operator
def box_sedim_flux(v: CellKF, dz: CellKF, phi: CellKF, dt: float) -> CellKF:
    z = gtx.lib.cumsum(dz, axis=KDim, forward=False)  # height of lower cell face
    z_up = minimum(z(KWinUp), z + abs(v(KWin)) * dt)
    z_low = minimum(z(KWin), z_up)
    return -neighbor_sum(phi(KWin) * (z_up - z_low), axis=KWin.local_dim)
```

Window entries reaching above the model top are skip-masked (contribute the
neutral element 0); contributions from levels out of physical reach are
already zero by self-masking of the formula. The window bound (here 4) is
the static maximum vertical Courant number. The reference implementations
this transcribes are in the appendix.

## 6. muphys graupel: decomposing the fused scan

Today (`C2SM/icon4py`,
`model/atmosphere/subgrid_scale_physics/muphys/implementations/graupel.py`
at commit `515763c`; analysis in research §A.7) — one scan, five
recurrences, 21 carry components:

```python
class PrecipStateQx(NamedTuple):
    x: ta.wpfloat
    p: ta.wpfloat
    vc: ta.wpfloat
    activated: bool


class TempState(NamedTuple):
    t: ta.wpfloat
    eflx: ta.wpfloat
    activated: bool


class IntegrationState(NamedTuple):
    r: PrecipStateQx
    s: PrecipStateQx
    i: PrecipStateQx
    g: PrecipStateQx
    t_state: TempState
    rho: ta.wpfloat  # delay line: rho at k-1
    pflx_tot: ta.wpfloat  # output channel, never read back


@gtx.scan_operator(axis=dims.KDim, forward=True, init=IntegrationState(...))
def _precip_and_t(
    previous_level: IntegrationState,
    t: ta.wpfloat,
    t_kp1: ta.wpfloat,  # lookahead, precomputed outside:
    #   concat_where(KDim < last_lev, t(Koff[1]), t)
    rho: ta.wpfloat,
    q: Q_scalar,  # 6 species components
    mask_r: bool,
    mask_s: bool,
    mask_i: bool,
    mask_g: bool,
    dt: ta.wpfloat,
    dz: ta.wpfloat,
) -> IntegrationState: ...
```

with the per-species recurrence factored into a helper called four times
(excerpt; note `previous_level_rho` — the delay line in use — and the
work-skipping branch):

```python
@gtx.field_operator
def precip_qx_level_update(previous_level_q: PrecipStateQx, previous_level_rho: ta.wpfloat,
                           prefactor: ta.wpfloat, ...) -> PrecipStateQx:
    current_level_activated = previous_level_q.activated | mask
    ...
    rhox_prev = (previous_level_q.x + q) * wpfloat(0.5) * previous_level_rho
    if current_level_activated:
        vt = (previous_level_q.vc * prefactor * power((rhox_prev + offset), exponent)
              if previous_level_q.activated else wpfloat(0.0))
        x = (zeta * (flx_eff - flx_partial)) / ((wpfloat(1.0) + zeta * vt) * rho)
        p = (x * rho * vt + flx_partial) * wpfloat(0.5)
    else:
        x = q
        p = wpfloat(0.0)
    return PrecipStateQx(x=x, p=p, vc=vc, activated=current_level_activated)
```

Surface outputs `pr, ps, pi, pg, pre` are extracted in the `@gtx.program` by
slicing the last level (`dims.KDim: (vertical_end - 1, vertical_end)`).

Proposed decomposition — one scan *per species* plus one for temperature;
under §3.8 the toolchain must fuse these into a single k-loop:

```python
class PrecipState(NamedTuple):
    x: float
    p: float
    vc: float
    activated: bool


@gtx.scan(axis=KDim, forward=True)
def species_step(
    carry: PrecipState,
    q: float,
    rho: gtx.KView[float],         # rho[-1] replaces the delay-line carry slot
    vc: float,                     # per-level scale factor, computed outside
    zeta: float,
    mask: bool,
    prefactor: float, exponent: float, offset: float,
) -> tuple[PrecipState, tuple[float, float]]:
    activated = carry.activated | mask
    rho_x = q * rho[0]
    flx_eff = rho_x / zeta + 2.0 * carry.p
    flx_partial = minimum(rho_x * vc * prefactor * (rho_x + offset) ** exponent, flx_eff)
    rhox_prev = (carry.x + q) * 0.5 * rho[-1]
    vt = where(carry.activated, carry.vc * prefactor * (rhox_prev + offset) ** exponent, 0.0)
    x = where(activated, zeta * (flx_eff - flx_partial) / ((1.0 + zeta * vt) * rho[0]), q)
    p = where(activated, (x * rho[0] * vt + flx_partial) * 0.5, 0.0)
    return PrecipState(x=x, p=p, vc=vc, activated=activated), (x, p)


@gtx.scan(axis=KDim, forward=True)
def temperature_step(
    eflx: float,
    t: gtx.KView[float],           # t[+1] replaces the precomputed t_kp1 input
    pr: float, pflx_tot: float,    # same-level outputs of the species scans
    q: Q, qliq: float, qice: float,
    rho: float, dz: float, dt: float,
) -> tuple[float, float]: ...


@gtx.field_operator
def graupel(...):
    ...
    r_final, (qr, pr) = species_step(q.r, rho, vc_r, zeta, mask_r, *RAIN_PARAMS, init=...)
    s_final, (qs, ps) = species_step(q.s, rho, vc_s, zeta, mask_s, *SNOW_PARAMS, init=...)
    i_final, (qi, pi) = species_step(q.i, rho, vc_i, zeta, mask_i, *ICE_PARAMS, init=...)
    g_final, (qg, pg) = species_step(q.g, rho, vc_g, zeta, mask_g, *GRAUPEL_PARAMS, init=...)
    eflx_final, t_out = temperature_step(
        t, pr, ps + pi + pg, q, qr + q.c, qs + qi + qg, rho, dz, dt, init=0.0
    )
    # surface diagnostics are final carries — no last-level program slicing:
    return t_out, (qr, qs, qi, qg), (r_final.p, s_final.p, i_final.p, g_final.p, eflx_final)
```

What disappeared: the 21-component shared carry (each scan carries only its
own 4 floats / 1 float), the `rho` delay line (`rho[-1]`), the external
`t_kp1` precomputation (`t[+1]`), the pass-through `t`/`pflx_tot` slots
(carry/output separation), the `*_scalar` helper twins (slice bodies call
field operators), the program-level surface slicing (final carries), and
the `if current_level_activated:` identity else-branch with its ~650 lines
of DaCe repair hooks (`where`-masking; "else is identity" becomes a
transformation concern). What it costs: correctness now depends on the §3.8
fusion obligation — five scans sharing `rho`, `zeta`, masks must compile to
one vertical loop, which is the acceptance test of the rewrite.

## Appendix: sedimentation reference implementations

Transcribed from https://gist.github.com/havogt/c52d19f2f7557c2c048d12be7330bda8
(`test_sedim_box_standalone.py`), preserved here because it is the canonical
executable specification of the scan + sub-reduce pattern (research §C.2)
and a future acceptance test for §3.4. License of the original: WTFPL
(Copyright (c) 2021, ETH Zurich); reference:
http://www.cosmo-model.org/content/model/documentation/core/docu_sedi_twomom.pdf

```python
from typing import Iterable

import numpy as np
import numpy.typing as npt


def box_sedim_naive(v: npt.NDArray, z: npt.NDArray, phi: npt.NDArray, dt: npt.DTypeLike):
    Pbar = np.zeros_like(v)
    iter = 0
    for k in range(v.shape[0]):
        for m in range(k + 1):
            z_up = min(z[k - m - 1], z[k] + np.abs(v[k - m]) * dt)
            z_low = min(z[k - m], z_up)
            Pbar[k] -= phi[k - m] * (z_up - z_low)
            iter += 1
    print(f"iter = {iter}")
    return Pbar


class rolling_buffer:
    def __init__(self, size):
        self.data = np.zeros(size)

    def __getitem__(self, index):
        return self.data[index]

    def next(self, value):
        self.data = np.roll(self.data, 1)
        self.data[0] = value
        return self.data


def box_sedim_scanlike(v: npt.NDArray, z: npt.NDArray, phi: npt.NDArray, dt: npt.DTypeLike):
    def delta(z_state, v_state, m):
        z_up = min(z_state[1 + m], z_state[0] + np.abs(v_state[m]) * dt)
        z_low = min(z_state[m], z_up)
        return z_up - z_low

    Pbar = np.zeros_like(v)
    z_state = rolling_buffer(4)
    v_state = rolling_buffer(3)
    phi_state = rolling_buffer(3)
    for k in range(v.shape[0]):
        z_state.next(z[k])
        v_state.next(v[k])
        phi_state.next(phi[k])
        if k <= 1:
            Pbar[k] = -phi_state[0] * delta(z_state, v_state, 0)
        elif k == 2:
            Pbar[k] = -phi_state[0] * delta(z_state, v_state, 0) - phi_state[1] * delta(
                z_state, v_state, 1
            )
        else:
            Pbar[k] = (
                -phi_state[0] * delta(z_state, v_state, 0)
                - phi_state[1] * delta(z_state, v_state, 1)
                - phi_state[2] * delta(z_state, v_state, 2)
            )
    return Pbar


def box_sedim_scanpass(k, state, v, z, phi, dt):  # k for simplicity
    def delta(z_state, v_state, m):
        z_up = min(z_state[1 + m], z_state[0] + np.abs(v_state[m]) * dt)
        z_low = min(z_state[m], z_up)
        return z_up - z_low

    _, z_state, v_state, phi_state = state
    z_state.next(z)
    v_state.next(v)
    phi_state.next(phi)
    if k <= 1:
        return -phi_state[0] * delta(z_state, v_state, 0), z_state, v_state, phi_state
    elif k == 2:
        return (
            -phi_state[0] * delta(z_state, v_state, 0) - phi_state[1] * delta(z_state, v_state, 1),
            z_state,
            v_state,
            phi_state,
        )
    else:
        return (
            -phi_state[0] * delta(z_state, v_state, 0)
            - phi_state[1] * delta(z_state, v_state, 1)
            - phi_state[2] * delta(z_state, v_state, 2),
            z_state,
            v_state,
            phi_state,
        )


def scan(scanpass, range_, init):
    def get_n(n):
        def foo(arg):
            if isinstance(arg, Iterable):
                return arg[n]
            else:
                return arg

        return foo

    def foo(*args):
        Pbar = np.zeros(len(range_))
        state = init
        for k in range_:
            state = scanpass(k, state, *list(map(get_n(k), args)))
            Pbar[k] = state[0]  # for simplicity
        return Pbar

    return foo


def box_sedim_scan(v: npt.NDArray, z: npt.NDArray, phi: npt.NDArray, dt: npt.DTypeLike):
    return scan(
        box_sedim_scanpass,
        range(v.shape[0]),
        (None, rolling_buffer(4), rolling_buffer(3), rolling_buffer(3)),
    )(v, z, phi, dt)


class shift_origin:
    def __init__(self, arr: npt.NDArray, offset: int):
        self.arr = arr
        self.offset = offset

    def __getitem__(self, index):
        return self.arr[index + self.offset]


def box_sedim_naive_func(v: npt.NDArray, z: npt.NDArray, phi: npt.NDArray, dt: npt.DTypeLike):
    Pbar = np.zeros_like(v)
    iter = 0
    for k in range(v.shape[0]):

        def foo(m, v, z, phi, dt):
            nonlocal iter
            z_up = min(z[-m - 1], z[0] + np.abs(v[-m]) * dt)
            z_low = min(z[-m], z_up)
            iter += 1
            return -phi[-m] * (z_up - z_low)

        Pbar[k] = sum(
            foo(m, shift_origin(v, k), shift_origin(z, k), shift_origin(phi, k), dt)
            for m in range(k + 1)
        )
    print(f"iter = {iter}")
    return Pbar


def box_sedim_naive_break(v: npt.NDArray, z: npt.NDArray, phi: npt.NDArray, dt: npt.DTypeLike):
    Pbar = np.zeros_like(v)
    iter = 0
    for k in range(v.shape[0]):
        for m in range(k):
            if z[k - m] > z[k] + np.abs(v[k - m] * dt):
                break
            z_up = min(z[k - m - 1], z[k] + np.abs(v[k - m]) * dt)
            z_low = min(z[k - m], z_up)
            Pbar[k] -= phi[k - m] * (z_up - z_low)
            iter += 1
    print(f"iter = {iter}")
    return Pbar


def box_sedim_paper(v: npt.NDArray, delta_z: npt.NDArray, phi: npt.NDArray, dt: npt.DTypeLike):
    Pbar = np.zeros_like(v)
    for k in range(v.shape[0]):
        m = 0
        delta_sum = 0.0
        while delta_sum < np.abs(v[k]) * dt and k + m < v.shape[0]:
            # z_low = 0.0
            z_up = min(delta_z[k], -delta_sum + np.abs(v[k]) * dt)
            Pbar[k + m] -= phi[k] * z_up
            m += 1
            if k + m < v.shape[0]:
                delta_sum += delta_z[k + m]
    return Pbar


def test_box_sedim():
    z = np.asarray(range(11, 0, -1))
    delta_z = -(z[1:] - z[:-1])
    v = np.array([0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 0.5, 0.5])
    dt = 1.0
    phi = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    ref = np.asarray([-0.05, -0.05, -0.1, -0.1, -0.1, -0.2, -0.2, -0.2, -0.25, -0.15])

    paper = box_sedim_paper(v, delta_z, phi, dt)
    naive = box_sedim_naive(v, z, phi, dt)
    naive_break = box_sedim_naive_break(v, z, phi, dt)
    naive_func = box_sedim_naive_func(v, z, phi, dt)
    scan_like = box_sedim_scanlike(v, z, phi, dt)
    scanned = box_sedim_scan(v, z, phi, dt)

    print(paper)
    print(naive)
    print(naive_break)
    print(naive_func)
    print(scan_like)
    print(scanned)

    assert np.allclose(paper, ref)
    assert np.allclose(paper[1:], naive[1:])  # TODO check bounds
    assert np.allclose(paper[1:-1], naive_break[1:-1])  # TODO check bounds
    assert np.allclose(paper[1:], naive_func[1:])  # TODO check bounds
    assert np.allclose(paper[1:], scan_like[1:])  # TODO check bounds


test_box_sedim()
```
