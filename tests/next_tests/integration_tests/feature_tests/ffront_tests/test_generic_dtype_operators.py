# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for field operators that are generic in the field dtype (value-constrained 'TypeVar's)."""

import typing

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import errors, sin

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, cartesian_case
from next_tests.integration_tests.cases_utils import exec_alloc_descriptor


pytestmark = pytest.mark.uses_generic_dtype

FloatT = typing.TypeVar("FloatT", gtx.float32, gtx.float64)


@gtx.field_operator
def generic_diff(
    a: gtx.Field[gtx.Dims[IDim], FloatT], b: gtx.Field[gtx.Dims[IDim], FloatT]
) -> gtx.Field[gtx.Dims[IDim], FloatT]:
    return a - b


def _allocate_diff_args(case, dtype):
    size = case.default_sizes[IDim]
    a = case.as_field([IDim], np.arange(size, dtype=dtype))
    b = case.as_field([IDim], np.ones(size, dtype=dtype))
    out = case.as_field([IDim], np.zeros(size, dtype=dtype))
    return a, b, out


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_generic_field_operator(cartesian_case, dtype):
    a, b, out = _allocate_diff_args(cartesian_case, dtype)

    cases.verify(
        cartesian_case,
        generic_diff,
        a,
        b,
        out=out,
        ref=a.asnumpy() - b.asnumpy(),
    )
    assert out.asnumpy().dtype == dtype


def test_one_variant_per_dtype(cartesian_case):
    if cartesian_case.backend is None:
        pytest.skip("Embedded execution does not compile variants.")

    testee = generic_diff.with_backend(cartesian_case.backend)
    for dtype in (np.float32, np.float64):
        a, b, out = _allocate_diff_args(cartesian_case, dtype)
        testee(a, b, out=out, offset_provider=cartesian_case.offset_provider)
        assert out.asnumpy().dtype == dtype
        assert np.allclose(out.asnumpy(), a.asnumpy() - b.asnumpy())

    assert len(testee._compiled_programs.compiled_programs) == 2


def test_inconsistent_binding_error(cartesian_case):
    if cartesian_case.backend is None:
        pytest.skip("Embedded execution does not type-check arguments of direct operator calls.")

    a, _, _ = _allocate_diff_args(cartesian_case, np.float32)
    _, b, out = _allocate_diff_args(cartesian_case, np.float64)

    with pytest.raises(errors.DSLTypeError, match="Can not specialize"):
        generic_diff.with_backend(cartesian_case.backend)(
            a, b, out=out, offset_provider=cartesian_case.offset_provider
        )


def test_generic_scalar_param(cartesian_case):
    @gtx.field_operator
    def generic_scale(
        a: gtx.Field[gtx.Dims[IDim], FloatT], s: FloatT
    ) -> gtx.Field[gtx.Dims[IDim], FloatT]:
        return a * s

    for dtype in (np.float32, np.float64):
        size = cartesian_case.default_sizes[IDim]
        a = cartesian_case.as_field([IDim], np.arange(size, dtype=dtype))
        out = cartesian_case.as_field([IDim], np.zeros(size, dtype=dtype))
        s = dtype(3.0)

        cases.verify(cartesian_case, generic_scale, a, s, out=out, ref=a.asnumpy() * s)
        assert out.asnumpy().dtype == dtype


def test_generic_tuple_return(cartesian_case):
    @gtx.field_operator
    def generic_sum_diff(
        a: gtx.Field[gtx.Dims[IDim], FloatT], b: gtx.Field[gtx.Dims[IDim], FloatT]
    ) -> tuple[gtx.Field[gtx.Dims[IDim], FloatT], gtx.Field[gtx.Dims[IDim], FloatT]]:
        return a + b, a - b

    for dtype in (np.float32, np.float64):
        a, b, _ = _allocate_diff_args(cartesian_case, dtype)
        size = cartesian_case.default_sizes[IDim]
        out = (
            cartesian_case.as_field([IDim], np.zeros(size, dtype=dtype)),
            cartesian_case.as_field([IDim], np.zeros(size, dtype=dtype)),
        )

        cases.verify(
            cartesian_case,
            generic_sum_diff,
            a,
            b,
            out=out,
            ref=(a.asnumpy() + b.asnumpy(), a.asnumpy() - b.asnumpy()),
        )
        assert out[0].asnumpy().dtype == dtype


def test_generic_math_builtin(cartesian_case):
    @gtx.field_operator
    def generic_sin(a: gtx.Field[gtx.Dims[IDim], FloatT]) -> gtx.Field[gtx.Dims[IDim], FloatT]:
        return sin(a)

    for dtype in (np.float32, np.float64):
        size = cartesian_case.default_sizes[IDim]
        a = cartesian_case.as_field([IDim], np.linspace(0, 1, size, dtype=dtype))
        out = cartesian_case.as_field([IDim], np.zeros(size, dtype=dtype))

        cases.verify(cartesian_case, generic_sin, a, out=out, ref=np.sin(a.asnumpy()))
        assert out.asnumpy().dtype == dtype
