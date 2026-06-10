# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import typing

from gt4py.next import Dimension, Field, float32, float64
from gt4py.next.ffront.foast_passes.specialize_type_vars import SpecializeTypeVars
from gt4py.next.ffront.foast_passes.type_deduction import FieldOperatorTypeDeduction
from gt4py.next.ffront.func_to_foast import FieldOperatorParser
from gt4py.next.type_system import type_info, type_specifications as ts


TDim = Dimension("TDim")
FloatT = typing.TypeVar("FloatT", float32, float64)

float32_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT32)
float64_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)


def _parse_generic_operator():
    def generic_op(
        a: Field[[TDim], FloatT], b: Field[[TDim], FloatT], s: FloatT
    ) -> Field[[TDim], FloatT]:
        return a - b * s

    return FieldOperatorParser.apply_to_function(generic_op)


def test_specialize_full_binding():
    parsed = _parse_generic_operator()
    assert type_info.is_generic(parsed.type)

    specialized = SpecializeTypeVars.apply(parsed, {"FloatT": float64_type})

    assert not type_info.is_generic(specialized.type)
    assert specialized.type.returns == ts.FieldType(dims=[TDim], dtype=float64_type)
    assert [param.type for param in specialized.params] == [
        ts.FieldType(dims=[TDim], dtype=float64_type),
        ts.FieldType(dims=[TDim], dtype=float64_type),
        float64_type,
    ]
    # all nodes are concrete, e.g. the body expression
    assert specialized.body.stmts[-1].value.type == ts.FieldType(dims=[TDim], dtype=float64_type)


def test_specialize_commutes_with_type_deduction():
    parsed = _parse_generic_operator()
    specialized = SpecializeTypeVars.apply(parsed, {"FloatT": float32_type})
    rededuced = FieldOperatorTypeDeduction.apply(specialized)
    assert rededuced.type == specialized.type


def test_unbound_vars_stay_generic():
    parsed = _parse_generic_operator()
    specialized = SpecializeTypeVars.apply(parsed, {"OtherT": float64_type})
    assert type_info.is_generic(specialized.type)
    assert specialized.type == parsed.type


def test_original_tree_unchanged():
    parsed = _parse_generic_operator()
    _ = SpecializeTypeVars.apply(parsed, {"FloatT": float64_type})
    assert type_info.is_generic(parsed.type)
