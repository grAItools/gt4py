# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for dimension variables (generic dimensions) in the type system.

Prototype for the design described in
`docs/development/investigations/dimension-generic-fields.md`: a `Dims[...]`
annotation may contain a `TypeVar` bound to `Dimension` (a single unknown
dimension, `ts.DimensionVar`) or an unpacked `TypeVarTuple` (an unknown list
of dimensions, `ts.DimsVar`), and `type_info.bind_type_vars` /
`substitute_type_vars` handle them alongside dtype type variables.
"""

from typing import TypeVar

import pytest
from typing_extensions import TypeVarTuple, Unpack

import gt4py.next as gtx
from gt4py.next import common
from gt4py.next.type_system import type_info, type_specifications as ts, type_translation


IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")
KDim = gtx.Dimension("KDim", kind=common.DimensionKind.VERTICAL)

DT = TypeVar("DT", gtx.float32, gtx.float64)
D = TypeVar("D", bound=common.Dimension)
D2 = TypeVar("D2", bound=common.Dimension)
Ds = TypeVarTuple("Ds")

FLOAT64 = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
FLOAT32 = ts.ScalarType(kind=ts.ScalarKind.FLOAT32)


def field_t(*dims: common.Dimension, dtype: ts.ScalarType = FLOAT64) -> ts.FieldType:
    return ts.FieldType(dims=list(dims), dtype=dtype)


# --- type translation ---------------------------------------------------------


def test_dimension_var_translation():
    field_type = type_translation.from_type_hint(gtx.Field[gtx.Dims[D], gtx.float64])
    assert field_type == field_t(ts.DimensionVar(value="D"))


def test_dims_var_translation():
    field_type = type_translation.from_type_hint(gtx.Field[gtx.Dims[Unpack[Ds]], gtx.float64])
    assert field_type == field_t(ts.DimsVar(value="Ds"))


def test_mixed_concrete_and_variadic_translation():
    field_type = type_translation.from_type_hint(gtx.Field[gtx.Dims[IDim, Unpack[Ds]], gtx.float64])
    assert field_type == field_t(IDim, ts.DimsVar(value="Ds"))


def test_dim_var_combines_with_dtype_var():
    field_type = type_translation.from_type_hint(gtx.Field[gtx.Dims[Unpack[Ds]], DT])
    assert field_type.dims == [ts.DimsVar(value="Ds")]
    assert isinstance(field_type.dtype, ts.TypeVarType)


def test_two_variadic_vars_rejected():
    Ds2 = TypeVarTuple("Ds2")
    with pytest.raises(ValueError, match="At most one variadic"):
        type_translation.from_type_hint(gtx.Field[gtx.Dims[Unpack[Ds], Unpack[Ds2]], gtx.float64])


def test_unbound_type_var_in_dims_rejected():
    U = TypeVar("U")
    with pytest.raises(ValueError, match="bound=Dimension"):
        type_translation.from_type_hint(gtx.Field[gtx.Dims[U], gtx.float64])


# --- genericity predicate ------------------------------------------------------


def test_is_generic():
    assert type_info.is_generic(field_t(ts.DimensionVar(value="D")))
    assert type_info.is_generic(field_t(ts.DimsVar(value="Ds")))
    assert not type_info.is_generic(field_t(IDim, KDim))


def test_dims_validator_skips_ordering_for_generic_dims():
    # ordering of concrete dims is still enforced ...
    with pytest.raises(ValueError, match="not ordered correctly"):
        field_t(KDim, IDim)
    # ... but undefined while dimension variables are present
    field_t(ts.DimsVar(value="Ds"), IDim)


# --- binding -------------------------------------------------------------------


def test_bind_variadic_binds_all_dims():
    binding = type_info.bind_type_vars([field_t(ts.DimsVar(value="Ds"))], [field_t(IDim, KDim)])
    assert binding == {"Ds": (IDim, KDim)}


def test_bind_variadic_with_prefix_and_suffix():
    binding = type_info.bind_type_vars(
        [field_t(IDim, ts.DimsVar(value="Ds"), KDim)], [field_t(IDim, JDim, KDim)]
    )
    assert binding == {"Ds": (JDim,)}


def test_bind_variadic_to_empty():
    binding = type_info.bind_type_vars([field_t(IDim, ts.DimsVar(value="Ds"))], [field_t(IDim)])
    assert binding == {"Ds": ()}


def test_bind_single_dim_var_positionally():
    binding = type_info.bind_type_vars(
        [field_t(ts.DimensionVar(value="D"), KDim)], [field_t(IDim, KDim)]
    )
    assert binding == {"D": IDim}


def test_bind_same_dim_var_across_params():
    binding = type_info.bind_type_vars(
        [field_t(ts.DimensionVar(value="D")), field_t(ts.DimensionVar(value="D"))],
        [field_t(IDim), field_t(IDim)],
    )
    assert binding == {"D": IDim}


def test_bind_inconsistent_dim_var_rejected():
    with pytest.raises(ValueError, match="bound inconsistently"):
        type_info.bind_type_vars(
            [field_t(ts.DimensionVar(value="D")), field_t(ts.DimensionVar(value="D"))],
            [field_t(IDim), field_t(JDim)],
        )


def test_bind_dim_and_dtype_vars_together():
    param = ts.FieldType(
        dims=[ts.DimsVar(value="Ds")],
        dtype=ts.TypeVarType(name="DT", constraints=(FLOAT32, FLOAT64)),
    )
    binding = type_info.bind_type_vars([param], [field_t(IDim, KDim, dtype=FLOAT32)])
    assert binding == {"DT": FLOAT32, "Ds": (IDim, KDim)}


def test_bind_rank_mismatch_leaves_unbound():
    # structural mismatches are reported by the signature checks, not here
    binding = type_info.bind_type_vars([field_t(ts.DimensionVar(value="D"), KDim)], [field_t(IDim)])
    assert binding == {}


def test_bind_scalar_arg_binds_variadic_to_zero_dims():
    binding = type_info.bind_type_vars([field_t(ts.DimsVar(value="Ds"))], [FLOAT64])
    assert binding == {"Ds": ()}


# --- substitution ---------------------------------------------------------------


def test_substitute_variadic():
    generic = field_t(ts.DimsVar(value="Ds"))
    assert type_info.substitute_type_vars(generic, {"Ds": (IDim, KDim)}) == field_t(IDim, KDim)


def test_substitute_single_dim_var():
    generic = field_t(ts.DimensionVar(value="D"), KDim)
    assert type_info.substitute_type_vars(generic, {"D": IDim}) == field_t(IDim, KDim)


def test_substitute_keeps_unbound_vars():
    generic = field_t(ts.DimensionVar(value="D"))
    assert type_info.substitute_type_vars(generic, {"E": IDim}) == generic


def test_substitute_function_type_end_to_end():
    """The dim-generic 'shape-preserving operator' use case."""
    generic_field = ts.FieldType(
        dims=[ts.DimsVar(value="Ds")],
        dtype=ts.TypeVarType(name="DT", constraints=(FLOAT32, FLOAT64)),
    )
    func = ts.FunctionType(
        pos_only_args=[generic_field, generic_field],
        pos_or_kw_args={},
        kw_only_args={},
        returns=generic_field,
    )
    args = [field_t(IDim, KDim, dtype=FLOAT32), field_t(IDim, KDim, dtype=FLOAT32)]
    binding = type_info.bind_type_vars(func.pos_only_args, args)
    concrete = type_info.substitute_type_vars(func, binding)
    assert concrete.returns == field_t(IDim, KDim, dtype=FLOAT32)
    assert not type_info.is_generic(concrete)
