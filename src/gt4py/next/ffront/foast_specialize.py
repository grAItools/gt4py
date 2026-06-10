# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

from gt4py.next import errors
from gt4py.next.ffront import stages as ffront_stages, type_specifications as ts_ffront
from gt4py.next.ffront.foast_passes.specialize_type_vars import SpecializeTypeVars
from gt4py.next.ffront.foast_passes.type_deduction import FieldOperatorTypeDeduction
from gt4py.next.ffront.stages import ConcreteFOASTOperatorDef
from gt4py.next.otf import toolchain, workflow
from gt4py.next.type_system import type_info


@dataclasses.dataclass(frozen=True)
class FoastSpecializeTypeVars(
    workflow.Workflow[ConcreteFOASTOperatorDef, ConcreteFOASTOperatorDef]
):
    """
    Specialize a generic FOAST operator definition for the given compile-time arguments.

    The type variables in the signature of a generic operator are bound by structurally
    matching the concrete argument types and then substituted throughout the FOAST tree,
    such that all later steps work on a fully concrete (monomorphic) operator definition.
    For non-generic operators this step is a no-op.
    """

    def __call__(self, inp: ConcreteFOASTOperatorDef) -> ConcreteFOASTOperatorDef:
        foast_node = inp.data.foast_node
        operator_type = foast_node.type
        if not isinstance(operator_type, ts_ffront.FieldOperatorType) or not type_info.is_generic(
            operator_type
        ):
            # generic scan operators are rejected at decoration time
            return inp

        definition = operator_type.definition
        param_types = [
            *definition.pos_only_args,
            *definition.pos_or_kw_args.values(),
            *definition.kw_only_args.values(),
        ]
        arg_types = list(inp.args.args)
        if len(arg_types) == len(param_types) + 1:
            # in the program context of an operator the last argument is the 'out'
            # argument, carrying the type of the operator's return value
            param_types.append(definition.returns)

        try:
            binding = type_info.bind_type_vars(param_types, arg_types)
        except ValueError as err:
            raise errors.DSLTypeError(
                foast_node.location,
                f"Can not specialize generic operator '{foast_node.id}' for arguments of"
                f" type '{', '.join(str(t) for t in arg_types)}': {err}",
            ) from err

        new_foast_node = SpecializeTypeVars.apply(foast_node, binding)
        if type_info.is_generic(new_foast_node.type):
            raise errors.DSLTypeError(
                foast_node.location,
                f"Can not specialize generic operator '{foast_node.id}': not all type"
                f" variables could be bound from arguments of type"
                f" '{', '.join(str(t) for t in arg_types)}' (deduced binding:"
                f" '{binding}').",
            )
        if __debug__:
            # substituting the binding must commute with type deduction
            assert FieldOperatorTypeDeduction.apply(new_foast_node).type == new_foast_node.type

        return toolchain.ConcreteArtifact(
            data=dataclasses.replace(inp.data, foast_node=new_foast_node), args=inp.args
        )


def foast_specialize_factory(
    cached: bool = True,
) -> workflow.Workflow[ConcreteFOASTOperatorDef, ConcreteFOASTOperatorDef]:
    """Optionally wrap `FoastSpecializeTypeVars` in a `CachedStep`."""
    wf: workflow.Workflow[ConcreteFOASTOperatorDef, ConcreteFOASTOperatorDef] = (
        FoastSpecializeTypeVars()
    )
    if cached:
        wf = workflow.CachedStep(wf, hash_function=ffront_stages.fingerprint_stage)
    return wf
