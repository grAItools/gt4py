# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Mapping, TypeVar

from gt4py.eve import NodeTranslator
from gt4py.next.ffront import field_operator_ast as foast, type_specifications as ts_ffront
from gt4py.next.type_system import type_info, type_specifications as ts


_NodeT = TypeVar("_NodeT", bound=foast.LocatedNode)


class SpecializeTypeVars(NodeTranslator):
    """
    Substitute bound type variables in the types of all nodes of a FOAST tree.

    The binding maps type variable names to concrete scalar types, see
    :func:`gt4py.next.type_system.type_info.bind_type_vars`. Unbound type variables
    are kept, such that the caller can decide whether a partial specialization is an
    error (e.g. using :func:`gt4py.next.type_system.type_info.is_generic`).
    """

    binding: Mapping[str, ts.ScalarType]

    def __init__(self, binding: Mapping[str, ts.ScalarType]):
        super().__init__()
        self.binding = binding

    @classmethod
    def apply(cls, node: _NodeT, binding: Mapping[str, ts.ScalarType]) -> _NodeT:
        if not binding:
            return node
        return cls(binding).visit(node)

    def _substitute(self, type_: ts.TypeSpec) -> ts.TypeSpec:
        match type_:
            case ts_ffront.FieldOperatorType(definition=definition):
                new_definition = type_info.substitute_type_vars(definition, self.binding)
                assert isinstance(new_definition, ts.FunctionType)
                return ts_ffront.FieldOperatorType(definition=new_definition)
            case ts_ffront.ScanOperatorType(axis=axis, definition=definition):
                new_definition = type_info.substitute_type_vars(definition, self.binding)
                assert isinstance(new_definition, ts.FunctionType)
                return ts_ffront.ScanOperatorType(axis=axis, definition=new_definition)
        return type_info.substitute_type_vars(type_, self.binding)

    def visit_LocatedNode(self, node: foast.LocatedNode, **kwargs: Any) -> foast.LocatedNode:
        new_node = self.generic_visit(node, **kwargs)
        if isinstance(node_type := getattr(new_node, "type", None), ts.TypeSpec):
            new_node.type = self._substitute(node_type)
        return new_node
