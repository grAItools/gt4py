# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import enum
from typing import Any

import gt4py.next.ffront.field_operator_ast as foast
from gt4py._core import definitions as core_defs
from gt4py.eve import NodeTranslator, concepts, traits
from gt4py.next import common, errors
from gt4py.next.ffront import (
    dialect_ast_enums,
    experimental,
    fbuiltins,
    gtcallable,
    transform_utils,
)
from gt4py.next.type_system import type_translation


#: GT4Py builtins indexed by object identity, mapping back to their canonical name.
_BUILTIN_NAME_BY_ID: dict[int, str] = {
    id(value): name
    for name, value in {
        **fbuiltins.BUILTINS,
        **{
            name: getattr(experimental, name)
            for name in experimental.EXPERIMENTAL_FUN_BUILTIN_NAMES
        },
    }.items()
}


@dataclasses.dataclass
class ClosureVarResolution(NodeTranslator, traits.VisitorWithSymbolTableTrait):
    """
    Resolve references to closure variables to a canonical, value-based form.

    Closure variables are captured by name, but their meaning is defined by the
    *value* the name is bound to at definition time. This pass makes the rest of
    the toolchain independent of how a value was referenced in the source
    (directly, under an alias, or as an attribute of a module or namespace):

    - References to compile-time constants are replaced by `Constant` nodes.
      This covers attributes of immutable namespaces (`eve.utils.FrozenNamespace`,
      `enum` classes — previously handled by ``ClosureVarFolding``), scalar
      attributes of modules (e.g. ``np.pi``), and module-level global scalars
      annotated `typing.Final`.
    - References to GT4Py builtins are rewritten to the builtin's canonical name
      (e.g. ``gtx.where`` or an alias ``my_where`` become ``where``), so the
      name-based builtin handling in later passes applies.
    - Attribute references to `GTCallable`s (field operators, scan operators,
      ...), `FieldOffset`s and `Dimension`s (e.g. ``some_module.an_operator``)
      are rewritten to a synthesized name derived from the value (see
      :func:`transform_utils.mangled_function_name`), so module-prefixed
      accesses resolve like direct references. Direct references — including
      aliases — keep their source name; the program lowering registers IR
      function definitions under the reference name, so aliasing needs no
      rewriting.

    Synthesized names are registered as additional closure variables: the pass
    **mutates** the `closure_vars` mapping passed to :meth:`apply` and appends
    matching symbols to `FunctionDefinition.closure_vars`. Entries that are no
    longer referenced (e.g. the module object itself) are removed from the AST
    by the subsequent `DeadClosureVarElimination` pass, but remain in the
    mapping.

    References this pass cannot resolve to a constant or canonical name (e.g.
    scalars not annotated `typing.Final`, captured fields) are left untouched:
    they remain valid in embedded execution and are rejected with a dedicated
    error message when lowering to IR.
    """

    closure_vars: dict[str, Any]
    final_names: frozenset[str] = frozenset()

    @classmethod
    def apply(
        cls,
        node: foast.FunctionDefinition | foast.FieldOperator,
        closure_vars: dict[str, Any],
        final_names: frozenset[str] = frozenset(),
    ) -> foast.FunctionDefinition:
        return cls(closure_vars=closure_vars, final_names=final_names).visit(node)

    def visit_FunctionDefinition(
        self, node: foast.FunctionDefinition, **kwargs: Any
    ) -> foast.FunctionDefinition:
        self._added_closure_vars: dict[str, Any] = {}
        new_node: foast.FunctionDefinition = self.generic_visit(
            node, current_closure_vars=node.closure_vars, **kwargs
        )
        existing = {symbol.id for symbol in new_node.closure_vars}
        new_symbols: list[foast.Symbol] = [
            foast.Symbol(
                id=name,
                type=type_translation.from_value(value),
                namespace=dialect_ast_enums.Namespace.CLOSURE,
                location=new_node.location,
            )
            for name, value in self._added_closure_vars.items()
            if name not in existing
        ]
        if not new_symbols:
            return new_node
        return foast.FunctionDefinition(
            id=new_node.id,
            params=new_node.params,
            body=new_node.body,
            closure_vars=[*new_node.closure_vars, *new_symbols],
            type=new_node.type,
            location=new_node.location,
        )

    def visit_Name(
        self,
        node: foast.Name,
        current_closure_vars: list[foast.Symbol],
        symtable: dict[str, foast.Symbol],
        **kwargs: Any,
    ) -> foast.Name | foast.Constant | foast.Attribute:
        if node.id in symtable and symtable[node.id] in current_closure_vars:
            value = self.closure_vars[node.id]
            return self._resolve_value_reference(
                node, node.id, value, via_namespace=False, symtable=symtable
            )
        return node

    def visit_Attribute(
        self,
        node: foast.Attribute,
        symtable: dict[str, foast.Symbol],
        **kwargs: Any,
    ) -> foast.Constant | foast.Attribute | foast.Name:
        value = self.visit(node.value, symtable=symtable, **kwargs)
        if isinstance(value, foast.Constant):
            if isinstance(value.value, type_translation.PythonNamespaceObject):
                if not hasattr(value.value, node.attr):
                    raise errors.MissingAttributeError(node.location, node.attr)
                leaf = getattr(value.value, node.attr)
                if isinstance(leaf, enum.Enum):
                    leaf = leaf.value
                return self._resolve_value_reference(
                    node, node.attr, leaf, via_namespace=True, symtable=symtable
                )
            if hasattr(value.value, node.attr):
                return foast.Constant(value=getattr(value.value, node.attr), location=node.location)
            raise errors.MissingAttributeError(node.location, node.attr)
        return foast.Attribute(value=value, attr=node.attr, location=node.location)

    def _resolve_value_reference(
        self,
        node: foast.Expr,
        source_name: str,
        value: Any,
        *,
        via_namespace: bool,
        symtable: dict[str, foast.Symbol],
    ) -> foast.Name | foast.Constant | foast.Attribute:
        location = node.location

        if (builtin_name := _BUILTIN_NAME_BY_ID.get(id(value))) is not None:
            return self._make_reference(builtin_name, value, location, symtable)

        if isinstance(value, type_translation.PythonNamespaceObject):
            # placeholder, to be consumed by an enclosing `visit_Attribute`
            return foast.Constant(value=value, location=location)

        if isinstance(value, gtcallable.GTCallable):
            if via_namespace:
                return self._make_reference(
                    transform_utils.mangled_function_name(value), value, location, symtable
                )
            return node  # type: ignore[return-value]  # always a foast.Name here

        if isinstance(value, common.Dimension):
            # dimensions resolve by value through the type system ('NamespaceProxy'),
            # so attribute references can stay as they are. Renaming them to the
            # dimension's name could collide with the offset of the same name
            # (e.g. local dimension 'V2E' vs. offset 'V2E').
            return node  # type: ignore[return-value]  # foast.Name or foast.Attribute

        if isinstance(value, fbuiltins.FieldOffset):
            if via_namespace:
                # the offset name is also the key used for offset-provider lookups
                return self._make_reference(str(value.value), value, location, symtable)
            return node  # type: ignore[return-value]  # always a foast.Name here

        if core_defs.is_scalar_type(value) and (via_namespace or source_name in self.final_names):
            return foast.Constant(value=value, location=location)

        if via_namespace:
            # preserve the behavior of the former `ClosureVarFolding` pass:
            # remaining namespace attributes are folded as constants and
            # validated by the type deduction.
            return foast.Constant(value=value, location=location)

        return node  # type: ignore[return-value]  # always a foast.Name here

    def _make_reference(
        self,
        name: str,
        value: Any,
        location: concepts.SourceLocation,
        symtable: dict[str, foast.Symbol],
    ) -> foast.Name:
        if not name.isidentifier():
            raise errors.DSLError(
                location, f"Cannot use '{name}' as a canonical symbol name in GT4Py."
            )
        if (existing := self.closure_vars.get(name, _MISSING)) is not _MISSING:
            if existing is not value and existing != value:
                raise errors.DSLError(
                    location,
                    f"The reference resolves to the canonical name '{name}', but a different "
                    f"closure variable of that name already exists.",
                )
        elif name in symtable:
            raise errors.DSLError(
                location,
                f"The reference resolves to the canonical name '{name}', which is shadowed by "
                f"a local variable or parameter of the same name. Please rename the local.",
            )
        self.closure_vars[name] = value
        self._added_closure_vars.setdefault(name, value)
        return foast.Name(id=name, location=location)


class _Missing:
    pass


_MISSING = _Missing()
