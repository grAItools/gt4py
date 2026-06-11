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

import gt4py.next.ffront.program_ast as past
from gt4py._core import definitions as core_defs
from gt4py.eve import NodeTranslator, concepts, traits
from gt4py.next import common, errors
from gt4py.next.ffront import dialect_ast_enums, fbuiltins, gtcallable, transform_utils
from gt4py.next.type_system import type_translation


@dataclasses.dataclass
class ClosureVarCanonicalization(NodeTranslator, traits.VisitorWithSymbolTableTrait):
    """
    Resolve references to closure variables of a program to a canonical, value-based form.

    This is the program (PAST) counterpart of the field operator (FOAST) pass
    `ClosureVarResolution`, see there for the rationale. In programs the relevant
    references are calls to `GTCallable`s as an attribute of a module (direct
    references — including aliases — keep their source name) and scalar
    constants used as arguments or in domain expressions.

    The pass **mutates** the `closure_vars` mapping passed to :meth:`apply`:
    entries for newly introduced canonical names are added.
    """

    closure_vars: dict[str, Any]
    final_names: frozenset[str] = frozenset()

    @classmethod
    def apply(
        cls,
        node: past.Program,
        closure_vars: dict[str, Any],
        final_names: frozenset[str] = frozenset(),
    ) -> past.Program:
        return cls(closure_vars=closure_vars, final_names=final_names).visit(node)

    def visit_Program(self, node: past.Program, **kwargs: Any) -> past.Program:
        self._added_closure_vars: dict[str, Any] = {}
        new_node: past.Program = self.generic_visit(
            node, current_closure_vars=node.closure_vars, **kwargs
        )
        existing = {symbol.id for symbol in new_node.closure_vars}
        new_symbols: list[past.Symbol] = [
            past.Symbol(
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
        return past.Program(
            id=new_node.id,
            type=new_node.type,
            params=new_node.params,
            body=new_node.body,
            closure_vars=[*new_node.closure_vars, *new_symbols],
            location=new_node.location,
        )

    def visit_Call(self, node: past.Call, **kwargs: Any) -> past.Call:
        new_node: past.Call = self.generic_visit(node, **kwargs)
        if not isinstance(new_node.func, past.Name):
            raise errors.DSLError(
                node.location,
                "Functions must be referenced by their name or as an attribute of a module.",
            )
        return new_node

    def visit_Name(
        self,
        node: past.Name,
        current_closure_vars: list[past.Symbol],
        symtable: dict[str, past.Symbol],
        **kwargs: Any,
    ) -> past.Name | past.Constant | past.Attribute:
        if node.id in symtable and symtable[node.id] in current_closure_vars:
            value = self.closure_vars[node.id]
            return self._resolve_value_reference(
                node, node.id, value, via_namespace=False, symtable=symtable
            )
        return node

    def visit_Attribute(
        self,
        node: past.Attribute,
        symtable: dict[str, past.Symbol],
        **kwargs: Any,
    ) -> past.Constant | past.Attribute | past.Name:
        value = self.visit(node.value, symtable=symtable, **kwargs)
        if isinstance(value, past.Constant) and isinstance(
            value.value, type_translation.PythonNamespaceObject
        ):
            if not hasattr(value.value, node.attr):
                raise errors.DSLError(node.location, f"Object has no attribute '{node.attr}'.")
            leaf = getattr(value.value, node.attr)
            if isinstance(leaf, enum.Enum):
                leaf = leaf.value
            return self._resolve_value_reference(
                node, node.attr, leaf, via_namespace=True, symtable=symtable
            )
        return past.Attribute(value=value, attr=node.attr, location=node.location)

    def _resolve_value_reference(
        self,
        node: past.Expr,
        source_name: str,
        value: Any,
        *,
        via_namespace: bool,
        symtable: dict[str, past.Symbol],
    ) -> past.Name | past.Constant | past.Attribute:
        location = node.location

        if isinstance(value, type_translation.PythonNamespaceObject):
            # placeholder, to be consumed by an enclosing `visit_Attribute`
            return past.Constant(value=value, location=location)

        if isinstance(value, gtcallable.GTCallable):
            if via_namespace:
                return self._make_reference(
                    transform_utils.mangled_function_name(value), value, location, symtable
                )
            return node  # type: ignore[return-value]  # always a past.Name here

        if isinstance(value, common.Dimension):
            # dimensions resolve by value through the type system ('NamespaceProxy'),
            # so attribute references can stay as they are. Renaming them to the
            # dimension's name could collide with the offset of the same name
            # (e.g. local dimension 'V2E' vs. offset 'V2E').
            return node  # type: ignore[return-value]  # past.Name or past.Attribute

        if isinstance(value, fbuiltins.FieldOffset):
            if via_namespace:
                return self._make_reference(str(value.value), value, location, symtable)
            return node  # type: ignore[return-value]  # always a past.Name here

        if core_defs.is_scalar_type(value) and (via_namespace or source_name in self.final_names):
            return past.Constant(
                value=value, type=type_translation.from_value(value), location=location
            )

        return node  # type: ignore[return-value]  # always a past.Name here

    def _make_reference(
        self,
        name: str,
        value: Any,
        location: concepts.SourceLocation,
        symtable: dict[str, past.Symbol],
    ) -> past.Name:
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
                f"a program parameter of the same name. Please rename the parameter.",
            )
        self.closure_vars[name] = value
        self._added_closure_vars.setdefault(name, value)
        return past.Name(id=name, location=location)


class _Missing:
    pass


_MISSING = _Missing()
