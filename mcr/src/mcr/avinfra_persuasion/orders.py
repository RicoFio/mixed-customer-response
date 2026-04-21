"""
Module for handling pre-orders and partial orders.
From https://github.com/mit-zardini-lab/PosetalGameLearningPriority/blob/main/orders.py
By Yujun Huang
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterator, TypeAlias

import networkx as nx


Relation: TypeAlias = tuple[Any, Any]
HasseNode: TypeAlias = frozenset[Any]


@dataclass
class PreOrder:
    """
    Represents a pre-order (reflexive and transitive relation).
    Stored as a dictionary of comparable pairs.
    """

    elements: set[Any]
    relations: set[Relation]
    """(a, b) means a <= b"""
    hasse_diagram: nx.DiGraph = field(init=False, default_factory=nx.DiGraph)
    """One node per equivalence class, each equivalence class is a set of elements"""

    def __post_init__(self) -> None:
        self.relations = self._reflexive_closure()
        self._build_hasse_diagram()
        self._validate()

    def _transitive_closure(self) -> set[Relation]:
        """Compute the transitive closure of the current relations."""
        closure = set(self.relations)
        added = True

        while added:
            added = False

            for a, b in list(closure):
                for c, d in list(closure):
                    if b == c and (a, d) not in closure:
                        closure.add((a, d))
                        added = True

        return closure

    def _reflexive_closure(self) -> set[Relation]:
        """Compute the reflexive closure of the current relations."""
        closure = set(self.relations)
        closure.update((element, element) for element in self.elements)
        return closure

    def build_sub_preorder(self, subset: set[Any]) -> PreOrder:
        """Build the sub-preorder induced on the given subset of elements."""
        assert subset.issubset(self.elements), (
            "Subset must be within the elements of the preorder"
        )
        sub_relations = {
            (a, b)
            for a, b in self.relations
            if a in subset and b in subset
        }
        return PreOrder(subset, sub_relations)

    def leq(self, a: Any, b: Any) -> bool:
        """Check whether a <= b."""
        return (a, b) in self.relations

    def less(self, a: Any, b: Any) -> bool:
        """Check whether a < b."""
        return self.leq(a, b) and not self.leq(b, a)

    def geq(self, a: Any, b: Any) -> bool:
        """Check whether a >= b."""
        return self.leq(b, a)

    def greater(self, a: Any, b: Any) -> bool:
        """Check whether a > b."""
        return self.less(b, a)

    def _validate(self) -> None:
        for elem in self.elements:
            if (elem, elem) not in self.relations:
                raise ValueError(f"Pre-order not reflexive: missing ({elem}, {elem})")

        for a, b, c in combinations(self.elements, 3):
            if (a, b) in self.relations and (b, c) in self.relations:
                if (a, c) not in self.relations:
                    raise ValueError(
                        "Pre-order not transitive: "
                        f"missing ({a}, {c}) from ({a}, {b}) and ({b}, {c})"
                    )

    def _build_hasse_diagram(self) -> None:
        """Build the Hasse diagram for the pre-order."""
        graph = nx.DiGraph()
        equivalence_classes = self._equivalence_classes()

        for eq_class in equivalence_classes:
            graph.add_node(frozenset(eq_class))

        for class_a, class_b in combinations(equivalence_classes, 2):
            a_rep = next(iter(class_a))
            b_rep = next(iter(class_b))

            if self.less(a_rep, b_rep):
                graph.add_edge(frozenset(class_a), frozenset(class_b))
            elif self.less(b_rep, a_rep):
                graph.add_edge(frozenset(class_b), frozenset(class_a))

        self.hasse_diagram = nx.transitive_reduction(graph)

    def _equivalence_classes(self) -> list[set[Any]]:
        equivalence_classes: list[set[Any]] = []

        for element in self.elements:
            for eq_class in equivalence_classes:
                representative = next(iter(eq_class))
                if (
                    self.leq(element, representative)
                    and self.leq(representative, element)
                ):
                    eq_class.add(element)
                    break
            else:
                equivalence_classes.append({element})

        return equivalence_classes

    def _source_nodes(self) -> list[HasseNode]:
        return [
            node for node, degree in self.hasse_diagram.in_degree() if degree == 0
        ]

    def _terminal_nodes(self) -> list[HasseNode]:
        return [
            node for node, degree in self.hasse_diagram.out_degree() if degree == 0
        ]

    def maximal_elements(self) -> set[Any]:
        """Find maximal elements in a pre-order/partial order."""
        return {
            element
            for node in self._terminal_nodes()
            for element in node
        }

    def minimal_elements(self) -> set[Any]:
        """Find minimal elements in a pre-order/partial order."""
        return {
            element
            for node in self._source_nodes()
            for element in node
        }

    def __hash__(self) -> int:
        return hash((frozenset(self.elements), frozenset(self.relations)))

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, PreOrder):
            return False
        return self.elements == value.elements and self.relations == value.relations

    def __repr__(self) -> str:
        edges_to_show = [
            (_format_hasse_node(u), _format_hasse_node(v))
            for u, v in self.hasse_diagram.edges()
        ]
        return f"PreOrder(Elements: {self.elements}, Hasse edges: {edges_to_show})"


class PartialOrder(PreOrder):
    """
    Represents a partial order (reflexive, transitive, antisymmetric).
    """

    def _validate(self) -> None:
        PreOrder._validate(self)

        for a, b in combinations(self.elements, 2):
            if (a, b) in self.relations and (b, a) in self.relations and a != b:
                raise ValueError(
                    "Pre-order not antisymmetric: "
                    f"both ({a}, {b}) and ({b}, {a}) for a != b"
                )

    def __repr__(self) -> str:
        edges_to_show = [
            (next(iter(u)), next(iter(v)))
            for u, v in self.hasse_diagram.edges()
        ]
        return f"PartialOrder(Elements: {self.elements}, Hasse edges: {edges_to_show})"


def total_order_from_list(elements: list[Any]) -> PartialOrder:
    """Create a total order from a list where the first element is smallest."""
    relations: set[Relation] = set()

    for i, a in enumerate(elements):
        for j, b in enumerate(elements):
            if i <= j:
                relations.add((a, b))
    return PartialOrder(set(elements), relations)


def completions_of_poset(poset: PartialOrder) -> Iterator[PartialOrder]:
    """
    Generate all total order completions of the given partial order.
    Note: This can be computationally expensive for large posets.
    """
    node_to_elem = {}

    for node in poset.hasse_diagram.nodes():
        if isinstance(node, frozenset):
            if len(node) != 1:
                raise ValueError("PartialOrder has non-singleton node in Hasse diagram")
            elem = next(iter(node))
        else:
            elem = node
        node_to_elem[node] = elem

    adj = {node_to_elem[n]: set() for n in poset.hasse_diagram.nodes()}
    indeg = {node_to_elem[n]: 0 for n in poset.hasse_diagram.nodes()}

    for u, v in poset.hasse_diagram.edges():
        ue = node_to_elem[u]
        ve = node_to_elem[v]
        if ve not in adj[ue]:
            adj[ue].add(ve)
            indeg[ve] += 1

    def _backtrack(
        order: list[Any],
        available: set[Any],
        current_indeg: dict[Any, int],
    ) -> Iterator[PartialOrder]:
        if len(order) == len(adj):
            yield total_order_from_list(order)
            return

        for elem in list(available):
            new_available = set(available)
            new_available.remove(elem)
            new_indeg = dict(current_indeg)
            for succ in adj[elem]:
                new_indeg[succ] -= 1
                if new_indeg[succ] == 0:
                    new_available.add(succ)
            yield from _backtrack(order + [elem], new_available, new_indeg)

    initial_available = {e for e, d in indeg.items() if d == 0}
    yield from _backtrack([], initial_available, indeg)


def _format_hasse_node(node: HasseNode) -> Any:
    if len(node) == 1:
        return next(iter(node))
    return "{" + ", ".join(map(str, node)) + "}"
