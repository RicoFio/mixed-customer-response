"""
Module for handling pre-orders and partial orders.
From https://github.com/mit-zardini-lab/PosetalGameLearningPriority/blob/main/orders.py
By Yujun Huang
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, TypeAlias

import networkx as nx


Relation: TypeAlias = tuple[Any, Any]
"""Where (a, b) implies a ⪯ b"""
HasseNode: TypeAlias = frozenset[Any]
"""One node of a Hasse Diagram"""

@dataclass
class PreOrder:
    """
    Represents a pre-order (reflexive and transitive relation).
    Stored as a dictionary of comparable pairs.
    """

    elements: set[Any]
    relations: set[Relation]
    """(a, b) means a ⪯ b"""
    hasse_diagram: nx.DiGraph = field(init=False, default_factory=nx.DiGraph)
    """One node per equivalence class, each equivalence class is a set of elements"""

    def __post_init__(self) -> None:
        self.relations = self._transitive_closure()
        self.relations = self._reflexive_closure()
        self._build_hasse_diagram()
        self._validate()

    def is_degenerate(self) -> bool:
        return len(self.elements) == 1 and len(self.relations) == 1

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

    def draw_hasse_diagram(
        self,
        ax: Any | None = None,
        *,
        greatest_on_top: bool = True,
        show: bool = True,
        title: str | None = None,
    ) -> Any:
        """Draw the Hasse diagram.

        Args:
            ax: Optional matplotlib axis to draw on.
            greatest_on_top: If True, larger elements are shown higher.
            show: If True and no axis is provided, call plt.show().
            title: Optional title override.

        Returns:
            The matplotlib axis used for plotting.
        """
        import matplotlib.pyplot as plt

        if self.hasse_diagram.number_of_nodes() == 0:
            if ax is None:
                _, ax = plt.subplots(figsize=(6, 4))
            ax.set_axis_off()
            ax.set_title(title or "Empty Hasse diagram")
            if show:
                plt.show()
            return ax

        generations = list(nx.topological_generations(self.hasse_diagram))
        node_labels = {
            node: str(_format_hasse_node(node)).replace("_", "\n")
            for node in self.hasse_diagram.nodes()
        }

        pos: dict[HasseNode, tuple[float, float]] = {}
        for layer_idx, layer in enumerate(generations):
            ordered_layer = sorted(layer, key=lambda node: node_labels[node])
            width = max(1, len(ordered_layer) - 1)
            for col_idx, node in enumerate(ordered_layer):
                x = 0.0 if len(ordered_layer) == 1 else col_idx / width
                y = float(layer_idx)
                pos[node] = (x, y if greatest_on_top else -y)

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        nx.draw_networkx_nodes(
            self.hasse_diagram,
            pos,
            ax=ax,
            node_size=3400,
            node_shape="s",
            node_color="#FFFFFF",
            edgecolors="#FFFFFF",
            linewidths=1.25,
        )
        nx.draw_networkx_edges(
            self.hasse_diagram,
            pos,
            ax=ax,
            arrows=True,
            arrowstyle="-",
            arrowsize=16,
            width=2,
            edge_color="#000000",
        )
        nx.draw_networkx_labels(
            self.hasse_diagram,
            pos,
            labels=node_labels,
            ax=ax,
            font_size=18,
            font_weight='bold'
        )

        ax.set_axis_off()
        if title is None:
            direction = "maximal elements on top" if greatest_on_top else "minimal elements on top"
            title = f"Hasse diagram ({direction})"
        ax.set_title(title)

        if show and ax.figure:
            ax.figure.tight_layout()
            plt.show()

        return ax


def _format_hasse_node(node: HasseNode) -> Any:
    if len(node) == 1:
        return next(iter(node))
    return "{" + ", ".join(map(str, node)) + "}"
