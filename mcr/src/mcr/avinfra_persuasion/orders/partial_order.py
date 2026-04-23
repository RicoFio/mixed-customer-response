"""
Module for handling pre-orders and partial orders.
From https://github.com/mit-zardini-lab/PosetalGameLearningPriority/blob/main/orders.py
By Yujun Huang
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Iterator
from typing import Set, Tuple, List

import networkx as nx
from .pre_order import PreOrder, Relation


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


def _dag_transitive_closure(nodes: List[Any], edges: List[Tuple[Any, Any]]) -> Set[Tuple[Any, Any]]:
    """Compute transitive closure relation set (including reflexive) for a DAG."""
    index = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    # adjacency matrix
    reach = [[False]*n for _ in range(n)]
    for i in range(n):
        reach[i][i] = True
    for u, v in edges:
        iu, iv = index[u], index[v]
        reach[iu][iv] = True
    # Floyd–Warshall for reachability
    for k in range(n):
        for i in range(n):
            if reach[i][k]:
                row_k = reach[k]
                for j in range(n):
                    if row_k[j]:
                        reach[i][j] = True
    closure = set()
    for i in range(n):
        for j in range(n):
            if reach[i][j]:
                closure.add((nodes[i], nodes[j]))
    return closure


def all_partial_orders(elements: Set[Any]) -> Iterator[PartialOrder]:
    """
    Iterate over all distinct partial orders (reflexive, antisymmetric, transitive relations)
    on the given finite set of labeled elements.

    Strategy:
    - Enumerate all directed acyclic graphs (DAGs) on the vertex set (edges taken from all ordered pairs without loops).
    - For each DAG, compute its transitive closure; this closure uniquely determines a partial order.
    - Deduplicate closures (different DAGs can represent the same partial order).
    Complexity: For n elements we examine up to 2^{n(n-1)} edge subsets, but practical up to n<=5.
    Counts produced match known sequence: 1,3,19,219,4231 for n=1..5.
    """
    if len(elements) == 0:
        return iter([])
    if len(elements) > 5:
        raise ValueError("Generating all partial orders is only feasible for sets of size <= 5")

    nodes = list(elements)  # preserve arbitrary ordering
    m = len(nodes)
    # All possible directed edges without self-loops
    possible_edges: List[Tuple[Any, Any]] = []
    for i in range(m):
        for j in range(m):
            if i != j:
                possible_edges.append((nodes[i], nodes[j]))

    num_edges = len(possible_edges)
    seen_closures: Set[frozenset] = set()

    # Bitmask enumeration over possible_edges; skip masks containing a 2-cycle early
    for mask in range(1 << num_edges):
        edges_subset: List[Tuple[Any, Any]] = []
        mutual_conflict = False
        # quick presence map to detect 2-cycles early
        present = {}
        for idx in range(num_edges):
            if mask & (1 << idx):
                u, v = possible_edges[idx]
                edges_subset.append((u, v))
                if (v, u) in present:  # 2-cycle -> not a DAG / violates antisymmetry
                    mutual_conflict = True
                    break
                present[(u, v)] = True
        if mutual_conflict:
            continue
        # Acyclic check using networkx (fast for small graphs)
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges_subset)
        if not nx.is_directed_acyclic_graph(G):
            continue
        # Compute transitive closure (includes reflexive pairs)
        closure = _dag_transitive_closure(nodes, edges_subset)
        frozen = frozenset(closure)
        if frozen in seen_closures:
            continue
        seen_closures.add(frozen)

        # Yield a PartialOrder using the full transitive closure (including reflexive pairs).
        yield PartialOrder(set(nodes), set(closure))
