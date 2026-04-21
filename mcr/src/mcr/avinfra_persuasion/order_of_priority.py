"""
Partial order of all possible partial orders on a given set.
Warning: This is computationally intensive for sets larger than size 5.
From https://github.com/mit-zardini-lab/PosetalGameLearningPriority/blob/main/order_of_priority.py
By Yujun Huang
"""
from typing import Set, Any, Iterator, List, Tuple
import networkx as nx

from .orders import PartialOrder


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
