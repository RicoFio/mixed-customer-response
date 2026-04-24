# distutils: language = c++
# cython: language_level = 3

from libc.stdint cimport uint32_t, uint64_t
from libcpp.vector cimport vector


cdef extern from "mosp_adapter.h" namespace "mcr_mosp":

    cdef struct PathResult:
        vector[uint32_t] node_ids
        vector[uint32_t] arc_ids
        vector[uint32_t] costs

    cdef struct RunStats:
        uint64_t iterations
        uint64_t extractions
        uint64_t permanents
        uint64_t solutions_count
        uint64_t max_heap_size
        uint64_t memory_consumption
        double duration

    cdef struct MospResult:
        vector[PathResult] paths
        RunStats stats

    int dimension() except +

    MospResult _cpp_run_mda "mcr_mosp::run_mda"(
        uint32_t node_count,
        const vector[uint32_t]& tails,
        const vector[uint32_t]& heads,
        const vector[uint32_t]& flat_costs,
        uint32_t source,
        uint32_t target,
    ) except +


def compiled_dimension():
    """Return the objective-space dimension the extension was compiled with."""
    return dimension()


def run_mda(
    int node_count,
    list tails,
    list heads,
    list flat_costs,
    int source,
    int target,
):
    """
    Run the MDA multi-objective shortest-path algorithm.

    Returns a dict with keys ``"paths"`` and ``"stats"``.
    Each path is a dict with ``"node_ids"``, ``"arc_ids"``, and ``"costs"``.
    """
    cdef vector[uint32_t] c_tails = tails
    cdef vector[uint32_t] c_heads = heads
    cdef vector[uint32_t] c_flat_costs = flat_costs

    cdef MospResult result = _cpp_run_mda(
        <uint32_t>node_count,
        c_tails,
        c_heads,
        c_flat_costs,
        <uint32_t>source,
        <uint32_t>target,
    )

    paths = []
    cdef size_t i
    for i in range(result.paths.size()):
        pr = result.paths[i]
        paths.append({
            "node_ids": list(pr.node_ids),
            "arc_ids": list(pr.arc_ids),
            "costs": list(pr.costs),
        })

    stats = {
        "iterations": result.stats.iterations,
        "extractions": result.stats.extractions,
        "permanents": result.stats.permanents,
        "solutions_count": result.stats.solutions_count,
        "max_heap_size": result.stats.max_heap_size,
        "memory_consumption": result.stats.memory_consumption,
        "duration": result.stats.duration,
    }

    return {"paths": paths, "stats": stats}
