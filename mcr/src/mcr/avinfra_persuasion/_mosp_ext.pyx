# distutils: language = c++
# cython: language_level=3

from libc.stdint cimport uint32_t, uint64_t
from libcpp.vector cimport vector


cdef extern from "mosp_adapter.h":
    cdef cppclass PathResult "mcr_mosp::PathResult":
        vector[uint32_t] node_ids
        vector[uint32_t] arc_ids
        vector[uint32_t] costs

    cdef cppclass RunStats "mcr_mosp::RunStats":
        uint64_t iterations
        uint64_t extractions
        uint64_t permanents
        uint64_t solutions_count
        uint64_t max_heap_size
        uint64_t memory_consumption
        double duration

    cdef cppclass MospResult "mcr_mosp::MospResult":
        vector[PathResult] paths
        RunStats stats

    int cpp_dimension "mcr_mosp::dimension"()

    MospResult cpp_run_mda "mcr_mosp::run_mda"(
        uint32_t node_count,
        const vector[uint32_t]& tails,
        const vector[uint32_t]& heads,
        const vector[uint32_t]& flat_costs,
        uint32_t source,
        uint32_t target
    ) except +


cdef uint32_t _as_uint32(object value, str name):
    if value < 0 or value > 0xFFFFFFFF:
        raise OverflowError(f"{name} must fit in uint32_t.")
    return <uint32_t>value


cdef vector[uint32_t] _uint32_vector(object values, str name):
    cdef vector[uint32_t] converted
    cdef object value
    for value in values:
        converted.push_back(_as_uint32(value, name))
    return converted


def compiled_dimension():
    return int(cpp_dimension())


def run_mda(node_count, tails, heads, flat_costs, source, target):
    cdef vector[uint32_t] c_tails = _uint32_vector(tails, "tail")
    cdef vector[uint32_t] c_heads = _uint32_vector(heads, "head")
    cdef vector[uint32_t] c_flat_costs = _uint32_vector(flat_costs, "cost")
    cdef MospResult result = cpp_run_mda(
        _as_uint32(node_count, "node_count"),
        c_tails,
        c_heads,
        c_flat_costs,
        _as_uint32(source, "source"),
        _as_uint32(target, "target"),
    )

    cdef size_t path_idx
    cdef size_t value_idx
    paths = []
    for path_idx in range(result.paths.size()):
        node_ids = []
        for value_idx in range(result.paths[path_idx].node_ids.size()):
            node_ids.append(int(result.paths[path_idx].node_ids[value_idx]))

        arc_ids = []
        for value_idx in range(result.paths[path_idx].arc_ids.size()):
            arc_ids.append(int(result.paths[path_idx].arc_ids[value_idx]))

        costs = []
        for value_idx in range(result.paths[path_idx].costs.size()):
            costs.append(int(result.paths[path_idx].costs[value_idx]))

        paths.append(
            {
                "node_ids": tuple(node_ids),
                "arc_ids": tuple(arc_ids),
                "costs": tuple(costs),
            }
        )

    return {
        "stats": {
            "iterations": int(result.stats.iterations),
            "extractions": int(result.stats.extractions),
            "permanents": int(result.stats.permanents),
            "solutions_count": int(result.stats.solutions_count),
            "max_heap_size": int(result.stats.max_heap_size),
            "memory_consumption": int(result.stats.memory_consumption),
            "duration": float(result.stats.duration),
        },
        "paths": tuple(paths),
    }
