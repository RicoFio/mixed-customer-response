#ifndef MCR_AVINFRA_PERSUASION_MOSP_ADAPTER_H
#define MCR_AVINFRA_PERSUASION_MOSP_ADAPTER_H

#include <cstdint>
#include <vector>

namespace mcr_mosp {

struct PathResult {
    std::vector<std::uint32_t> node_ids;
    std::vector<std::uint32_t> arc_ids;
    std::vector<std::uint32_t> costs;
};

struct RunStats {
    std::uint64_t iterations{0};
    std::uint64_t extractions{0};
    std::uint64_t permanents{0};
    std::uint64_t solutions_count{0};
    std::uint64_t max_heap_size{0};
    std::uint64_t memory_consumption{0};
    double duration{0.0};
};

struct MospResult {
    std::vector<PathResult> paths;
    RunStats stats;
};

int dimension();

MospResult run_mda(
    std::uint32_t node_count,
    const std::vector<std::uint32_t>& tails,
    const std::vector<std::uint32_t>& heads,
    const std::vector<std::uint32_t>& flat_costs,
    std::uint32_t source,
    std::uint32_t target
);

}  // namespace mcr_mosp

#endif
