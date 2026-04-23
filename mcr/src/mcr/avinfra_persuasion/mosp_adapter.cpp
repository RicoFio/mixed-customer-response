#include "mosp_adapter.h"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "graph.h"
#include "MDA.h"
#include "NodeInfo.h"
#include "NodeInfoContainer.h"
#include "Preprocessor.h"
#include "Solution.h"
#include "typedefs.h"

namespace {

std::unique_ptr<Graph> build_graph(
    const std::uint32_t node_count,
    const std::vector<std::uint32_t>& tails,
    const std::vector<std::uint32_t>& heads,
    const std::vector<std::uint32_t>& flat_costs,
    const std::uint32_t source,
    const std::uint32_t target
) {
    if (node_count == 0) {
        throw std::invalid_argument("node_count must be positive.");
    }
    if (source >= node_count || target >= node_count) {
        throw std::invalid_argument("source and target must be valid node ids.");
    }
    if (source == target) {
        throw std::invalid_argument("source and target must be distinct.");
    }
    if (tails.size() != heads.size()) {
        throw std::invalid_argument("tails and heads must have equal length.");
    }
    if (tails.empty()) {
        throw std::invalid_argument("at least one arc is required.");
    }
    if (flat_costs.size() != tails.size() * static_cast<std::size_t>(DIM)) {
        throw std::invalid_argument("flat_costs must contain one DIM cost vector per arc.");
    }
    if (tails.size() > static_cast<std::size_t>(INVALID_ARC / 2)) {
        throw std::overflow_error("too many arcs for mod ArcId limits.");
    }

    std::vector<NeighborhoodSize> in_degree(node_count, 0);
    std::vector<NeighborhoodSize> out_degree(node_count, 0);
    std::vector<std::pair<Node, Arc>> arcs;
    arcs.reserve(tails.size());

    for (std::size_t arc_id = 0; arc_id < tails.size(); ++arc_id) {
        const Node tail = tails[arc_id];
        const Node head = heads[arc_id];
        if (tail >= node_count || head >= node_count) {
            throw std::invalid_argument("arc endpoint outside node range.");
        }
        if (out_degree[tail] == MAX_DEGREE || in_degree[head] == MAX_DEGREE) {
            throw std::overflow_error("node degree exceeds mod NeighborhoodSize limits.");
        }

        ++out_degree[tail];
        ++in_degree[head];

        CostArray costs = generate(0);
        for (Dimension dim = 0; dim < DIM; ++dim) {
            costs[dim] = flat_costs[arc_id * static_cast<std::size_t>(DIM) + dim];
        }
        arcs.emplace_back(tail, Arc(head, costs, static_cast<ArcId>(arc_id)));
    }

    auto graph = std::make_unique<Graph>(
        node_count,
        static_cast<ArcId>(tails.size()),
        source,
        target
    );
    graph->setName("mcr_mosp");

    for (Node node = 0; node < node_count; ++node) {
        graph->setNodeInfo(node, in_degree[node], out_degree[node]);
    }

    std::vector<NeighborhoodSize> incoming_arcs_per_node(node_count, 0);
    std::vector<NeighborhoodSize> outgoing_arcs_per_node(node_count, 0);
    for (auto& arc_pair : arcs) {
        const Node tail_id = arc_pair.first;
        Arc arc = arc_pair.second;

        NodeAdjacency& tail = graph->node(tail_id);
        NodeAdjacency& head = graph->node(arc.n);

        arc.revArcIndex = incoming_arcs_per_node[head.id];
        tail.outgoingArcs[outgoing_arcs_per_node[tail.id]++] = arc;

        arc.n = tail.id;
        arc.revArcIndex = outgoing_arcs_per_node[tail.id] - 1;
        head.incomingArcs[incoming_arcs_per_node[head.id]++] = arc;
    }

    return graph;
}

}  // namespace

namespace mcr_mosp {

int dimension() {
    return static_cast<int>(DIM);
}

MospResult run_mda(
    const std::uint32_t node_count,
    const std::vector<std::uint32_t>& tails,
    const std::vector<std::uint32_t>& heads,
    const std::vector<std::uint32_t>& flat_costs,
    const std::uint32_t source,
    const std::uint32_t target
) {
    std::unique_ptr<Graph> graph_ptr = build_graph(
        node_count,
        tails,
        heads,
        flat_costs,
        source,
        target
    );
    Graph& graph = *graph_ptr;

    Preprocessor preprocessor(graph);
    NodeInfoContainer<NodeInfo> prep_info(graph);
    preprocessor.run(prep_info);

    std::vector<CostArray> potential(graph.nodesCount);
    for (Node node = 0; node < graph.nodesCount; ++node) {
        potential[node] = prep_info.getInfo(node).potential;
    }

    MDA mda{graph, potential};
    Solution solution("mcr_mosp", source, target, graph);
    mda.run(solution);

    MospResult result;
    result.stats.iterations = solution.iterations;
    result.stats.extractions = solution.extractions;
    result.stats.permanents = solution.permanents;
    result.stats.solutions_count = solution.solutionsCt;
    result.stats.max_heap_size = solution.maxHeapSize;
    result.stats.memory_consumption = solution.memoryConsumption;
    result.stats.duration = solution.duration;

    const auto extracted_paths = mda.extractPaths();
    result.paths.reserve(extracted_paths.size());
    for (const auto& extracted_path : extracted_paths) {
        PathResult path_result;
        path_result.node_ids.reserve(extracted_path.nodeIds.size());
        for (const Node node_id : extracted_path.nodeIds) {
            path_result.node_ids.push_back(node_id);
        }

        path_result.arc_ids.reserve(extracted_path.arcIds.size());
        for (const ArcId arc_id : extracted_path.arcIds) {
            path_result.arc_ids.push_back(arc_id);
        }

        path_result.costs.reserve(static_cast<std::size_t>(DIM));
        for (Dimension dim = 0; dim < DIM; ++dim) {
            path_result.costs.push_back(extracted_path.costs[dim]);
        }

        result.paths.push_back(std::move(path_result));
    }

    return result;
}

}  // namespace mcr_mosp
