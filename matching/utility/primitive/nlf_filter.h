#ifndef SUBGRAPHMATCHING_NLF_FILTER_H
#define SUBGRAPHMATCHING_NLF_FILTER_H

#include "../../graph/graph.h"
#include "../relation/catalog.h"
#include "../../index/embedding.h"
#include <vector>

class nlf_filter {
private:
    Graph* query_graph_;
    Graph* data_graph_;
    std::vector<char> status_;
    std::vector<uint32_t> updated_;

private:
    void filter_ordered_relation(uint32_t u, edge_relation* relation, uint32_t other);
    void filter_unordered_relation(uint32_t u, edge_relation* relation, uint32_t other);
public:
    nlf_filter(Graph* query_graph, Graph* data_graph) {
        query_graph_ = query_graph;
        data_graph_ = data_graph;
        status_.resize(data_graph->getVerticesCount(), 'u');
        updated_.reserve(1024);
    }

    void execute(std::vector<std::vector<uint32_t>> &candidate_sets);
    void execute(catalog* storage);
};


#endif //SUBGRAPHMATCHING_NLF_FILTER_H
