#pragma once
#include "../configuration/config.h"
#include "../utility/relation/catalog.h"
#include "../graph/graph.h"
#include "../index/embedding.h"

#include <vector>
#include <unordered_map>

class TrieRelation{
public:
    uint32_t size_;
    unordered_map<Vertex, pair<Vertex*, uint32_t>> offset_map; // id -> start_offset, size
    Vertex* children_;
#if INDEX_ORDER == 1
    float* scores_;
#endif
    Vertex src_, dst_;

    TrieRelation(catalog* storage, Vertex src, Vertex dst);

    void remove_candidate(Vertex src_u, Vertex src_v, Graph* data_graph);

    ~TrieRelation();
};

class TrieEncoder{
public:
    catalog* storage_;
    vector<Vertex> order_;
    vector<Vertex> order_index_;
    Graph* query_graph_;
    Graph* data_graph_;
    vector<vector<TrieRelation*>> candidate_edges;
    TrieEncoder(catalog* storage, vector<Vertex>& order, vector<Vertex>& order_index, Graph* query_graph, Graph* data_graph);
    
    void get_candidates(uint32_t u_depth, vector<Vertex>& result);
    Vertex* get_edge_candidate(uint32_t src_depth, uint32_t dst_depth, Vertex src_v, uint32_t& count);

#if INDEX_ORDER == 1
    vector<unordered_map<Vertex, float>> scores;
    void get_candidates_ordered(uint32_t u_depth, vector<Vertex>& result);
#endif

    ~TrieEncoder();
};
