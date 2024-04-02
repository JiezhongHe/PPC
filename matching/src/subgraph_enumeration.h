#pragma once
#include "../graph/graph.h"
#include "../configuration/config.h"
#include "preprocessor.h"
// #include "encoder.h"
#include "trie_encoder.h"
#include "query_plan_generator.h"
#include "../utility/utils.h"

#if PRINT_MEM_INFO == 1
#include <thread>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#include <unistd.h>

#define VMRSS_LINE 22
#define PROCESS_ITEM 14

inline int GetCurrentPid();

inline float GetMemoryUsage(int pid);

void thread_get_mem_info(float& peak_memory, bool& stop);
#endif

enum LeafStateType{
    CONFLICT, EMPTYSET, SUCCESSOR_EQ_CACHE, SUBTREE_REDUCTION, PGHOLE_FILTERING, FAILING_SETS, RESULT
};


class SubgraphEnum{
public:
    Graph* data_graph_;
    Graph* query_graph_;
    vector<Vertex> order_;
    vector<Vertex> order_index_;

    uint32_t time_limit_; // seconds

    // enumeration metrics
    long count_limit_;
    long emb_count_;
    long long state_count_;
    double enumeration_time_;
    double preprocessing_time_;
    double order_adjust_time_;
    double ordering_time_;
    double query_time_;
    float peak_memory_;

    vector<uint64_t> leaf_states_counter_;

    SubgraphEnum(Graph* data_graph);

    void match(Graph* query_graph, string ordering_method, long count_limit, uint32_t time_limit);

private:
    catalog* storage_;
    preprocessor* pp_;
    uint32_t query_vertex_count_, data_vertex_count_;

    bool stop_;
    
    vector<bitset<MAX_QUERY_SIZE>> ancestors_depth_;
    vector<vector<uint32_t>> successor_neighbors_in_depth_;
    vector<vector<uint32_t>> predecessor_neighbors_in_depth_;

    vector<vector<Vertex>> matches_;

    vector<vector<vector<Vertex>>> candidates_stack_;
    vector<vector<Vertex>> searching_candidates_;
    vector<FailingSet> search_failing_set_recorder_;

    bitset<MAX_QUERY_SIZE> full_descendent_; // for extended subtree reduction
    vector<vector<bitset<MAX_QUERY_SIZE>>> parent_failing_set_map_;

    Vertex* candidates_offset_;
    vector<Vertex> embedding_depth_;
    // bool* visited_vertices_;
    // Vertex* visited_query_vertices_;
    Vertex* visited_query_depth_;
    bool* find_matches_;


    bitset<MAX_QUERY_SIZE> check_neighbor_conflict_;
    unordered_set<Vertex> neighbor_candidates_;
    vector<vector<vector<Vertex>>> unmatched_group;
    vector<vector<vector<Vertex>>> matched_group;
    vector<Vertex> conflict_checking_order;

    vector<vector<vector<Vertex>>> successor_with_same_label_;
    
    void initialization();

    void order_adjustment();
};