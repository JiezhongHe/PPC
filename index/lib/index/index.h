#pragma once
#include "../graph/graph.h"
#include "../utility/embedding.h"
#include "../utility/utils.h"
#include "feature_counter.h"
#include "cycle_counting.h"
#include "path_counting.h"

void print_features(vector<vector<Label>> features);

void* thread_count(void* arg);

Tensor* get_frequency_from_multi_counters(Graph& graph, vector<feature_counter*>& counter_list, int level, int thread_num=1);

typedef tuple<Graph*, pair<Vertex, Vertex>*, Vertex**, int> input_para;

void* compute_common_neighbor_func(void* args);

// compute the common edge neighbors for graph
void common_edge_neighbor_multi_threads(Graph* graph, int thread_num);

class Index_constructer{
public:
    feature_counter* counter;
    double build_time;

    Index_constructer();
    Index_constructer(feature_counter* counter_);

    void construct_index_single_batch(Graph& graph, string index_file_name, int thread_num, int level);

    // note that residual mechanism is not supported
    void construct_index_in_batch(Graph& graph, string index_file_name, int max_feature_size_per_batch, int thread_num, int level);

    void analyse_redundant_features(vector<vector<Label>>& features, vector<vector<bool>>& redundant_mask);

    Tensor* count_features(Graph& graph, int thread_num, int level);

    vector<Tensor*> count_with_multi_thread(Graph& graph, int thread_num, int level);
};
