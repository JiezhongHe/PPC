#pragma once
#include <bitset>
#include <cstdlib>

#include "../utility/embedding.h"
#include "../graph/graph.h"
#include "feature_counter.h"
#include "cycle_counting.h"
#include "path_counting.h"
#include "index.h"

void dump_features(vector<vector<Label>> features, string filename);

class Feature_selector{
public:
    int num_labels;
    int feature_num;
    int feature_length;
    bool enable_cycle;
    string tmp_path;

    Feature_selector();
    Feature_selector(int num_labels_, int feature_num_, int feature_length_, bool enable_cycle_, string tmp_path_);

    string generate_tmp_files();

    vector<int> select_by_complementariness(vector<bitset<MAX_SAMPLE_NUM>>& result_bit, int sample_num);

    vector<vector<Label>> extract_for_singe_graph(vector<Graph>& sample_queries, vector<vector<Vertex>>& sample_query_anchors, vector<vector<Vertex>>& sample_data_anchors, Graph& data_graph, int level, int max_batch_size, int thread_num=1);
    vector<vector<Label>> extract_for_multi_graph(vector<Graph>& sample_queries, vector<vector<Vertex>>& sample_query_anchors, vector<vector<Vertex>>& sample_data_anchors, vector<Vertex>& data_graph_ids, vector<Graph>& data_graphs, int level, int max_batch_size, int thread_num=1);
};
