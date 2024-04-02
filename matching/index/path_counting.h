#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include <cstdlib>
#include <ctime>

#include "../graph/graph.h"
#include "embedding.h"
#include "feature_counter.h"

using namespace std;


class Path_counter:public feature_counter{
public:
    bool enable_residual;
    int feature_type = 0; // path

    vector<vector<Label>> slot_labels;
    vector<vector<Label>> original_features;

    vector<unordered_map<Label, vector<Value>>> mask_maps;
    
    Path_counter(bool enable_residual_, vector<vector<Label> >& original_features_);
    Path_counter(const Path_counter& other);
    Path_counter(feature_counter* other);

    void feature_initialization();

    int get_feature_type();
    vector<vector<Label>> get_features();
    bool get_residual();

    void construct_mask_map();

    Tensor* merge_paths_for_edges(Graph& graph, Tensor* path_tensor);
    void count_for_edges(Graph& graph, Tensor**& result, int& result_size);
    void count_for_vertices(Graph& graph, Tensor**& result, int& result_size);
};
