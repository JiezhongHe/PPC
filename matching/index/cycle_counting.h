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

// class label_path{
// public:
//     vector<Label> content;

//     label_path(vector<Label> c){
//         copy(c.begin(), c.end(), inserter(content, content.begin()));
//     }

//     label_path(const label_path& l){
//         content = l.content;
//     }

//     void push_back(const Label& v){
//         content.push_back(v);
//     }

//     void reverse_content(){
//         reverse(content.begin(), content.end());
//     }

//     bool operator==(const label_path& c) const{
//         if(c.content.size() != content.size()){
//             return false;
//         }
//         for(int i=0;i<content.size();++i){
//             if(content[i] != c.content[i]){
//                 return false;
//             }
//         }
//         return true;
//     }
// };

// struct hash_label_path 
// {
//     size_t operator()(const label_path& c) const {
//         if(c.content.size() == 0){
//             return 0;
//         }
//         size_t h = hash<int>()(c.content[0]);
//         for(int i=1; i<c.content.size(); ++i){
//             h = h^hash<int>()(c.content[i]);
//         }
//         return h;
//     }
// };

bool reverse_path(vector<Label>& path, vector<Label>& reversed_path);


class Cycle_counter:public feature_counter{
public:
    bool enable_residual;
    int feature_type = 1; //cycle

    vector<vector<Label>> slot_labels;
    vector<vector<Label>> original_features;
    vector<vector<Label>> counting_features;
    vector<vector<Label>> extended_features;

    vector<unordered_map<Label, vector<Value>>> mask_maps;

    // utilized for edge
    vector<uint32_t> extended_slot_map; // if the j-th feature is a mirror feature, the value is the position of the original feature

    // utilized for vertex
    vector<Label> front_feature_vertex;
    
    Cycle_counter(bool enable_residual_, vector<vector<Label> >& original_features_);
    Cycle_counter(const Cycle_counter& other);
    Cycle_counter(feature_counter* other);

    int get_feature_type();
    vector<vector<Label>> get_features();
    bool get_residual();

    void feature_initialization();
    void construct_mask_map();

    Tensor* merge_cycle_tensor_for_edges(Tensor* cycle_tensor, Tensor* reversed_cycle_tensor);
    Tensor* merge_cycle_tensor_for_vertices(Graph& graph, Tensor* cycle_tensor, Tensor* reversed_cycle_tensor, unordered_map<Label, vector<Value>>& mask_map);

    void count_for_vertices(Graph& graph, Tensor**& result, int& result_size);
    void count_for_edges(Graph& graph, Tensor**& result, int& result_size);
};
