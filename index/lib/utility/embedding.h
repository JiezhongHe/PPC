#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <assert.h>
#include <string.h>
#include <bits/stdc++.h>
#include <sys/time.h>

#include <immintrin.h>
#include <x86intrin.h>

#include "../configuration/config.h"

#define OPTIMIZE 0

using namespace std;


bool cmp(const pair<Value, Value>& a, const pair<Value, Value>& b);

// utilities
void dump_vector(ofstream& fout, Value* content, int dim);
void vector_concat(Value*& t1, Value* t2, int size_1, int size_2);
void vector_add_mul(Value*& t1, Value* t2, Value* t3, int size);
void vector_add(Value*& t1, Value* t2, int size);
void vector_add_shift(Value*& t1, Value* t2, int size, int shift);
bool vec_validation(Value* t1, Value* t2, int size);
string vector_to_string(Value* vec, int size);
string vector_to_string(vector<Value>& vec);

// class
class Tensor{
public:
    Value** content=NULL;
    int row_size;
    int column_size;

    Tensor(int row, int column, Value value=0);
    Tensor(Tensor* t);
    Tensor(int row);

    void extract_row(int i, vector<Value>& result);
    Value get_value(int i, int j);

    string to_string();

    void dump_tensor(ofstream& fout);

    void concat_with(Tensor* tensor);
    void add_mul_with(Tensor* t1, Tensor* t2);
    void add_with(Tensor* t1);
    void add_shift_with(Tensor* t1, int shift);

    void clear_content();

    ~Tensor();
};

class Index_manager{
public:
    string filename;
    bool is_scaned;
    vector<vector<vector<size_t>>> offset_vertex_map; //0->sparse/dense 1->offset
    vector<size_t> offset_graph_map;
    vector<size_t> offset_dim_map;

    Index_manager();
    Index_manager(string filename_);
    
    // [sparse/dense (int)] [num of row (int)] [size of the row]
    void dump_tensor(Tensor* tensor);
    Value* load_embedding(ifstream& fin, Value dim);
    vector<Tensor*> load_all_graphs();
    Tensor* load_graph_tensor(int graph_offset);
    pair<int, int> get_shape_of_index(int graph_offset);

    // can be used to load edges
    void load_vertex_embedding(int graph_offset, Vertex v, vector<Value>& result);
    void quick_scan();
};

Tensor* sum_tensor_by_row(Tensor* tensor);

Tensor* merge_multi_Tensors(vector<Tensor*>& vec);

Tensor* merge_multi_Tensors_with_mask(vector<Tensor*>& vec, vector<vector<bool>>& mask);

void dump_index_with_mask(Tensor* tensor, string target_filename, vector<bool>& mask);

void merge_multi_index_files(vector<string>& filenames, string target_filename);

void merge_multi_index_files_with_bounded_memory(vector<string>& filenames, string target_filename);