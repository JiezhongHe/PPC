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

#include <unistd.h>


#define _GNU_SOURCE
 
#include <sys/types.h>
#include <fcntl.h>
#include <malloc.h>

#include <errno.h>

#define OPTIMIZE 0

typedef uint16_t Value;

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

class Direct_IO_reader{
public:
	char* read_buffer;
	int buffer_size;
	int fd;
	string filename;
	int ret;
	int page_size;
	int ptr;
	int global_read_offset;
	unsigned long long file_ptr;
	unsigned long long file_size;

	Direct_IO_reader(string filename_, int buffer_size_);

	void read_file(char* buf, size_t size);

	bool is_file_end();

	~Direct_IO_reader();
};

class CompactTensor{
public:
    Value** content;
    int row_size;
    int column_size;
    CompactTensor(int row_size_);
    ~CompactTensor();
};

bool compact_vec_validation(Value* vec1, Value* vec2);
CompactTensor* merge_bi_CompactTensors(CompactTensor* ct1, CompactTensor* ct2);
bool valid_gnn_embedding(float* query_emb, float* data_emb, int size);

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

CompactTensor* merge_bi_CompactTensors(Tensor* t1, Tensor* t2);

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
    Value* load_compact_embedding(ifstream& fin, Value dim);
    vector<Tensor*> load_all_graphs();
    Tensor* load_graph_tensor(int graph_offset);
    CompactTensor* load_graph_compact_tensor(int graph_offset);
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

vector<vector<Label>> load_label_path(string file_name);

extern string vertex_path_index, edge_path_index, vertex_cycle_index, edge_cycle_index;
extern Tensor *query_vertex_emb, *data_vertex_emb, *query_edge_emb, *data_edge_emb;
extern CompactTensor *query_vertex_emb_comp, *data_vertex_emb_comp, *query_edge_emb_comp, *data_edge_emb_comp;
extern pair<vector<int>, vector<float>> query_gnn_emb, data_gnn_emb;