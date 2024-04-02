#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <assert.h>
#include <string.h>
#include <bits/stdc++.h>
#include <sys/time.h>

#include "config.hpp"
#include "utils.hpp"

#define OPTIMIZE 0

using namespace std;


bool cmp(const pair<Value, Value>& a, const pair<Value, Value>& b){
    return a.first < b.first;
}


// utilities

void dump_vector(ofstream& fout, Value* content, int dim){
    uint32_t zero_count = 0;
    for(int i=0;i<dim;++i){
        if(content[i] == 0)
            ++ zero_count;
    }
    bool is_sparse = true;
    if(zero_count > dim/2){
        is_sparse = true;
        fout.write((char*)&is_sparse, sizeof(bool));
        vector<Value> val;
        vector<Value> idx;
        val.reserve(dim-zero_count);
        idx.reserve(dim-zero_count);
        for(int i=0;i<dim;++i){
            if(content[i] != 0){
                val.push_back(content[i]);
                idx.push_back(i);
            }
        }
        int size = val.size();
        fout.write((char*)&size, sizeof(int));
        if(size > 0){
            fout.write((char*)&(idx[0]), sizeof(Value)*size);
            fout.write((char*)&(val[0]), sizeof(Value)*size);
        }
    }else{
        is_sparse = false;
        fout.write((char*)&is_sparse, sizeof(bool));
        fout.write((char*)&(content[0]), sizeof(Value)*dim);
    }
}

void vector_concat(Value*& t1, Value* t2, int size_1, int size_2){
    Value* t1_tmp = t1;
    t1 = new Value [size_1+size_2];
    memcpy(t1, t1_tmp, sizeof(Value)*size_1);
    memcpy(t1+size_1, t2, sizeof(Value)*size_2);
    delete [] t1_tmp;
}

void vector_add_mul(Value*& t1, Value* t2, Value* t3, int size){
    for(int i=0;i<size;++i){
#ifdef OVERFLOW_CHECK
        Value tmp = t2[i]*t3[i];
        sum_safe_without_overflow(t1[i], tmp);
#else
        t1[i] += t2[i]*t3[i];
#endif
    }
}

void vector_add(Value*& t1, Value* t2, int size){
    for(int i=0;i<size;++i){
#ifdef OVERFLOW_CHECK
        sum_safe_without_overflow(t1[i], t2[i]);
#else
        t1[i] += t2[i];
#endif
    }
}

void vector_add_shift(Value*& t1, Value* t2, int size, int shift){
    for(int i=0;i<size;++i){
#ifdef OVERFLOW_CHECK
        sum_safe_without_overflow(t1[i+shift], t2[i]);
#else
        t1[i+shift] += t2[i];
#endif
    }
}

bool vec_validation(Value* t1, Value* t2, int size){
    for(int i=0;i<size;++i){
        if(t1[i]>t2[i]){
            return false;
        }
    }
    return true;
}

string vector_to_string(Value* vec, int size){
    stringstream ss;
    ss<<"{";
    for(int i=0; i<size; ++i){
        ss<<vec[i]<<" ";
    }
    ss<<"}\n";
    return ss.str();
}

string vector_to_string(vector<Value>& vec){
    stringstream ss;
    ss<<"{";
    for(int i=0; i<vec.size(); ++i){
        ss<<vec[i]<<" ";
    }
    ss<<"}\n";
    return ss.str();
}

double get_time(timeval& st, timeval& et){
    return (double)(et.tv_sec-st.tv_sec+(double)(et.tv_usec-st.tv_usec)/1000000);
}

class Tensor{
public:
    Value** content=NULL;
    int row_size;
    int column_size;

    Tensor(int row, int column, Value value=0){
        row_size = row;
        column_size = column;
        content = new Value* [row];
        for(int i=0;i<row;++i){
            content[i] = new Value [column];
            if(value == 0){
                memset(content[i], 0, sizeof(Value)*column);
            }else{
                for(int j=0;j<column;++j){
                    content[i][j] = value;
                }
            }
        }
    }

    Tensor(Tensor* t){
        row_size = t->row_size;
        column_size = t->column_size;
        content = new Value* [row_size];
        for(int i=0;i<row_size;++i){
            content[i] = new Value [column_size];
            memcpy(content[i], t->content[i], sizeof(Value)*column_size);
        }
    }

    void extract_row(int i, vector<Value>& result){
        result.resize(column_size);
        memcpy(&(result[0]), content[i], sizeof(Value)*column_size);
    }

    Value get_value(int i, int j){
        return content[i][j];
    }

    Tensor(int row){
        row_size = row;
        column_size = 0;
        content = new Value* [row];
    }

    string to_string(){
        stringstream ss;
        for(int i=0;i<row_size;++i){
            ss<<vector_to_string(content[i], column_size);
        }
        if(row_size > 0){
            ss<<"shape:["<<row_size<<","<<column_size<<"]"<<endl;
        }else{
            ss<<"shape:[0,0]"<<endl;
        }
        return ss.str();
    }

    ~Tensor(){
        if(column_size > 0){
            for(int i=0;i<row_size;++i){
                delete [] content[i];
            }
        }
        delete content;
    }

    void dump_tensor(ofstream& fout){
        fout.write((char*)&(row_size), sizeof(int));
        fout.write((char*)&(column_size), sizeof(int));
        for(int i=0;i<row_size;++i){
            dump_vector(fout, content[i], column_size);
        }
    }

    void concat_with(Tensor* tensor){
        assert(row_size == tensor->row_size);
        for(int i=0;i<row_size;++i){
            vector_concat(content[i], tensor->content[i], column_size, tensor->column_size);
        }
    }

    void add_mul_with(Tensor* t1, Tensor* t2){
        assert(column_size == t1->column_size);
        for(int i=0;i<row_size;++i){
            vector_add_mul(content[i], t1->content[i], t2->content[i], column_size);
        }
    }

    void add_with(Tensor* t1){
        assert(column_size == t1->column_size);
        for(int i=0;i<row_size;++i){
            vector_add(content[i], t1->content[i], column_size);
        }
    }

    void add_shift_with(Tensor* t1, int shift){
        assert(row_size == t1->row_size);
        assert(column_size >= t1->column_size+shift);
        for(int i=0;i<row_size;++i){
            vector_add_shift(content[i], t1->content[i], t1->column_size, shift);
        }
    }

    void clear_content(){
        for(int i=0;i<row_size;++i){
            memset(content[i], 0, sizeof(Value)*column_size);
        }
    }
};

class Index_manager{
public:
    string filename;
    bool is_scaned;
    vector<vector<vector<size_t>>> offset_vertex_map; //0->sparse/dense 1->offset
    vector<size_t> offset_graph_map;
    vector<size_t> offset_dim_map;

    Index_manager(string filename_){
        ifstream fin(filename_);
        is_scaned = false;
        filename = filename_;
        fin.close();
    }
    Index_manager(){};

    // [sparse/dense (int)] [num of row (int)] [size of the row]
    void dump_tensor(Tensor* tensor){
        is_scaned = false;
        ofstream fout(filename, ios::binary|ios::app);
        tensor->dump_tensor(fout);
        fout.close();
    }

    Value* load_embedding(ifstream& fin, Value dim){
        Value* result = new Value [dim];
        memset(result, 0, sizeof(Value)*dim);
        bool is_sparse;
        fin.read((char*)&is_sparse, sizeof(bool));
        if(is_sparse){
            int content_size;
            fin.read((char*)&content_size, sizeof(int));
            vector<Value> idx(content_size, 0);
            vector<Value> val(content_size, 0);
            fin.read((char*)&(idx[0]), content_size*sizeof(Value));
            fin.read((char*)&(val[0]), content_size*sizeof(Value));
            for(int j=0;j<content_size;++j){
                result[idx[j]] = val[j];
            }
        }else{
            fin.read((char*)&(result[0]), dim*sizeof(Value));
        }
        return result;
    }

    vector<Tensor*> load_all_graphs(){
        ifstream fin(filename, ios::binary);
        vector<Tensor*> results;
        while(!fin.eof()){
            int row_size, dim;
            fin.read((char*)&row_size, sizeof(int));
            if(fin.eof()){
                break;
            }
            fin.read((char*)&dim, sizeof(int));
            Tensor* emb = new Tensor(row_size);
            emb->column_size = dim;
            for(int i=0;i<row_size;++i){
                emb->content[i] = load_embedding(fin, dim);
            }
            results.push_back(emb);
        }
        return results;
    }

    Tensor* load_graph_tensor(int graph_offset){
        if(!is_scaned){
            quick_scan();
        }
        if(graph_offset>offset_graph_map.size()){
            cout<<"graph offset overflow"<<endl;
            return NULL;
        }
        long offset = offset_graph_map[graph_offset];
        ifstream fin(filename, ios::binary);
        fin.seekg(offset, ios::beg);
        int row_size, dim;
        fin.read((char*)&row_size, sizeof(int));
        fin.read((char*)&dim, sizeof(int));
        
        Tensor* emb = new Tensor(row_size);
        emb->column_size = dim;
        for(int i=0;i<row_size;++i){
            emb->content[i] = load_embedding(fin, dim);
        }
        fin.close();

        return emb;
    }

    pair<int, int> get_shape_of_index(int graph_offset){
        if(!is_scaned){
            quick_scan();
        }
        if(graph_offset>offset_graph_map.size()){
            cout<<"graph offset overflow"<<endl;
            return {0,0};
        }
        long offset = offset_graph_map[graph_offset];
        ifstream fin(filename, ios::binary);
        fin.seekg(offset, ios::beg);
        int row_size, dim;
        fin.read((char*)&row_size, sizeof(int));
        fin.read((char*)&dim, sizeof(int));
        
        return {row_size, dim};
    }

    // can be used to load edges
    void load_vertex_embedding(int graph_offset, Vertex v, vector<Value>& result){
        if(!is_scaned){
            quick_scan();
        }
        result.clear();
        
        if(graph_offset>offset_graph_map.size()){
            cout<<"graph offset overflow"<<endl;
        }
        if(v > offset_vertex_map[graph_offset].size()){
            cout<<"vertex offset overflow"<<endl;
        }
        vector<size_t>& offset = offset_vertex_map[graph_offset][v];
        ifstream fin(filename, ios::binary);
        fin.seekg(offset[1], ios::beg);

        Value* r = load_embedding(fin, offset_dim_map[graph_offset]);
        result.resize(offset_dim_map[graph_offset]);
        memcpy(&(result[0]), r, sizeof(Value)*offset_dim_map[graph_offset]);
        delete r;
        fin.close();
    }

    void quick_scan(){
        is_scaned = true;
        ifstream fin(filename, ios::binary);
        while(true){
            offset_graph_map.push_back(fin.tellg());
            int num_rows, row_size;
            fin.read((char*)&num_rows, sizeof(int));
            fin.read((char*)&row_size, sizeof(int));
            offset_dim_map.push_back(row_size);

            vector<vector<size_t>> vertex_map;
            vertex_map.reserve(num_rows);
            for(int i=0;i<num_rows; ++i){
                bool is_sparse;
                size_t offset = fin.tellg();
                fin.read((char*)&is_sparse, sizeof(bool));
                vector<size_t> record(2, 0);
                if(is_sparse){
                    int content_size;
                    fin.read((char*)&content_size, sizeof(int));
                    record[0] = 0;
                    record[1] = offset;
                    fin.seekg(sizeof(Value)*content_size*2, ios::cur);
                    vertex_map.push_back(record);
                }else{
                    record[0] = 0;
                    record[1] = offset;
                    fin.seekg(sizeof(Value)*row_size, ios::cur);
                    vertex_map.push_back(record);
                }
            }
            offset_vertex_map.push_back(vertex_map);
            long offset_end = fin.tellg();
            fin.seekg(0, ios::end);
            if(fin.tellg() == offset_end){
                break;
            }
            fin.seekg(offset_end);
        }
        fin.close();
        offset_dim_map.shrink_to_fit();
        offset_graph_map.shrink_to_fit();
        offset_vertex_map.shrink_to_fit();
    }
};

Tensor* sum_tensor_by_row(Tensor* tensor){
    int column_size = tensor->column_size;
    Tensor* result = new Tensor(1, column_size, 0);
    for(int i=0;i<tensor->row_size;++i){
        vector_add(result->content[0], tensor->content[i], column_size);
    }
    return result;
}

Tensor* merge_multi_Tensors(vector<Tensor*>& vec){
    if(vec.size() == 0){
        return NULL;
    }
    int size = 0;
    for(auto t : vec){
        size += t->column_size;
    }
    Tensor* result = new Tensor(vec[0]->row_size, size, 0);
    int shift = 0;
    Value** result_content = result->content;
    
    for(auto t : vec){
        int t_column_size = t->column_size;
        Value** t_content = t->content;
        for(int i=0;i<result->row_size;++i){
            memcpy(result_content[i]+shift, t_content[i], t_column_size*sizeof(Value));
        }
        shift += t_column_size;
    }
    return result;
}

Tensor* merge_multi_Tensors_with_mask(vector<Tensor*>& vec, vector<vector<bool>>& mask){
    assert(vec.size() == mask.size());
    int size = 0;
    vector<vector<pair<int, int>>> valid_span;
    int offset = 0;
    pair<int, int> record;
    
    for(auto m : mask){
        valid_span.push_back({});
        vector<pair<int, int>>& span_list = *(valid_span.rbegin());
        record.first = -1;
        for(int i=0;i<m.size();++i){
            if(m[i]){
                size++;
                if(record.first == -1){
                    record.first = i;
                }
            }else{
                if(record.first != -1){
                    record.second = i;
                    span_list.push_back(record);
                    record.first = -1;
                }
            }
        }
        if(record.first != -1){
            record.second = m.size();
            span_list.push_back(record);
        }
    }
    Tensor* result = new Tensor(vec[0]->row_size, size, 0);
    Value** result_content = result->content;
    int shift = 0;
    for(int j=0;j<mask.size();++j){
        Value** t_content = vec[j]->content;
        for(auto span : valid_span[j]){
            int span_size = span.second-span.first;
            for(int i=0;i<result->row_size;++i){
                memcpy(result_content[i]+shift, t_content[i]+span.first, span_size*sizeof(Value));
            }
            shift += span_size;
        }
    }
    return result;
}

void dump_index_with_mask(Tensor* tensor, string target_filename, vector<bool>& mask){
    int size = 0;
    vector<pair<int, int>> valid_span;
    int offset = 0;
    pair<int, int> record;
    
    record.first = -1;
    for(int i=0;i<mask.size();++i){
        if(mask[i]){
            size++;
            if(record.first == -1){
                record.first = i;
            }
        }else{
            if(record.first != -1){
                record.second = i;
                valid_span.push_back(record);
                record.first = -1;
            }
        }
    }
    if(record.first != -1){
        record.second = mask.size();
        valid_span.push_back(record);
    }

    Tensor* result = new Tensor(tensor->row_size, size, 0);
    int shift = 0;
    
    Value** t_content = tensor->content;
    Value** result_content = result->content;
    for(auto span : valid_span){
        int span_size = span.second-span.first;
        for(int i=0;i<result->row_size;++i){
            memcpy(result_content[i]+shift, t_content[i]+span.first, span_size*sizeof(Value));
        }
        shift += span_size;
    }
        
    Index_manager manager(target_filename);
    manager.dump_tensor(result);
    delete result;
}

void merge_multi_index_files(vector<string>& filenames, string target_filename){
    vector<Index_manager> index_parser_list;
    
    int column_size = 0;
    int row_size = 0;
    vector<ifstream> fin_list;
    int chunk_size = 40000;
    vector<Tensor*> split_chunk_vec;

    for(auto file : filenames){
        index_parser_list.emplace_back(Index_manager(file));
        fin_list.emplace_back(ifstream(file, ios::binary));
        fin_list.rbegin()->seekg(sizeof(int)*2, ios::beg);

        pair<int, int> shape = index_parser_list.rbegin()->get_shape_of_index(0);

        Tensor* tensor = new Tensor(chunk_size, shape.second);
        split_chunk_vec.push_back(tensor);
        column_size += shape.second;
        row_size = shape.first;
    }

    ofstream fout(target_filename, ios::binary);
    fout.write((char*)&row_size, sizeof(int));
    fout.write((char*)&column_size, sizeof(int));
    
    // Tensor* chunk = new Tensor(chunk_size, column_size, 0);
    Value* vector_tmp = new Value [column_size];
    for(int i=0; i<row_size;){
        int row_index = 0;
        for(int j=0; j<index_parser_list.size(); ++j){
            Value** read_tensor = split_chunk_vec[j]->content;
            int read_column_size = split_chunk_vec[j]->column_size;
            row_index = 0;
            do{
                // read the row
                ifstream& fin_row = fin_list[j];
                Value* read_row = read_tensor[row_index];
                memset(read_row, 0, sizeof(Value)*read_column_size);
                bool is_sparse;
                fin_row.read((char*)&is_sparse, sizeof(bool));
                if(is_sparse){
                    int content_size;
                    fin_row.read((char*)&content_size, sizeof(int));
                    vector<Value> idx(content_size, 0);
                    vector<Value> val(content_size, 0);
                    fin_row.read((char*)&(idx[0]), content_size*sizeof(Value));
                    fin_row.read((char*)&(val[0]), content_size*sizeof(Value));
                    for(int m=0;m<content_size;++m){
                        read_row[idx[m]] = val[m];
                    }
                }else{
                    fin_row.read((char*)&(read_row[0]), read_column_size*sizeof(Value));
                }
                
                ++ row_index;
            }while(row_index<chunk_size && row_index+i<row_size);
        }
        i += row_index;
        // add and dump the vectors
        int valid_chunk_size = chunk_size;
        if(i == row_size){
            valid_chunk_size = row_size%chunk_size;
        }
        for(int x=0; x<valid_chunk_size; ++x){
            memset(vector_tmp, 0, sizeof(Value)*column_size);
            int offset = 0;
            for(int y=0;y<index_parser_list.size();++y){
                vector_add_shift(vector_tmp, split_chunk_vec[y]->content[x], split_chunk_vec[y]->column_size, offset);
                offset += split_chunk_vec[y]->column_size;
            }
            dump_vector(fout, vector_tmp, column_size);
        }
    }
    // release
    delete [] vector_tmp;
    for(auto t : split_chunk_vec){
        delete t;
    }
    fout.close();
}

void merge_multi_index_files_with_bounded_memory(vector<string>& filenames, string target_filename){
    int column_size = 0;
    int row_size = 0;
    Direct_IO_reader** reader_list;
    int chunk_size = 100000;
    int num_indices = filenames.size();
    vector<Tensor*> split_chunk_vec;
    int o = 0;
    reader_list = new Direct_IO_reader* [num_indices];
    for(int i=0;i<num_indices;++i){
        string file = filenames[i];
        reader_list[i] = new Direct_IO_reader(file, 1024*128);

        Direct_IO_reader* reader = reader_list[i];
        int row_size_local, column_size_local;
        reader->read_file((char*)&row_size_local, sizeof(int));
        reader->read_file((char*)&column_size_local, sizeof(int));

        pair<int, int> shape;
        shape.first = row_size_local;
        shape.second = column_size_local;

        Tensor* tensor = new Tensor(chunk_size, shape.second);
        split_chunk_vec.push_back(tensor);
        column_size += shape.second;
        row_size = shape.first;
        o++;
    }

    ofstream fout(target_filename, ios::binary);
    fout.write((char*)&row_size, sizeof(int));
    fout.write((char*)&column_size, sizeof(int));
    // Tensor* chunk = new Tensor(chunk_size, column_size, 0);
    Value* vector_tmp = new Value [column_size];
    for(int i=0; i<row_size;){
        int row_index = 0;
        for(int j=0; j<num_indices; ++j){
            Value** read_tensor = split_chunk_vec[j]->content;
            int read_column_size = split_chunk_vec[j]->column_size;
            row_index = 0;
            do{
                // read the row
                Direct_IO_reader* reader = reader_list[j];

                Value* read_row = read_tensor[row_index];
                memset(read_row, 0, sizeof(Value)*read_column_size);
                bool is_sparse;
                reader->read_file((char*)&is_sparse, sizeof(bool));
                if(is_sparse){
                    int content_size;
                    reader->read_file((char*)&content_size, sizeof(int));
                    vector<Value> idx(content_size, 0);
                    vector<Value> val(content_size, 0);
                    reader->read_file((char*)&(idx[0]), content_size*sizeof(Value));
                    reader->read_file((char*)&(val[0]), content_size*sizeof(Value));
                    for(int m=0;m<content_size;++m){
                        read_row[idx[m]] = val[m];
                    }
                }else{
                    reader->read_file((char*)&(read_row[0]), read_column_size*sizeof(Value));
                }
                
                ++ row_index;
            }while(row_index<chunk_size && row_index+i<row_size);
        }
        i += row_index;
        // add and dump the vectors
        int valid_chunk_size = chunk_size;
        if(i == row_size){
            valid_chunk_size = row_size%chunk_size;
        }
        for(int x=0; x<valid_chunk_size; ++x){
            memset(vector_tmp, 0, sizeof(Value)*column_size);
            int offset = 0;
            for(int y=0;y<num_indices;++y){
                vector_add_shift(vector_tmp, split_chunk_vec[y]->content[x], split_chunk_vec[y]->column_size, offset);
                offset += split_chunk_vec[y]->column_size;
            }
            dump_vector(fout, vector_tmp, column_size);
        }
    }
    // release
    delete [] vector_tmp;
    for(auto t : split_chunk_vec){
        delete t;
    }
    for(int i=0;i<num_indices;++i){
        delete reader_list[i];
    }
    delete [] reader_list;
    fout.close();
}


