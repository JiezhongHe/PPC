#include "embedding.h"

string vertex_path_index, edge_path_index, vertex_cycle_index, edge_cycle_index;
Tensor *query_vertex_emb, *data_vertex_emb, *query_edge_emb, *data_edge_emb;
CompactTensor *query_vertex_emb_comp, *data_vertex_emb_comp, *query_edge_emb_comp, *data_edge_emb_comp;
pair<vector<int>, vector<float>> query_gnn_emb, data_gnn_emb;

inline void sum_safe_without_overflow(Value& a, Value& b){
    Value tmp = a+b;
    if(tmp<a || tmp<b){
        a = 0xffffffff;
    }else{
        a = tmp;
    }
}

bool cmp(const pair<Value, Value>& a, const pair<Value, Value>& b){
    return a.first < b.first;
}

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
#if AVX == 2 
    size_t value_bitsize = sizeof(Value)*8;
    int op = 6; // _MM_CMPINT_NLE
    for(int i=0; i<size*value_bitsize/512; ++i){
        if(value_bitsize == 16){
            __m512i src;
            __mmask32 k = -1;
            __m512i t1_vec = _mm512_mask_loadu_epi16(src, k, (__m512i*)t1+i);
            __m512i t2_vec = _mm512_mask_loadu_epi16(src, k, (__m512i*)t2+i);
            __mmask32 cmp_result = _mm512_cmp_epu16_mask(t1_vec, t2_vec, op);
            
            if(cmp_result > 0){
                return false;
            }
        }else if(value_bitsize == 32){
            __m512i t1_vec = _mm512_loadu_epi32((__m512i*)t1+i);
            __m512i t2_vec = _mm512_loadu_epi32((__m512i*)t2+i);
            __mmask16 cmp_result = _mm512_cmp_epu32_mask(t1_vec, t2_vec, op);

            if(cmp_result > 0){
                return false;
            }
        }else{
            cout<<"value type not supported"<<endl;
            exit(0);
        }
    }
    int remains = (size*value_bitsize%512)/value_bitsize;
    if(remains>0){
        int offset = size*value_bitsize/512;
        if(value_bitsize == 16){
            __mmask32 mask = 1<<remains-1;
            __m512i src = _mm512_set_epi64(0,0,0,0,0,0,0,0);
            __m512i t1_vec = _mm512_mask_expandloadu_epi16(src, mask, (__m512i*)t1+offset);
            __m512i t2_vec = _mm512_mask_expandloadu_epi16(src, mask, (__m512i*)t2+offset);
            __mmask32 result = _mm512_cmp_epi16_mask(t1_vec, t2_vec, op);
            
            if(result > 0){
                return false;
            }
        }else if(value_bitsize == 32){
            __mmask16 mask = 1<<remains-1;
            __m512i src = _mm512_set_epi64(0,0,0,0,0,0,0,0);
            __m512i t1_vec = _mm512_mask_expandloadu_epi32(src, mask, (__m512i*)t1+offset);
            __m512i t2_vec = _mm512_mask_expandloadu_epi32(src, mask, (__m512i*)t2+offset);
            __mmask16 result = _mm512_cmp_epu32_mask(t1_vec, t2_vec, op);
            
            if(result > 0){
                return false;
            }
        }else{
            cout<<"value type not supported"<<endl;
            exit(0);
        }
    }
    return true;
#elif AVX==1
    size_t value_bitsize = sizeof(Value)*8;
    for(int i=0; i<size*value_bitsize/256; ++i){
        if(value_bitsize == 16){
            __m256i src;
            __mmask16 k = -1;
            __m256i t1_vec = _mm256_mask_loadu_epi16(src, k, (__m256i*)t1+i);
            __m256i t2_vec = _mm256_mask_loadu_epi16(src, k, (__m256i*)t2+i);
            __mmask16 cmp_result = _mm256_cmp_epu16_mask(t1_vec, t2_vec, 6);
            
            if(cmp_result > 0){
                return false;
            }
        }else if(value_bitsize == 32){
            __m256i t1_vec = _mm256_loadu_si256((__m256i*)t1+i);
            __m256i t2_vec = _mm256_loadu_si256((__m256i*)t2+i);
            __mmask8 cmp_result = _mm256_cmp_epu32_mask(t1_vec, t2_vec, 6);

            if(cmp_result > 0){
                return false;
            }
        }else{
            cout<<"value type not supported"<<endl;
            exit(0);
        }
    }
    int remains = (size*value_bitsize%256)/value_bitsize;
    if(remains>0){
        int offset = size*value_bitsize/256;
        if(value_bitsize == 16){
            __mmask16 mask = 1<<remains-1;
            __m256i src = _mm256_set_epi32(0,0,0,0,0,0,0,0);
            __m256i t1_vec = _mm256_mask_expandloadu_epi16(src, mask, (__m256i*)t1+offset);
            __m256i t2_vec = _mm256_mask_expandloadu_epi16(src, mask, (__m256i*)t2+offset);
            __mmask32 result = _mm256_cmp_epi16_mask(t1_vec, t2_vec, 6);
            
            if(result > 0){
                return false;
            }
        }else if(value_bitsize == 32){
            __mmask8 mask = 1<<remains-1;
            __m256i src = _mm256_set_epi32(0,0,0,0,0,0,0,0);
            __m256i t1_vec = _mm256_mask_expandloadu_epi32(src, mask, (__m256i*)t1+offset);
            __m256i t2_vec = _mm256_mask_expandloadu_epi32(src, mask, (__m256i*)t2+offset);
            __mmask8 result = _mm256_cmp_epu32_mask(t1_vec, t2_vec, 6);
            
            if(result > 0){
                return false;
            }
        }else{
            cout<<"value type not supported"<<endl;
            exit(0);
        }
    }
    return true;
#else
    for(int i=0;i<size;++i){
        if(t1[i]>t2[i]){
            return false;
        }
    }
    return true;
#endif
}

bool valid_gnn_embedding(float* query_emb, float* data_emb, int size){
    float result = 0;
    for(int i=0;i<size;++i){
        float tmp = query_emb[i]-data_emb[i];
        if(tmp > 0){
            result += tmp*tmp;
        }
    }
    return result < 6.4; // 6.4 is the margin size
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

vector<vector<Label>> load_label_path(string file_name){
	vector<vector<Label>> result;
	ifstream fin(file_name);
	char c;
	int label;
	string tmp;
	vector<Label> vec;
	while(fin>>c){
		if(c=='['){
			continue;
		}else if(c==',' || c==' '){
			if(tmp.size() > 0){
				vec.push_back(atoi(tmp.c_str()));
			}
			tmp = "";
		}else if(c==']'){
			if(tmp.size() > 0){
				vec.push_back(atoi(tmp.c_str()));
			}
			tmp = "";
			if(vec.size() > 0){
				result.push_back(vec);
			}
			vec.clear();
		}else{
			string s(1, c);
			tmp += s;
		}
	}
	return result;
}

////////////////////////////////
Direct_IO_reader::Direct_IO_reader(string filename_, int buffer_size_){
	filename = filename_;
	page_size = getpagesize();
	buffer_size = (buffer_size_/page_size)*page_size;
	ret = 0;
	// get file size
	ifstream fin(filename);
	fin.seekg(0, ios::end);
	file_size = fin.tellg();
	fin.close();
	fd = open(filename.c_str(), O_RDWR | O_DIRECT, 0644);
	if(fd < 0){
		cout<<"Failed to open file:"<<filename<<endl;
	}
	// allocat the buffer
	read_buffer = (char*)memalign(page_size, buffer_size);
	if(read_buffer == NULL){
		cout<<"Failed to allocat buffers"<<endl;
	}
	
	ptr = 0;
	// prefetch the data
	read(fd, read_buffer, buffer_size);
	file_ptr += buffer_size;
}

void Direct_IO_reader::read_file(char* buf, size_t size){
	int remaining_size = size;
	int offset = 0;
	while(remaining_size > 0){
		int remaining_buf_size = buffer_size-ptr;
		if(remaining_size >= remaining_buf_size){
			memcpy(buf+offset, read_buffer+ptr, remaining_buf_size);
			ptr = 0;
			read(fd, read_buffer, buffer_size);
			file_ptr += buffer_size;
			remaining_size -= remaining_buf_size;
			offset += remaining_buf_size;
		}else{
			memcpy(buf+offset, read_buffer+ptr, remaining_size);
			ptr += remaining_size;
			remaining_size = 0;
		}
	}
}

bool Direct_IO_reader::is_file_end(){
	if(file_ptr >= file_size){
		return true;
	}
	return false;
}

Direct_IO_reader::~Direct_IO_reader(){
	close(fd);
	delete [] read_buffer;
}

////////////////////////////////////////////////
Tensor::Tensor(int row, int column, Value value){
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

Tensor::Tensor(Tensor* t){
    row_size = t->row_size;
    column_size = t->column_size;
    content = new Value* [row_size];
    for(int i=0;i<row_size;++i){
        content[i] = new Value [column_size];
        memcpy(content[i], t->content[i], sizeof(Value)*column_size);
    }
}

Tensor::Tensor(int row){
    row_size = row;
    column_size = 0;
    content = new Value* [row];
}

void Tensor::extract_row(int i, vector<Value>& result){
    result.resize(column_size);
    memcpy(&(result[0]), content[i], sizeof(Value)*column_size);
}

Value Tensor::get_value(int i, int j){
    return content[i][j];
}

string Tensor::to_string(){
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

void Tensor::dump_tensor(ofstream& fout){
    fout.write((char*)&(row_size), sizeof(int));
    fout.write((char*)&(column_size), sizeof(int));
    for(int i=0;i<row_size;++i){
        dump_vector(fout, content[i], column_size);
    }
}

void Tensor::concat_with(Tensor* tensor){
    assert(row_size == tensor->row_size);
    for(int i=0;i<row_size;++i){
        vector_concat(content[i], tensor->content[i], column_size, tensor->column_size);
    }
}

void Tensor::add_mul_with(Tensor* t1, Tensor* t2){
    assert(column_size == t1->column_size);
    for(int i=0;i<row_size;++i){
        vector_add_mul(content[i], t1->content[i], t2->content[i], column_size);
    }
}

void Tensor::add_with(Tensor* t1){
    assert(column_size == t1->column_size);
    for(int i=0;i<row_size;++i){
        vector_add(content[i], t1->content[i], column_size);
    }
}

void Tensor::add_shift_with(Tensor* t1, int shift){
    assert(row_size == t1->row_size);
    assert(column_size >= t1->column_size+shift);
    for(int i=0;i<row_size;++i){
        vector_add_shift(content[i], t1->content[i], t1->column_size, shift);
    }
}

void Tensor::clear_content(){
    for(int i=0;i<row_size;++i){
        memset(content[i], 0, sizeof(Value)*column_size);
    }
}

Tensor::~Tensor(){
    if(column_size > 0){
        for(int i=0;i<row_size;++i){
            delete [] content[i];
        }
    }
    delete [] content;
}

////////////////////////////////////////////
Index_manager::Index_manager(){};

Index_manager::Index_manager(string filename_){
    ifstream fin(filename_);
    is_scaned = false;
    filename = filename_;
    fin.close();
}

void Index_manager::dump_tensor(Tensor* tensor){
    is_scaned = false;
    ofstream fout(filename, ios::binary|ios::app);
    tensor->dump_tensor(fout);
    fout.close();
}

Value* Index_manager::load_embedding(ifstream& fin, Value dim){
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

vector<Tensor*> Index_manager::load_all_graphs(){
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

CompactTensor::CompactTensor(int row_size_){
    row_size = row_size_;
    content = new Value* [row_size];
}

CompactTensor::~CompactTensor(){
    for(int i=0;i<row_size;++i){
        delete content[i];
    }
}

bool compact_vec_validation(Value* vec1, Value* vec2){
    Value vec1_size = vec1[0];
    Value vec2_size = vec2[0];
    if(vec1_size > vec2_size){
        return false;
    }
    Value* vec1_key = vec1+1;
    Value* vec2_key = vec2+1;
    Value* vec1_value = vec1+1+vec1_size;
    Value* vec2_value = vec2+1+vec2_size;
    for(int i=0,j=0;i<vec1_size && j<vec2_size;){
        if(vec1_key[i] == vec2_key[j]){
            if(vec1_value[i] > vec2_value[j]){
                return false;
            }
            i++;
            j++;
        }else if(vec1_key[i] < vec2_key[j]){
            return false;
        }else{
            j++;
        }
    }
    return true;
}

CompactTensor* merge_bi_CompactTensors(CompactTensor* ct1, CompactTensor* ct2){
    int row_size = ct1->row_size;
    CompactTensor* result = new CompactTensor(row_size);
    Value** result_content = result->content;
    Value** ct1_content = ct1->content;
    Value** ct2_content = ct2->content;
    int ct1_column_size = ct1->column_size;
    int v1_size;
    int v2_size;
    for(int i=0;i<row_size;++i){
        v1_size = ct1_content[i][0];
        v2_size = ct2_content[i][0];
        result_content[i] = new Value [1+2*v1_size+2*v2_size];
        result_content[i][0] = v1_size + v2_size;
        Value* start = result_content[i]+1;
        
        memcpy(start, ct1_content[i]+1, sizeof(Value)*v1_size);
        start += v1_size;
        memcpy(start, ct2_content[i]+1, sizeof(Value)*v2_size);
        for(int x=0;x<v2_size;++x){
            start[x] += ct1_column_size;
        }
        start += v2_size;
        memcpy(start, ct1_content[i]+v1_size+1, sizeof(Value)*v1_size);
        start += v1_size;
        memcpy(start, ct2_content[i]+v2_size+1, sizeof(Value)*v2_size);
    }

    return result;
}

CompactTensor* merge_bi_CompactTensors(Tensor* t1, Tensor* t2){
    int row_size = t1->row_size;
    CompactTensor* result = new CompactTensor(row_size);
    Value** result_content = result->content;
    Value** t1_content = t1->content;
    Value** t2_content = t2->content;
    vector<Value> tmp_key;
    vector<Value> tmp_value;
    int column_size1 = t1->column_size;
    int column_size2 = t2->column_size;
    for(int i=0;i<row_size;++i){
        tmp_key.clear();
        tmp_value.clear();
        for(int j=0;j<column_size1;++j){
            if(t1_content[i][j] > 0){
                tmp_key.push_back(j);
                tmp_value.push_back(t1_content[i][j]);
            }
        }
        for(int j=0;j<column_size2;++j){
            if(t2_content[i][j] > 0){
                tmp_key.push_back(j+column_size1);
                tmp_value.push_back(t2_content[i][j]);
            }
        }
        int content_size = tmp_key.size();
        result_content[i] = new Value [1+2*content_size];
        result_content[i][0] = content_size;
        memcpy(result_content[i]+1, &(tmp_key[0]), sizeof(Value)*content_size);
        memcpy(result_content[i]+1+content_size, &(tmp_value[0]), sizeof(Value)*content_size);
    }
    return result;
}

CompactTensor* Index_manager::load_graph_compact_tensor(int graph_offset){
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
    CompactTensor* emb = new CompactTensor(row_size);
    emb->column_size = dim;
    for(int i=0;i<row_size;++i){
        emb->content[i] = load_compact_embedding(fin, dim);
    }
    fin.close();
    return emb;
}

Value* Index_manager::load_compact_embedding(ifstream& fin, Value dim){
    bool is_sparse;
    fin.read((char*)&is_sparse, sizeof(bool));
    if(is_sparse){
        int content_size;
        fin.read((char*)&content_size, sizeof(int));
        Value* result = new Value[1+2*content_size];

        // vector<Value> idx(content_size, 0);
        // vector<Value> val(content_size, 0);
        // fin.read((char*)&(idx[0]), content_size*sizeof(Value));
        // fin.read((char*)&(val[0]), content_size*sizeof(Value));
        result[0] = content_size;
        fin.read((char*)(result+1), content_size*2*sizeof(Value));
        return result;
    }else{
        Value* tmp = new Value[dim];
        fin.read((char*)tmp, dim*sizeof(Value));
        
        vector<Value> tmp_key;
        vector<Value> tmp_value;
        for(Value i=0;i<dim;++i){
            if(tmp[i] > 0){
                tmp_key.push_back(i);
                tmp_value.push_back(tmp[i]);
            }
        }
        delete tmp;
        int content_size = tmp_key.size();
        Value* result = new Value[1+2*content_size];
        result[0] = content_size;
        memcpy(result+1, &(tmp_key[0]), content_size*sizeof(Value));
        memcpy(result+1+content_size, &(tmp_value[0]), content_size*sizeof(Value));
        return result;
    }
}

Tensor* Index_manager::load_graph_tensor(int graph_offset){
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

pair<int, int> Index_manager::get_shape_of_index(int graph_offset){
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
void Index_manager::load_vertex_embedding(int graph_offset, Vertex v, vector<Value>& result){
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

void Index_manager::quick_scan(){
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

///////////////////////////////////////
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
