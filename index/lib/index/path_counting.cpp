#include "path_counting.h"

Path_counter::Path_counter(bool enable_residual_, vector<vector<Label> >& original_features_){
    original_features = original_features_;
    enable_residual = enable_residual_;
    feature_initialization();
    construct_mask_map();
}

Path_counter::Path_counter(const Path_counter& other){
    original_features = other.original_features;
    enable_residual = other.enable_residual;
    feature_initialization();
    construct_mask_map();
}

Path_counter::Path_counter(feature_counter* other){
    original_features = other->get_features();
    enable_residual = other->get_residual();
    feature_initialization();
    construct_mask_map();
}

void Path_counter::feature_initialization(){
    slot_labels.reserve(original_features[0].size()-1);
    for(int l=original_features[0].size()-1; l>=0; --l){
        vector<Label> inner;
        inner.reserve(original_features.size());
        for(int j=0; j < original_features.size(); ++j){
            inner.push_back(original_features[j][l]);
        }
        slot_labels.push_back(inner);
    }
}

int Path_counter::get_feature_type(){
    return feature_type;
}

vector<vector<Label>> Path_counter::get_features(){
    return original_features;
}

bool Path_counter::get_residual(){
    return enable_residual;
}

void Path_counter::construct_mask_map(){
    int feature_length = original_features[0].size();
    mask_maps.reserve(feature_length);
    for(int depth=0;depth<feature_length;++depth){
        int dim = slot_labels[depth].size();
        unordered_set<Label> labels;
        unordered_map<Label, vector<Value>> mask_map;
        for(int i=0; i<dim; ++i){
            Label label = slot_labels[depth][i];
            labels.insert(label);
        }
        for(auto label : labels){
            vector<Value> mask;
            for(int i=0; i<dim; ++i){
                if(slot_labels[depth][i] == label){
                    mask.push_back(1);
                }else{
                    mask.push_back(0);
                }
            }
            mask_map.insert({label, mask});
        }
        mask_maps.push_back(mask_map);
    }
}

Tensor* Path_counter::merge_paths_for_edges(Graph& graph, Tensor* path_tensor){
    uint32_t edge_count = graph.get_edge_count();
    uint32_t feature_num = original_features.size();
    Tensor* result = new Tensor(edge_count, feature_num);
    Value** content = result->content;
    Value** path_content = path_tensor->content;
    for(Vertex e_id=0; e_id<edge_count; ++e_id){
        Vertex small_id = graph.edge_set[e_id*2];
        Vertex large_id = graph.edge_set[e_id*2+1];
        vector_add(content[e_id], path_content[small_id], feature_num);
        vector_add(content[e_id], path_content[large_id], feature_num);
    }
    return result;
}

void Path_counter::count_for_edges(Graph& graph, Tensor**& result, int& result_size){
#ifdef ENABLE_TIME_INFO
    struct timeval start_t, end_t;
#endif
    if(enable_residual){
        result_size = original_features[0].size();
    }else{
        result_size = 1;
    }
    result = new Tensor* [result_size];
    int offset = 0;
    uint32_t vertex_count = graph.adj.size();
    Tensor* tensor_prev= new Tensor(vertex_count, original_features.size(), 1); // v<-n
    Tensor* tensor_cur = new Tensor(vertex_count, original_features.size(), 0);
    int total_iterations = original_features[0].size();
    for(int iteration=0; iteration<total_iterations; ++iteration){
#ifdef ENABLE_TIME_INFO
        cout<<"iteration:"<<iteration<<endl;
        gettimeofday(&start_t, NULL);
#endif
        // initialize the mask
        unordered_map<Label, vector<Value>>& mask_map = mask_maps[iteration];
        // iterate the vertices in order
        Value** embeddings_prev = tensor_prev->content;
        for(Vertex v=0; v<vertex_count; ++v){
            Value* embedding_cur = tensor_cur->content[v];
            for(auto n : graph.adj[v]){
                auto itf = mask_map.find(graph.label_map[n]);
                if(itf == mask_map.end()){
                    continue;
                }
                vector<Value>& mask = itf->second;
                vector_add_mul(embedding_cur, embeddings_prev[n], &(mask[0]), mask.size());
            }
        }
        // swap the embedding
        tensor_prev->clear_content();
        if(enable_residual == true && iteration < total_iterations-1){
            result[offset++] = merge_paths_for_edges(graph, tensor_cur);
        }
        swap(tensor_prev, tensor_cur);
#ifdef ENABLE_TIME_INFO
        gettimeofday(&end_t, NULL);
        cout<<"iteration:"<<iteration<<":"<<get_time(start_t, end_t)<<endl;
#endif
    }
    result[offset++] = merge_paths_for_edges(graph, tensor_prev);
    // Tensor* result = new Tensor(tensor_prev);
    // history_result->push_back(result);
    delete tensor_prev;
    delete tensor_cur;
    // return history_result;
}

void Path_counter::count_for_vertices(Graph& graph, Tensor**& result, int& result_size){
#ifdef ENABLE_TIME_INFO
    struct timeval start_t, end_t;
#endif
    if(enable_residual){
        result_size = original_features[0].size();
    }else{
        result_size = 1;
    }
    result = new Tensor* [result_size];
    int offset = 0;
    uint32_t vertex_count = graph.adj.size();
    Tensor* tensor_prev= new Tensor(vertex_count, original_features.size(), 1); // v<-n
    Tensor* tensor_cur = new Tensor(vertex_count, original_features.size(), 0);
    int total_iterations = original_features[0].size();
    for(int iteration=0; iteration<total_iterations; ++iteration){
#ifdef ENABLE_TIME_INFO
        cout<<"iteration:"<<iteration<<endl;
        gettimeofday(&start_t, NULL);
#endif
        // initialize the mask
        unordered_map<Label, vector<Value>>& mask_map = mask_maps[iteration];
        // iterate the vertices in order
        Value** embeddings_prev = tensor_prev->content;
        for(Vertex v=0; v<vertex_count; ++v){
            Value* embedding_cur = tensor_cur->content[v];
            for(auto n : graph.adj[v]){
                auto itf = mask_map.find(graph.label_map[n]);
                if(itf == mask_map.end()){
                    continue;
                }
                vector<Value>& mask = itf->second;
                vector_add_mul(embedding_cur, embeddings_prev[n], &(mask[0]), mask.size());
            }
        }
        // swap the embedding
        tensor_prev->clear_content();
        if(enable_residual == true && iteration < total_iterations-1){
            result[offset++] = new Tensor(tensor_cur);
        }
        swap(tensor_prev, tensor_cur);
#ifdef ENABLE_TIME_INFO
        gettimeofday(&end_t, NULL);
        cout<<"iteration:"<<iteration<<":"<<get_time(start_t, end_t)<<endl;
#endif
    }
    result[offset++] = new Tensor(tensor_prev);
    delete tensor_prev;
    delete tensor_cur;
}