#pragma once
#include "auxiliary.hpp"
#include "feature_counter.hpp"
#include "cycle_counting.hpp"
#include "path_counting.hpp"
#include "embedding.hpp"
#include "graph.hpp"
#include "index.hpp"

pair<uint32_t*, int> sum_tensor_safe_without_overflow(Tensor* tensor){
    int column_size = tensor->column_size;
    uint32_t* result = new uint32_t[column_size];
    memset(result, 0, sizeof(uint32_t)*column_size);
    for(int i=0;i<tensor->row_size;++i){
        for(int j=0;j<column_size;++j){
            result[j] += tensor->content[i][j];
        }
    }
    return {result, column_size};
}

bool validate_tensor_safe_without_overflow(uint32_t* t1, uint32_t* t2, int size){
    for(int i=0;i<size;++i){
        if(t1[i]>t2[i]){
            return false;
        }
    }
    return true;
}

vector<Tensor*> load_multi_index(vector<string>& file_list, vector<Graph>& graphs){
    vector<Tensor*> results;
    vector<vector<Tensor*>> embs; // first index indicates files, the second is the i-th graph
    vector<int> sizes;
    int column_size = 0;
    for(auto file : file_list){
        Index_manager manager(file);
        vector<Tensor*> r = manager.load_all_graphs();
        embs.push_back(r);
        column_size += r[0]->column_size;
        sizes.push_back(r[0]->column_size);
    }

    for(int i=0;i<graphs.size();++i){
        int row_size = embs[0][i]->row_size;
        Tensor* result = new Tensor(row_size, column_size);
        Value** r_content = result->content;
        int shift = 0;
        for(int j=0;j<embs.size();++j){
            Value** t_content = embs[j][i]->content;
            int s = sizes[j];
            for(int z=0;z<row_size;++z){
                memcpy(r_content[z]+shift, t_content[z], s*sizeof(Value));
            }
            shift += s;
        }
        results.push_back(result);
        for(int j=0;j<embs.size();++j){
            delete embs[j][i];
        }
    }
    return results;
}

// only support maximally two gnns with two different types
void sub_containment_vc(vector<Graph>& data_graphs, vector<Graph>& query_graphs, vector<int>& filtered_results, vector<double>& filtering_time){
    filtered_results.clear();
    filtering_time.clear();
    struct timeval start_t, end_t;

    for(int i=0; i<query_graphs.size(); ++i){
        int remain_graphs = 0;
        double time = 0;
        gettimeofday(&start_t, NULL);
        for(int j=0;j<data_graphs.size();++j){
            bool filterd = false;
            if(data_graphs[j].get_edge_count() < query_graphs[i].get_edge_count() || data_graphs[j].adj.size() < query_graphs[i].adj.size()){
                continue;
            }
            candidate_auxiliary aux(&(data_graphs[j]), &(query_graphs[i]), NULL, NULL, NULL, NULL, false);
            for(Vertex u=0;u<query_graphs[i].label_map.size();++u){
                if(aux.candidates[u].candidate.empty()){
                    filterd = true;
                    break;
                }
            }
            if(filterd == false){
                remain_graphs ++;
            }
        }
        gettimeofday(&end_t, NULL);
        filtered_results.push_back(remain_graphs);
        filtering_time.push_back(get_time(start_t, end_t));
    }
}

void sub_containment_vc_index(vector<Graph>& data_graphs, vector<Graph>& query_graphs, 
    vector<string>& index_edge_list, vector<string>& index_vertex_list, 
    vector<feature_counter*>& counter_edge_list, vector<feature_counter*>& counter_vertex_list,
    vector<int>& filtered_results, vector<double>& filtering_time
){
    filtered_results.clear();
    filtering_time.clear();
    cout<<"start loading index"<<endl;

    vector<Tensor*> data_edge_embs = load_multi_index(index_edge_list, data_graphs);
    vector<Tensor*> data_vertex_embs = load_multi_index(index_vertex_list, data_graphs);

    struct timeval start_t, end_t;

    for(int i=0; i<query_graphs.size(); ++i){
        int remain_graphs = 0;
        double time = 0;
        gettimeofday(&start_t, NULL);
        Tensor* query_edge_emb = get_frequency_from_multi_counters(query_graphs[i], counter_edge_list, 1);
        Tensor* query_vertex_emb = get_frequency_from_multi_counters(query_graphs[i], counter_vertex_list, 0);
        for(int j=0;j<data_graphs.size();++j){
            bool filterd = false;
            if(data_graphs[j].get_edge_count() < query_graphs[i].get_edge_count() || data_graphs[j].adj.size() < query_graphs[i].adj.size()){
                continue;
            }
            candidate_auxiliary aux(&(data_graphs[j]), &(query_graphs[i]), query_vertex_emb, data_vertex_embs[j], query_edge_emb, data_edge_embs[j], false);

            for(Vertex u=0;u<query_graphs[i].label_map.size();++u){
                if(aux.candidates[u].candidate.empty()){
                    filterd = true;
                    break;
                }
            }
            if(filterd == false){
                remain_graphs ++;
            }
        }
        delete query_edge_emb;
        delete query_vertex_emb;
        gettimeofday(&end_t, NULL);
        cout<<"query:"<<i<<":"<<get_time(start_t, end_t)<<":"<<remain_graphs<<endl;
        filtered_results.push_back(remain_graphs);
        filtering_time.push_back(get_time(start_t, end_t));
    }
    for(auto t:data_edge_embs){
        delete t;
    }
    for(auto t:data_vertex_embs){
        delete t;
    }
}

void sub_containment_graph_level(vector<Graph>& data_graphs, vector<Graph>& query_graphs, 
    vector<string>& index_edge_list, vector<string>& index_vertex_list, 
    vector<feature_counter*>& counter_edge_list, vector<feature_counter*>& counter_vertex_list,
    vector<int>& filtered_results, vector<double>& filtering_time
){
    filtered_results.clear();
    filtering_time.clear();
    struct timeval start_t, end_t;

    vector<pair<uint32_t*, int>> data_edge_embs_sum, data_vertex_embs_sum;
    vector<Tensor*> data_edge_embs = load_multi_index(index_edge_list, data_graphs);
    vector<Tensor*> data_vertex_embs = load_multi_index(index_vertex_list, data_graphs);
    for(int i=0;i<data_graphs.size();++i){
        data_edge_embs_sum.push_back(sum_tensor_safe_without_overflow(data_edge_embs[i]));
        delete data_edge_embs[i];
        data_vertex_embs_sum.push_back(sum_tensor_safe_without_overflow(data_vertex_embs[i]));
        delete data_vertex_embs[i];
    }
    int column_edge_size = data_edge_embs_sum[0].second;
    int column_vertex_size = data_vertex_embs_sum[0].second;

    for(int i=0; i<query_graphs.size(); ++i){
        int remain_graphs = 0;
        double time = 0;
        gettimeofday(&start_t, NULL);
        Tensor* query_edge_emb = get_frequency_from_multi_counters(query_graphs[i], counter_edge_list, 1);
        Tensor* query_vertex_emb = get_frequency_from_multi_counters(query_graphs[i], counter_vertex_list, 0);
        pair<uint32_t*, int>query_edge_emb_sum = sum_tensor_safe_without_overflow(query_edge_emb);
        pair<uint32_t*, int> query_vertex_emb_sum = sum_tensor_safe_without_overflow(query_vertex_emb);
        for(int j=0;j<data_graphs.size();++j){
            bool filterd = !(validate_tensor_safe_without_overflow(query_edge_emb_sum.first, data_edge_embs_sum[j].first, column_edge_size)) || !(validate_tensor_safe_without_overflow(query_vertex_emb_sum.first, data_vertex_embs_sum[j].first, column_vertex_size));
            
            if(filterd == false){
                remain_graphs ++;
            }
        }

        delete query_edge_emb, query_vertex_emb, query_edge_emb_sum, query_vertex_emb_sum;
        gettimeofday(&end_t, NULL);
        cout<<"query:"<<i<<":"<<get_time(start_t, end_t)<<":"<<remain_graphs<<endl;
        filtered_results.push_back(remain_graphs);
        filtering_time.push_back(get_time(start_t, end_t));
    }
    for(auto t:data_edge_embs_sum){
        delete t.first;
    }
    for(auto t:data_vertex_embs_sum){
        delete t.first;
    }
}
