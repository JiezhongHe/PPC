#include "feature_selector.h"

void dump_features(vector<vector<Label>> features, string filename){
    ofstream fout(filename);
    fout<<"[";
    for(int i=0;i<features.size();++i){
        fout<<"[";
        for(int j=0;j<features[i].size();++j){
            fout<<features[i][j];
            if(j < features[i].size()-1){
                fout<<",";
            }
        }
        fout<<"]";
        if(i<features.size()-1){
            fout<<",";
        }
    }
    fout<<"]";
}

///////////////////////////
Feature_selector::Feature_selector(){}

Feature_selector::Feature_selector(int num_labels_, int feature_num_, int feature_length_, bool enable_cycle_, string tmp_path_){
    num_labels = num_labels_;
    feature_num = feature_num_;
    feature_length = feature_length_;
    enable_cycle = enable_cycle_;
    tmp_path = tmp_path_;
}

string Feature_selector::generate_tmp_files(){
    while(true){
        string file = tmp_path+string("/")+to_string(rand());
        ifstream fin(file);
        if(!fin.good()){
            return file;
        }
    }
}

vector<int> Feature_selector::select_by_complementariness(vector<bitset<MAX_SAMPLE_NUM>>& result_bit, int sample_num){
    // select the first candiate feature with highest precision
    float max_precision = 0.0;
    int max_feature_id = 0;
    for(int i=0;i<result_bit.size();++i){
        float precision = result_bit[i].count()/(float)sample_num;
        if(precision > max_precision){
            max_feature_id = i;
            max_precision = precision;
        }
    }
    vector<int> candidate_feature_ids = {max_feature_id};
    unordered_set<int> remaining_ids;
    for(int i=0;i<result_bit.size();++i){
        remaining_ids.insert(i);
    }
    remaining_ids.erase(max_feature_id);
    // iteratively select the candidate ids
    while(candidate_feature_ids.size() < feature_num && remaining_ids.size()>0){
        bitset<MAX_SAMPLE_NUM> ensembled_result;
        for(auto id : candidate_feature_ids){
            ensembled_result |= result_bit[id];
        }
        int max_value = -1;
        int max_candidate = -1;
        for(auto c : remaining_ids){
            bitset<MAX_SAMPLE_NUM> tmp = (~ensembled_result) & result_bit[c];
            int value = tmp.count();
            if(value > max_value){
                max_value = value;
                max_candidate = c;
            }
        }
        if(max_value>0){
            candidate_feature_ids.push_back(max_candidate);
            remaining_ids.erase(max_candidate);
        }else{
            break;
        }
    }
    while(candidate_feature_ids.size() < feature_num){
        candidate_feature_ids.push_back(rand()%result_bit.size());
    }
    return candidate_feature_ids;
}

vector<vector<Label>> Feature_selector::extract_for_singe_graph(vector<Graph>& sample_queries, vector<vector<Vertex>>& sample_query_anchors, vector<vector<Vertex>>& sample_data_anchors, Graph& data_graph, int level, int max_batch_size, int thread_num){
    vector<vector<Label>> candidate_features;
    for(Label i=0;i<num_labels;++i){
        candidate_features.push_back({i});
    }
    data_graph.set_up_edge_id_map();
    for(int length=1;length<=feature_length;++length){
        // build the index for the data graph
        feature_counter* counter;
        if(enable_cycle){
            counter = new Cycle_counter(false, candidate_features);
        }else{
            counter = new Path_counter(false, candidate_features);
        }
        Index_constructer constructor(counter);
        string data_graph_index = generate_tmp_files();
        constructor.construct_index_in_batch(data_graph, data_graph_index, max_batch_size, thread_num, level);
        Index_manager data_manager(data_graph_index);
        if(sample_queries.size() > MAX_SAMPLE_NUM){
            cout<<"sample num overflow, reset MAX_SAMPLE_NUM as "<<sample_queries.size()<<endl;
            exit(0);
        }
        vector<bitset<MAX_SAMPLE_NUM>> result_bit;
        result_bit.resize(candidate_features.size());
        int sample_id = 0;
        for(int j=0;j<sample_queries.size();++j){
            Graph& query_graph = sample_queries[j];
            vector<Vertex>& query_anchors = sample_query_anchors[j];
            vector<Vertex>& data_anchors = sample_data_anchors[j];
            Vertex query_anchor = query_anchors[0];
            Vertex data_anchor = data_anchors[0];
            Vertex query_anchor_u, data_anchor_v;
            if(level == 1){
                query_anchor_u = query_anchors[1];
                data_anchor_v = data_anchors[1];
            }
            Tensor* query_emb = constructor.count_features(query_graph, thread_num, level);
            vector<Value> query_anchor_emb, data_anchor_emb;
            Vertex query_e_id, data_e_id;
            if(level == 1){
                query_e_id = (query_anchor<query_anchor_u) ? query_graph.edge_id_map[query_anchor][query_anchor_u] : query_graph.edge_id_map[query_anchor_u][query_anchor];
                data_e_id = (data_anchor<data_anchor_v) ? data_graph.edge_id_map[data_anchor][data_anchor_v] : data_graph.edge_id_map[data_anchor_v][data_anchor];
                query_emb->extract_row(query_e_id, query_anchor_emb);
                data_manager.load_vertex_embedding(0, data_e_id, data_anchor_emb);
            }else{
                query_emb->extract_row(query_anchor, query_anchor_emb);
                data_manager.load_vertex_embedding(0, data_anchor, data_anchor_emb);
            }
            // calculate the result bits
            for(int z=0;z<query_anchor_emb.size();++z){
                if(query_anchor_emb[z] > data_anchor_emb[z]){
                    result_bit[z].set(sample_id);
                }
            }
            delete query_emb;
            sample_id ++;
        }
        delete counter;
        // select the top-k features
        vector<int> result_ids = select_by_complementariness(result_bit, sample_queries.size());
        // keep the internal features
        vector<vector<Label>> features;
        for(auto id:result_ids){
            features.push_back(candidate_features[id]);
        }
        dump_features(features, data_graph_index+string(".features"));
        // expand the features
        vector<vector<Label>> candidate_features_next;
        for(auto id : result_ids){
            if(candidate_features[id].size() == feature_length){
                candidate_features_next.push_back(candidate_features[id]);
            }else{
                for(Vertex j=0;j<num_labels;++j){
                    vector<Label> tmp_f = candidate_features[id];
                    tmp_f.insert(tmp_f.begin(), j);
                    candidate_features_next.push_back(tmp_f);
                }
            }
        }
        cout<<"progress:"<<length<<":"<<candidate_features.size()<<":"<<candidate_features_next.size()<<":"<<result_ids.size()<<endl;
        swap(candidate_features, candidate_features_next);
        remove(data_graph_index.c_str());
    }
    return candidate_features;
}

vector<vector<Label>> Feature_selector::extract_for_multi_graph(vector<Graph>& sample_queries, vector<vector<Vertex>>& sample_query_anchors, vector<vector<Vertex>>& sample_data_anchors, vector<Vertex>& data_graph_ids, vector<Graph>& data_graphs, int level, int max_batch_size, int thread_num){
    vector<vector<Label>> candidate_features;
    for(Label i=0;i<num_labels;++i){
        candidate_features.push_back({i});
    }
    for(int i=0;i<data_graphs.size();++i){
        data_graphs[i].set_up_edge_id_map();
    }
    for(int length=1;length<=feature_length;++length){
        // build the index for the data graph
        feature_counter* counter;
        if(enable_cycle){
            counter = new Cycle_counter(false, candidate_features);
        }else{
            counter = new Path_counter(false, candidate_features);
        }
        Index_constructer constructor(counter);
        string data_graph_index = generate_tmp_files();
        for(auto data_graph : data_graphs){
            constructor.construct_index_in_batch(data_graph, data_graph_index, max_batch_size, thread_num, level);
        }
        Index_manager data_manager(data_graph_index);
        if(sample_queries.size() > MAX_SAMPLE_NUM){
            cout<<"sample num overflow, reset MAX_SAMPLE_NUM as "<<sample_queries.size()<<endl;
            exit(0);
        }
        vector<bitset<MAX_SAMPLE_NUM>> result_bit;
        result_bit.resize(candidate_features.size());
        int sample_id = 0;
        for(int j=0;j<sample_queries.size();++j){
            Graph& query_graph = sample_queries[j];
            vector<Vertex>& query_anchors = sample_query_anchors[j];
            vector<Vertex>& data_anchors = sample_data_anchors[j];
            Vertex data_graph_id = data_graph_ids[j];

            Vertex query_anchor = query_anchors[0];
            Vertex data_anchor = data_anchors[0];
            Vertex query_anchor_u, data_anchor_v;
            if(level == 1){
                query_anchor_u = query_anchors[1];
                data_anchor_v = data_anchors[1];
            }
            Tensor* query_emb = constructor.count_features(query_graph, thread_num, level);
            vector<Value> query_anchor_emb, data_anchor_emb;
            Vertex query_e_id, data_e_id;
            if(level == 1){
                query_e_id = (query_anchor<query_anchor_u) ? query_graph.edge_id_map[query_anchor][query_anchor_u] : query_graph.edge_id_map[query_anchor_u][query_anchor];
                data_e_id = (data_anchor<data_anchor_v) ? data_graphs[data_graph_id].edge_id_map[data_anchor][data_anchor_v] : data_graphs[data_graph_id].edge_id_map[data_anchor_v][data_anchor];
                query_emb->extract_row(query_e_id, query_anchor_emb);
                data_manager.load_vertex_embedding(0, data_e_id, data_anchor_emb);
            }else{
                query_emb->extract_row(query_anchor, query_anchor_emb);
                data_manager.load_vertex_embedding(0, data_anchor, data_anchor_emb);
            }
            // calculate the result bits
            for(int z=0;z<query_anchor_emb.size();++z){
                if(query_anchor_emb[z] > data_anchor_emb[z]){
                    result_bit[z].set(sample_id);
                }
            }
            delete query_emb;
            sample_id ++;
        }
        delete counter;
        // select the top-k features
        vector<int> result_ids = select_by_complementariness(result_bit, sample_queries.size());
        // keep the internal features
        vector<vector<Label>> features;
        for(auto id:result_ids){
            features.push_back(candidate_features[id]);
        }
        dump_features(features, data_graph_index+string(".features"));
        // expand the features
        vector<vector<Label>> candidate_features_next;
        for(auto id : result_ids){
            if(candidate_features[id].size() == feature_length){
                candidate_features_next.push_back(candidate_features[id]);
            }else{
                for(Vertex j=0;j<num_labels;++j){
                    vector<Label> tmp_f = candidate_features[id];
                    tmp_f.insert(tmp_f.begin(), j);
                    candidate_features_next.push_back(tmp_f);
                }
            }
        }
        cout<<"progress:"<<length<<":"<<candidate_features.size()<<":"<<candidate_features_next.size()<<":"<<result_ids.size()<<endl;
        swap(candidate_features, candidate_features_next);
        remove(data_graph_index.c_str());
    }
    return candidate_features;
}