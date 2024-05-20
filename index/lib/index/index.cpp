#include "index.h"

void print_features(vector<vector<Label>> features){
    cout<<"-------"<<endl;
    for(int i=0;i<features.size();++i){
        cout<<i<<":{";
        for(int j=0;j<features[i].size();++j){
            cout<<features[i][j]<<",";
        }
        cout<<"}"<<endl;
    }
}

void* thread_count(void* arg){
    tuple<feature_counter*, Graph*, int, Tensor**&, int&>& input = *((tuple<feature_counter*, Graph*, int, Tensor**&, int&>*)arg);
    feature_counter* counter = std::get<0>(input);
    Graph& graph = *(std::get<1>(input));
    int level = std::get<2>(input);
    Tensor**& result = std::get<3>(input);
    int& result_size = std::get<4>(input);
    
    if(level == 1){
        counter->count_for_edges(graph, result, result_size);
    }else{
        counter->count_for_vertices(graph, result, result_size);
    }
    return NULL;
}

Tensor* get_frequency_from_multi_counters(Graph& graph, vector<feature_counter*>& counter_list, int level, int thread_num){
    vector<Tensor*> vec;
    for(auto p : counter_list){
        Index_constructer con(p);
        vec.push_back(con.count_features(graph, thread_num, level));
    }
    Tensor* result = merge_multi_Tensors(vec);
    for(auto t : vec){
        delete t;
    }
    return result;
}

//////////////////////////////////
void* compute_common_neighbor_func(void* args){
    input_para& input = *(input_para*)args;
    Graph* graph = std::get<0>(input);
    pair<Vertex, Vertex>* task = std::get<1>(input);
    Vertex** common_neighbors = std::get<2>(input);
    int id = std::get<3>(input);
    
    vector<Vertex>& edge_set = graph->edge_set;
    vector<unordered_set<Vertex>>& adj = graph->adj;
    vector<unordered_map<Vertex, Vertex>>& edge_id_map = graph->edge_id_map;

    int test_id = 0;
    for(Vertex e_id=task->first; e_id<task->second; ++e_id){
        test_id++;
        Vertex v = graph->edge_set[e_id*2];
        Vertex n = graph->edge_set[e_id*2+1];
        int common_size = 0;
        int offset = 1;
        Vertex* vec = common_neighbors[e_id];
        for(auto v_n : graph->adj[n]){ // auto v_n : adj[n]
            if(v_n!=v && graph->adj[v].find(v_n)!=graph->adj[v].end()){
                vec[offset++] = v_n;
                common_size ++;
                if(v_n<v){
                    // if(graph->edge_id_map[v_n].find(v)==graph->edge_id_map[v_n].end()){
                    //     cout<<"t:"<<id<<"Error1:"<<v_n<<" is not connected to "<<v<<":"<<graph<<":"<<&(graph->edge_id_map)<<":"<<&(graph->adj)<<endl;
                    //     continue;
                    // }
                    vec[offset++] = graph->edge_id_map[v_n][v];
                }else{
                    // if(graph->edge_id_map[v].find(v_n)==graph->edge_id_map[v].end()){
                    //     cout<<"t:"<<id<<"Error2:"<<v<<" is not connected to "<<v_n<<":"<<graph<<":"<<&(graph->edge_id_map)<<":"<<&(graph->adj)<<endl;
                    //     continue;
                    // }
                    vec[offset++] = graph->edge_id_map[v][v_n];
                }
                if(v_n<n){
                    // if(graph->edge_id_map[v_n].find(n)==graph->edge_id_map[v_n].end()){
                    //     cout<<"t:"<<id<<"Error3:"<<v_n<<" is not connected to "<<n<<":"<<graph<<":"<<&(graph->edge_id_map)<<":"<<&(graph->adj)<<endl;
                    //     continue;
                    // }
                    vec[offset++] = graph->edge_id_map[v_n][n];
                }else{
                    // if(graph->edge_id_map[n].find(v_n)==graph->edge_id_map[n].end()){
                    //     cout<<"t:"<<id<<"Error4:"<<n<<" is not connected to "<<v_n<<":"<<graph<<":"<<&(graph->edge_id_map)<<":"<<&(graph->adj)<<endl;
                    //     continue;
                    // }
                    vec[offset++] = graph->edge_id_map[n][v_n];
                }
            }
            // if(common_size != graph->estimated_common_neighbor_count[e_id]){
            //     exit(0);
            //     cout<<"e_id:"<<e_id<<":"<<common_size<<":"<<graph->estimated_common_neighbor_count[e_id]<<endl;
            // }
            vec[0] = common_size;
        }
    }
    return NULL;
}

// compute the common edge neighbors for graph
void common_edge_neighbor_multi_threads(Graph* graph, int thread_num){
    graph->set_up_edge_id_map();
    if(graph->common_edge_neighbor != NULL){
        return;
    }
    // initialize common_edge_neighbor
    int edge_count = graph->get_edge_count();
    graph->common_edge_neighbor = new Vertex* [edge_count];
    for(Vertex e_id=0; e_id<edge_count; ++e_id){
        graph->common_edge_neighbor[e_id] = new Vertex [3*graph->estimated_common_neighbor_count[e_id]+1];
    }
    
    thread_num = (thread_num<edge_count) ? thread_num : edge_count;
    int average_edges_per_thread = edge_count/thread_num;
    int remain_edges = edge_count%thread_num;
    // initialise the task
    vector<pair<Vertex, Vertex>> tasks;
    int offset = 0;
    int acc_edges = 0;
    for(int i=0; i<thread_num; ++i){
        if(i<remain_edges){
            tasks.push_back({offset, offset+average_edges_per_thread+1});
            offset += average_edges_per_thread+1;
        }else{
            tasks.push_back({offset, offset+average_edges_per_thread});
            offset += average_edges_per_thread;
        }
    }
    assert(offset == edge_count);

    // initialize the parameters
    vector<input_para> input_parameters;
    vector<int> fd_list;
    for(int i=0;i<thread_num;++i){
        input_parameters.push_back(input_para(graph, &(tasks[i]), graph->common_edge_neighbor, i));
    }
    // start computation
    pthread_t* threads = new pthread_t [thread_num];

    for(int i=0;i<thread_num;++i){
        int res = pthread_create(&(threads[i]), NULL, compute_common_neighbor_func, (void*)&input_parameters[i]);
        if(res != 0){
            cout<<"Create thread:"<<i<<" failed for common neighbor computation"<<endl;
            exit(res);
        }
    }

    for(int i=0; i<thread_num; ++i){
        void* ret;
        pthread_join(threads[i], &ret);
    }
    delete [] threads;
}

//////////////////////////////////////
Index_constructer::Index_constructer(){}

Index_constructer::Index_constructer(feature_counter* counter_){
    counter = counter_;
}

void Index_constructer::construct_index_single_batch(Graph& graph, string index_file_name, int thread_num, int level){
    build_time = 0;
    Tensor* result = count_features(graph, thread_num, level);
    Index_manager manager(index_file_name);
    
    struct timeval start_t, end_t;
    gettimeofday(&start_t, NULL);
    
    manager.dump_tensor(result);
    
    gettimeofday(&end_t, NULL);
    
    delete result;
}

// note that residual mechanism is not supported
void Index_constructer::construct_index_in_batch(Graph& graph, string index_file_name, int max_feature_size_per_batch, int thread_num, int level){
    build_time = 0;
    bool enable_residual = counter->get_residual();
    int split_num = 0;
    int feature_offset = 0;
    vector<vector<Label>> label_path_origin = counter->get_features();
    vector<vector<string>> index_splits;
    if(enable_residual){
        index_splits.resize(label_path_origin[0].size());
    }else{
        index_splits.resize(1);
    }
    // analyse redundant features
    vector<vector<bool>> redundant_mask;
    if(enable_residual == true){
        analyse_redundant_features(label_path_origin, redundant_mask);
    }
    string type_name;
    if(counter->get_feature_type() == 1){
        type_name = string("cycle");
    }else{
        type_name = string("path");
    }
    
    while(feature_offset+max_feature_size_per_batch <= label_path_origin.size()){
        // split the features
        cout<<"building progress inner:split_num:"<<split_num<<":("<<label_path_origin.size()/max_feature_size_per_batch+1<<")"<<endl;
        vector<vector<Label>> label_paths_tmp;
        label_paths_tmp.assign(label_path_origin.begin()+feature_offset, label_path_origin.begin()+feature_offset+max_feature_size_per_batch);
        feature_counter* split_counter;
        if(counter->get_feature_type() == 0){
            split_counter = new Path_counter(counter->get_residual(), label_paths_tmp);
        }else{
            split_counter = new Cycle_counter(counter->get_residual(), label_paths_tmp);
        }
        Index_constructer constructor(split_counter);
        vector<Tensor*> result = constructor.count_with_multi_thread(graph, thread_num, level);
        if(enable_residual){
            for(int i=0;i<result.size();++i){
                string index_file_name_split = index_file_name+string("_")+to_string(split_num)+string("_")+to_string(i)+string("_")+type_name;
                index_splits[i].push_back(index_file_name_split);
                vector<bool> mask;
                mask.assign(redundant_mask[i].begin()+feature_offset, redundant_mask[i].begin()+feature_offset+max_feature_size_per_batch);
                remove(index_file_name_split.c_str());
                dump_index_with_mask(result[i], index_file_name_split, mask);
            }
        }else{
            string index_file_name_split = index_file_name+string("_")+to_string(split_num)+string("_")+to_string(0)+string("_")+type_name;
            index_splits[0].push_back(index_file_name_split);
            Index_manager manager(index_file_name_split);
            remove(index_file_name_split.c_str());
            manager.dump_tensor(result[0]);
        }
        for(auto r : result){
            delete r;
        }
        delete split_counter;
        ++ split_num;
        feature_offset += max_feature_size_per_batch;
    }
    if(feature_offset < label_path_origin.size()){
#ifdef PRINT_BUILD_PROGRESS
        cout<<"building progress final:split_num:"<<split_num<<":("<<label_path_origin.size()/max_feature_size_per_batch+1<<")"<<endl;
#endif
        vector<vector<Label>> label_paths_tmp;
        label_paths_tmp.assign(label_path_origin.begin()+feature_offset, label_path_origin.end());
        feature_counter* split_counter;
        if(counter->get_feature_type() == 0){
            split_counter = new Path_counter(counter->get_residual(), label_paths_tmp);
        }else{
            split_counter = new Cycle_counter(counter->get_residual(), label_paths_tmp);
        }
        Index_constructer constructor(split_counter);
        vector<Tensor*> result = constructor.count_with_multi_thread(graph, thread_num, level);
        
        if(enable_residual){
            for(int i=0;i<result.size();++i){
                string index_file_name_split = index_file_name+string("_")+to_string(split_num)+string("_")+to_string(i)+string("_")+type_name;
                index_splits[i].push_back(index_file_name_split);
                vector<bool> mask;
                mask.assign(redundant_mask[i].begin()+feature_offset, redundant_mask[i].end());
                remove(index_file_name_split.c_str());
                dump_index_with_mask(result[i], index_file_name_split, mask);
            }
        }else{
            string index_file_name_split = index_file_name+string("_")+to_string(split_num)+string("_")+to_string(0)+string("_")+type_name;
            index_splits[0].push_back(index_file_name_split);
            Index_manager manager(index_file_name_split);
            remove(index_file_name_split.c_str());
            manager.dump_tensor(result[0]);
        }
        for(auto r : result){
            delete r;
        }
        delete split_counter;
    }
    vector<string> filelist;
    for(auto vec : index_splits){
        filelist.insert(filelist.end(), vec.begin(), vec.end());
    }
    merge_multi_index_files_with_bounded_memory(filelist, index_file_name);
    for(auto file : filelist){
        int flag = remove(file.c_str());
        if(flag != 0){
            cout<<"removing temporary index file error:"<<file<<endl;
        }
    }
}

void Index_constructer::analyse_redundant_features(vector<vector<Label>>& features, vector<vector<bool>>& redundant_mask){
    // rearrange the features
    vector<vector<Label>> slot_labels;
    int feature_num = features.size();
    for(int l=features[0].size()-1; l>=0; --l){
        vector<Label> inner;
        for(int j=0; j < feature_num; ++j){
            inner.push_back(features[j][l]);
        }
        slot_labels.push_back(inner);
    }
    int current_state = 0;
    int current_new_state = 0;
    unordered_map<int, unordered_map<Label, int>> state_trans_table;
    vector<int> parent_state(slot_labels[0].size(), 0);
    redundant_mask.resize(slot_labels.size());
    for(int i=0;i<slot_labels.size();++i){
        vector<int> group;
        redundant_mask[i].resize(feature_num);
        for(int j=0; j<slot_labels[0].size(); ++j){
            current_state = parent_state[j];
            if(state_trans_table[current_state].find(slot_labels[i][j]) == state_trans_table[current_state].end()){
                current_new_state++;
                state_trans_table[current_state].insert({slot_labels[i][j], current_new_state});
                redundant_mask[i][j] = true;
            }else{
                redundant_mask[i][j] = false;
            }
            parent_state[j] = state_trans_table[current_state][slot_labels[i][j]];
        }
    }
}

Tensor* Index_constructer::count_features(Graph& graph, int thread_num, int level){
    vector<Tensor*> results = count_with_multi_thread(graph, thread_num, level);
    bool enable_residual = counter->get_residual();
    if(enable_residual){
        vector<vector<bool>> redundant_mask;
        vector<vector<Label>> features = counter->get_features();
        analyse_redundant_features(features, redundant_mask);
        Tensor* result = merge_multi_Tensors_with_mask(results, redundant_mask);
        // Tensor* result = merge_multi_Tensors(results);
        // release results
        for(auto t:results){
            delete t;
        }
        return result;
    }else{
        return results[0];
    }
}

vector<Tensor*> Index_constructer::count_with_multi_thread(Graph& graph, int thread_num, int level){

    struct timeval start_t, end_t;
    gettimeofday(&start_t, NULL);

    vector<Tensor*> final_results;
    if(thread_num <= 1){
        gettimeofday(&start_t, NULL);
        graph.construct_edge_common_neighbor();
        Tensor** tmp_result;
        int tmp_result_size;
        if(level == 1){
            counter->count_for_edges(graph, tmp_result, tmp_result_size);
        }else{
            counter->count_for_vertices(graph, tmp_result, tmp_result_size);
        }
        gettimeofday(&end_t, NULL);
        build_time += get_time(start_t, end_t);

        final_results.resize(tmp_result_size);
        for(int i=0; i<tmp_result_size; ++i){
            final_results[i] = tmp_result[i];
        }
    }else{
// #ifdef ENABLE_TIME_INFO
        
// #endif
        gettimeofday(&start_t, NULL);
        vector<vector<Label>> label_paths_split = counter->get_features();
        common_edge_neighbor_multi_threads(&graph, thread_num);
        gettimeofday(&end_t, NULL);
        build_time += get_time(start_t, end_t);


        // split the paths
        vector<vector<vector<Label>>> label_paths_for_threads;
        thread_num = (thread_num>label_paths_split.size()) ? label_paths_split.size() : thread_num;
        int average_path_count = label_paths_split.size()/thread_num;
        int remain_path_count = label_paths_split.size()%thread_num;
        label_paths_for_threads.resize(thread_num);
        int offset = 0;
        for(int i=0;i<thread_num; ++i){
            if(i<remain_path_count){
                label_paths_for_threads[i].assign(label_paths_split.begin()+offset, label_paths_split.begin()+offset+average_path_count+1);
                offset += average_path_count+1;
            }else{
                label_paths_for_threads[i].assign(label_paths_split.begin()+offset, label_paths_split.begin()+offset+average_path_count);
                offset += average_path_count;
            }
        }
        // start counting
        // preparing the input_parameters
        vector<tuple<feature_counter*, Graph*, int, Tensor**&, int&>> input_parameters; // enable_residual, graph, level;
        vector<Tensor**> results;
        vector<int> results_size;
        vector<feature_counter*> counter_thread;
        results.resize(thread_num);
        results_size.resize(thread_num);
        for(int i=0;i<thread_num; ++i){
            results[i] = NULL;
            results_size[i];
            if(counter->get_feature_type() == 1){    
                // cycle 1
                feature_counter* cycle_counter = new Cycle_counter(counter->get_residual(), label_paths_for_threads[i]);
                input_parameters.push_back(tuple<feature_counter*, Graph*, int, Tensor**&, int&>(cycle_counter, &graph, level, results[i], results_size[i]));
                counter_thread.push_back(cycle_counter);
            }else{
                // path 0
                feature_counter* path_counter = new Path_counter(counter->get_residual(), label_paths_for_threads[i]);
                input_parameters.push_back(tuple<feature_counter*, Graph*, int, Tensor**&, int&>(path_counter, &graph, level, results[i], results_size[i]));
                counter_thread.push_back(path_counter);
            }
        }
        pthread_t* threads = new pthread_t [thread_num];
        // pthread_barrier_init(&barrier, NULL, thread_num + 1);
        gettimeofday(&start_t, NULL);
        for(int i=0; i<thread_num; ++i){
            int res = pthread_create(&(threads[i]), NULL, thread_count, (void*)&(input_parameters[i]));
            if(res != 0){
                cout<<"Created thread:"<<i<<" failed"<<endl;
                exit(res);
            }
        }
        // pthread_barrier_wait(&barrier);
        for(int i=0; i<thread_num; ++i){
            void* ret;
            int res = pthread_join(threads[i], &ret);
        }
        gettimeofday(&end_t, NULL);
        build_time += get_time(start_t, end_t);

        // merge the result and release the results
        for(int i=0;i<results_size[0];++i){
            vector<Tensor*> vec;
            for(auto t : results){
                vec.push_back(t[i]);
            }
            final_results.push_back(merge_multi_Tensors(vec));
        }
        for(auto t:results){
            for(int i=0;i<results_size[0];++i){
                delete t[i];
            }
            delete [] t;
        }

        delete threads;
        for(auto c:counter_thread){
            delete c;
        }
    }
    return final_results;
}