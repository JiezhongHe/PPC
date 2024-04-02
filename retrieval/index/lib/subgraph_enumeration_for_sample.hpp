#pragma once
#include "auxiliary.hpp"
#include "feature_counter.hpp"
#include "cycle_counting.hpp"
#include "path_counting.hpp"
#include "embedding.hpp"
#include "graph.hpp"

#include <cstdlib>
#include <ctime>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>

timer_t id;
bool stop = false;
void set_stop_flag(union sigval val){
    *((bool*)val.sival_ptr) = true;
    timer_delete(id);
}

unordered_map<Vertex, pair<int, float>> vertex_samples;
unordered_map<Vertex, pair<int, float>> edge_samples;

void register_timer(bool* stop_flag){
    struct timespec spec;
    struct sigevent ent;
    struct itimerspec value;
    struct itimerspec get_val;

    /* Init */
    memset(&ent, 0x00, sizeof(struct sigevent));
    memset(&get_val, 0x00, sizeof(struct itimerspec));

    int test_val = 0;
    /* create a timer */
    ent.sigev_notify = SIGEV_THREAD;
    ent.sigev_notify_function = set_stop_flag;
    ent.sigev_value.sival_ptr = stop_flag;
    timer_create(CLOCK_MONOTONIC, &ent, &id);

    /* start a timer */
    value.it_value.tv_sec = TIME_LIMIT;
    value.it_value.tv_nsec = 0;
    value.it_interval.tv_sec = 0;
    value.it_interval.tv_nsec = 0;
    timer_settime(id, 0, &value, NULL);
}

void subgraph_enumeration_sample(Graph* data_graph, Graph* query_graph, 
    long count_limit, long& result_count, 
    double& enumeration_time, double& preprocessing_time, 
    Tensor* query_vertex_emb, Tensor* data_vertex_emb,
    Tensor* query_edge_emb, Tensor* data_edge_emb,
    bool enable_order, // whether order the candidates,
    string sample_file
){
    stop = false;
    register_timer(&stop);

    struct timeval start_t, end_t;

    gettimeofday(&start_t, NULL);
    
    candidate_auxiliary aux(data_graph, query_graph, query_vertex_emb, data_vertex_emb, query_edge_emb, data_edge_emb, enable_order);

    gettimeofday(&end_t, NULL);
    preprocessing_time = get_time(start_t, end_t);

    vector<Vertex> order;
    vector<vector<Vertex>> pre;
    aux.generate_search_order(order, pre);

    long emb_count = 0;
    
    // start enumeration
    vector<vector<Vertex>> candidates;
    vector<vector<Vertex>> candidates_offset_map;
    int query_vertex_count = query_graph->label_map.size();
    candidates.resize(query_vertex_count);
    candidates_offset_map.resize(query_vertex_count);
    
    for(int i=0;i<aux.candidates[order[0]].candidate.size(); ++i){
        candidates[0].push_back(aux.candidates[order[0]].candidate[i]);
        candidates_offset_map[0].push_back(i);
    }
    
    bool enable_edge_order = enable_order && query_edge_emb != NULL;
    Vertex* candidates_offset = new Vertex [query_vertex_count];
    Vertex* embedding = new Vertex [query_vertex_count];
    Vertex* embedding_offset = new Vertex [query_vertex_count];
    vector<float> candidate_score;
    bool* visited_vertices = new bool [data_graph->label_map.size()];
    memset(visited_vertices, 0, sizeof(bool)*data_graph->label_map.size());
    int cur_depth = 0;
    candidates_offset[cur_depth] = 0;
    vector<unordered_set<Vertex>>& adj_map = data_graph->adj;
    // vector<vector<pair<Vertex, auxiliary_entry*>>>& aux_content = aux.content;
    tree_node* aux_content = aux.candidates;
    gettimeofday(&start_t, NULL);
    while(true){
        while(candidates_offset[cur_depth]<candidates[cur_depth].size()){
            // ++debug;
            if(stop==true){
                goto EXIT;
            }

            Vertex offset = candidates_offset[cur_depth];
            Vertex u = order[cur_depth];
            Vertex v = candidates[cur_depth][offset];
            candidates_offset[cur_depth] ++;

            if(visited_vertices[v] == true){
                continue;
            }
            // cout<<cur_depth<<":"<<u<<":"<<v<<":"<<endl;
            embedding[u] = v;
            embedding_offset[u] = candidates_offset_map[cur_depth][offset];
            visited_vertices[v] = true;
            if(cur_depth == query_vertex_count-1){
                emb_count ++;
                // keep as the vertex samples
                for(int u=0;u<query_vertex_count;++u){
                    auto itf = vertex_samples.find(embedding[u]);
                    int data_vertex_offset = -1;
                    for(int y=0;y<aux.candidates[u].candidate.size();++y){
                        if(aux.candidates[u].candidate[y] == embedding[u]){
                            data_vertex_offset = y;
                            break;
                        }
                    }
                    if(data_vertex_offset == -1){
                        cout<<"error in finding the data vertex from the candidates set"<<endl;
                    }
                    float score = aux.calc_score(query_vertex_emb->content[u], data_vertex_emb->content[embedding[u]], data_vertex_emb->column_size);
                    // if(score > 1){
                    //     cout<<u<<" "<<embedding[u]<<" "<<score<<endl;
                    // }
                    if(itf == vertex_samples.end()){
                        vertex_samples.insert({embedding[u], {1, score}});
                    }else{
                        itf->second.first ++;
                        itf->second.second += score;
                    }
                }
                
                // keep as the edge samples
                for(int u=1;u<query_vertex_count;++u){
                    for(auto u_n : pre[u]){
                        Vertex q_e_id = u<u_n ? query_graph->edge_id_map[u][u_n] : query_graph->edge_id_map[u_n][u];
                        Vertex v = embedding[u];
                        Vertex v_n = embedding[u_n];
                        Vertex d_e_id = v<v_n ? data_graph->edge_id_map[v][v_n] : data_graph->edge_id_map[v_n][v];
                        float score = aux.calc_score(query_edge_emb->content[q_e_id], data_edge_emb->content[d_e_id], data_edge_emb->column_size);
                        auto itf = edge_samples.find(d_e_id);
                        if(itf == edge_samples.end()){
                            edge_samples.insert({d_e_id, {1, score}});
                        }else{
                            itf->second.first ++;
                            itf->second.second += score;
                        }
                    }
                }

                visited_vertices[v] = false;
                if(emb_count >= count_limit){
                    goto EXIT;
                }
            }else{
                cur_depth ++;
                vector<pair<Vertex*, int>> can;
                Vertex max_start_id = 0;
                for(auto parent : pre[order[cur_depth]]){
                    vector<Vertex>& tmp = aux_content[parent].neighbor_candidates[order[cur_depth]][embedding_offset[parent]].first;
                    can.push_back({&(tmp[0]), tmp.size()});
                    // if(max_start_id < tmp[0]){
                    //     max_start_id < tmp[0];
                    // }
                }

                // intersect the candiate neighbors
                if(enable_edge_order){
                    Vertex last_parent = *(pre[order[cur_depth]].rbegin());
                    vector<float>& score = aux_content[last_parent].neighbor_candidates[order[cur_depth]][embedding_offset[last_parent]].second;
                    int parent_num = can.size();
                    candidates[cur_depth].clear();
                    candidates_offset_map[cur_depth].clear();
                    vector<int> ptr(parent_num, 0);
                    candidate_score.clear();
                    int candidate_num;
                    while(true){
                        RESCAN1:
                        for(int j=0;j<parent_num;++j){
                            int cur_ptr = ptr[j];
                            Vertex* vec = can[j].first;
                            int vec_size = can[j].second;
                            while(vec[cur_ptr]<max_start_id && cur_ptr<vec_size) ++cur_ptr;
                            ptr[j] = cur_ptr;
                            if(cur_ptr >= vec_size){
                               goto OUT1;
                            }else if(vec[cur_ptr]>max_start_id){
                                max_start_id = vec[cur_ptr];
                                goto RESCAN1;
                            }else{
                                ptr[j]++;
                            }
                        }
                        candidates_offset_map[cur_depth].push_back(max_start_id);
                        candidate_score.push_back(score[ptr[parent_num-1]-1]);
                        candidates[cur_depth].push_back(aux_content[order[cur_depth]].candidate[max_start_id]);
                        if(ptr[0]<can[0].second){
                            max_start_id = can[0].first[ptr[0]];
                        }else{
                            break;
                        }
                    }
                    // reorder the candiates
                    if(cur_depth != query_vertex_count-1){
                        candidate_num = candidates_offset_map[cur_depth].size();
                        for(int i=0;i<candidate_num-1;++i){
                            for(int j=i+1;j<candidate_num;++j){
                                if(candidate_score[i]<candidate_score[j]){
                                    swap(candidate_score[i],candidate_score[j]);
                                    swap(candidates_offset_map[cur_depth][i], candidates_offset_map[cur_depth][j]);
                                    swap(candidates[cur_depth][i], candidates[cur_depth][j]);
                                }
                            }
                        }
                    }
                    
                    OUT1:
                    candidates_offset[cur_depth] = 0;
                }else{
                    // enable order the candiates by edges
                    candidates[cur_depth].clear();
                    candidates_offset_map[cur_depth].clear();
                    vector<int> ptr(can.size(), 0);
                    while(true){
                        RESCAN:
                        for(int j=0;j<can.size();++j){
                            int cur_ptr = ptr[j];
                            Vertex* vec = can[j].first;
                            int vec_size = can[j].second;
                            while(vec[cur_ptr]<max_start_id && cur_ptr<vec_size) ++cur_ptr;
                            ptr[j] = cur_ptr;
                            if(cur_ptr >= vec_size){
                               goto OUT;
                            }else if(vec[cur_ptr]>max_start_id){
                                max_start_id = vec[cur_ptr];
                                goto RESCAN;
                            }else{
                                ptr[j]++;
                            }
                        }
                        candidates_offset_map[cur_depth].push_back(max_start_id);
                        // candidates[cur_depth].push_back(aux_content[order[cur_depth]][max_start_id].first);
                        candidates[cur_depth].push_back(aux_content[order[cur_depth]].candidate[max_start_id]);
                        if(ptr[0]<can[0].second){
                            max_start_id = can[0].first[ptr[0]];
                        }else{
                            break;
                        }
                    }
                    OUT:
                    candidates_offset[cur_depth] = 0;
                }
            }
        }
        cur_depth --;
        // cout<<"out:"<<cur_depth<<":"<<candidates_offset[cur_depth]<<":"<<candidates[cur_depth].size()<<endl;
        if(cur_depth<0){
            break;
        }else{
            visited_vertices[embedding[order[cur_depth]]] = false;
        }
    }
EXIT:
    delete [] candidates_offset;
    delete [] embedding;
    delete [] embedding_offset;
    delete [] visited_vertices;
    gettimeofday(&end_t, NULL);
    if(stop == false){
        timer_delete(id);
    }
    // write the samples
    ofstream fout_vertex(sample_file+string("_vertex"));
    ofstream fout_edge(sample_file+string("_edge"));
    for(auto p : vertex_samples){
        fout_vertex<<p.first<<" "<<p.second.first<<" "<<p.second.second<<endl;
    }
    for(auto p : edge_samples){
        fout_edge<<p.first<<" "<<p.second.first<<" "<<p.second.second<<endl;
    }
    fout_vertex.close();
    fout_edge.close();
    result_count = emb_count;
    enumeration_time = get_time(start_t, end_t);
}

