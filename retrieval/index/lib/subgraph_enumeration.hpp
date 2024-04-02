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

void subgraph_enumeration_get_candidate_info(Graph* data_graph, Graph* query_graph, 
    Tensor* query_vertex_emb, Tensor* data_vertex_emb,
    Tensor* query_edge_emb, Tensor* data_edge_emb,
    uint32_t& vertex_candidate_num, uint32_t& edge_candidate_num
){
    candidate_auxiliary aux(data_graph, query_graph, query_vertex_emb, data_vertex_emb, query_edge_emb, data_edge_emb, false);
    aux.get_candidate_info(vertex_candidate_num, edge_candidate_num);
}

void subgraph_enumeration(Graph* data_graph, Graph* query_graph, 
    long count_limit, long& result_count, 
    double& enumeration_time, double& preprocessing_time, 
    Tensor* query_vertex_emb, Tensor* data_vertex_emb,
    Tensor* query_edge_emb, Tensor* data_edge_emb,
    bool enable_order // whether order the candidates
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
#ifdef FAILING_SET_PRUNING
    vector<bool>& check_point_failing_set = aux.check_point_failing_set;
    vector<int>& parent_check_point = aux.parent_check_point;
#endif

    // cout<<"search order:{";
    // for(auto v : order){
    //     cout<<v<<", ";
    // }
    // cout<<"}"<<endl;
    // cout<<"checkpoint:{";
    // for(auto v : aux.check_point_failing_set){
    //     cout<<v<<", ";
    // }
    // cout<<"}"<<endl;
    // cout<<"parent:{";
    // for(int i=0; i<aux.parent_check_point.size(); ++i){
    //     cout<<i<<":"<<aux.parent_check_point[i]<<", ";
    // }
    // cout<<"}"<<endl;
    // order = {9, 5, 11, 16, 18, 20, 23, 22, 24, 26, 32, 35, 1, 6, 36, 7, 10, 3, 4, 13, 14, 25, 19, 27, 28, 29, 31, 33, 34, 39, 30, 38, 2, 0, 8, 12, 17, 15, 21, 37};
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
    bool* find_matches = new bool [query_vertex_count+1];
    find_matches[query_vertex_count] = true;

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
            find_matches[cur_depth] = false;
            embedding[u] = v;
            embedding_offset[u] = candidates_offset_map[cur_depth][offset];
            visited_vertices[v] = true;
            if(cur_depth == query_vertex_count-1){
                find_matches[cur_depth] = true;
                emb_count ++;
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
            find_matches[cur_depth-1] |= find_matches[cur_depth];
        }
        cur_depth --;
        // cout<<"out:"<<cur_depth<<":"<<candidates_offset[cur_depth]<<":"<<candidates[cur_depth].size()<<endl;
        if(cur_depth<0){
            break;
        }else{
#ifdef FAILING_SET_PRUNING
            if(check_point_failing_set[cur_depth] == true && find_matches[cur_depth] == false){
                // cout<<"goto parent:"<<cur_depth<<"->"<<parent_check_point[order[cur_depth]]<<endl;
                for(int depth=cur_depth; depth>=parent_check_point[order[cur_depth]]; --depth){
                    visited_vertices[embedding[order[depth]]] = false;
                }
                // cout<<"finish goto"<<endl;
                cur_depth = parent_check_point[order[cur_depth]];
            }else{
                visited_vertices[embedding[order[cur_depth]]] = false;
            }
#else
            visited_vertices[embedding[order[cur_depth]]] = false;
#endif
        }
    }
EXIT:
    delete [] candidates_offset;
    delete [] embedding;
    delete [] embedding_offset;
    delete [] visited_vertices;
    delete [] find_matches;
    gettimeofday(&end_t, NULL);
    if(stop == false){
        timer_delete(id);
    }
    result_count = emb_count;
    enumeration_time = get_time(start_t, end_t);
}

