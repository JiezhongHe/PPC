#include "subgraph_enumeration.h"
#include "../utility/computesetintersection.h"
#include <chrono>
#include "signal.h"
#include "time.h"

timer_t id;
void set_stop_flag_enum(union sigval val){
    *((bool*)val.sival_ptr) = true;
    timer_delete(id);
}

void register_timer_enum(bool* stop_flag){
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
    ent.sigev_notify_function = set_stop_flag_enum;
    ent.sigev_value.sival_ptr = stop_flag;
    timer_create(CLOCK_MONOTONIC, &ent, &id);

    /* start a timer */
    value.it_value.tv_sec = TIME_LIMIT;
    value.it_value.tv_nsec = 0;
    value.it_interval.tv_sec = 0;
    value.it_interval.tv_nsec = 0;
    timer_settime(id, 0, &value, NULL);
}

#if PRINT_MEM_INFO == 1
inline int GetCurrentPid(){
    return getpid();
}

void thread_get_mem_info(float& peak_memory, bool& stop){
    peak_memory = 0.0; // MB
    float read_memory;
    int current_pid = GetCurrentPid();
    while (stop==false)
    {
        read_memory = GetMemoryUsage(current_pid);
        peak_memory = (peak_memory>read_memory) ? peak_memory : read_memory;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

inline float GetMemoryUsage(int pid){
    char file_name[64] = { 0 };
    FILE* fd;
    char line_buff[512] = { 0 };
    sprintf(file_name, "/proc/%d/status", pid);

    fd = fopen(file_name, "r");
    if (nullptr == fd)
        return 0;

    char name[64];
    int vmrss = 0;
    for (int i = 0; i < VMRSS_LINE - 1; i++)
        fgets(line_buff, sizeof(line_buff), fd);

    fgets(line_buff, sizeof(line_buff), fd);
    sscanf(line_buff, "%s %d", name, &vmrss);
    fclose(fd);

    // cnvert VmRSS from KB to MB
    return vmrss / 1024.0;
}
#endif



SubgraphEnum::SubgraphEnum(Graph* data_graph){
    data_graph_ = data_graph;

    storage_ = NULL;
    pp_ = NULL;
}

void SubgraphEnum::initialization(){
    // initialize the order_index
    order_index_ = vector<Vertex>(order_.size(), 0);
    for(int offset=0;offset<order_.size();++offset){
        order_index_[order_[offset]] = offset;
    }

    // initialize the ancestors
    uint32_t query_vertex_count = query_graph_->getVerticesCount();
    ancestors_depth_.resize(query_vertex_count+1);
    for(uint32_t i=0; i<query_vertex_count+1; ++i){
        ancestors_depth_[i].reset();
    }
    for(uint32_t i=1; i<=query_vertex_count; ++i){
        ancestors_depth_[i].set(i);

        uint32_t nbrs_count;
        const Vertex* nbrs = query_graph_->getVertexNeighbors(order_[i], nbrs_count);
        for(uint32_t j=0;j<nbrs_count;++j){
            Vertex n = nbrs[j];
            uint32_t depth = order_index_[n];
            if(depth > i){
                ancestors_depth_[depth].set(depth);
                ancestors_depth_[depth] = ancestors_depth_[i] | ancestors_depth_[depth];
            }
        }
    }

    // initialize the successors and predecessors
    successor_neighbors_in_depth_.clear();
    predecessor_neighbors_in_depth_.clear();
    successor_neighbors_in_depth_.resize(query_vertex_count+1);
    predecessor_neighbors_in_depth_.resize(query_vertex_count+1);
    for(uint32_t i=1; i<=query_vertex_count; ++i){
        uint32_t nbrs_count;
        const Vertex* nbrs = query_graph_->getVertexNeighbors(order_[i], nbrs_count);
        for(uint32_t j=0;j<nbrs_count;++j){
            Vertex n = nbrs[j];
            uint32_t depth = order_index_[n];
            if(depth > i){
                successor_neighbors_in_depth_[i].push_back(depth);
            }else{
                predecessor_neighbors_in_depth_[i].push_back(depth);
            }
        }
    }
    for(uint32_t i=1;i<=query_vertex_count;++i){
        sort(successor_neighbors_in_depth_[i].begin(), successor_neighbors_in_depth_[i].end());
    }

    // initialize the marker marks all the unmatched queries having candidates
    full_descendent_.set();
    bitset<MAX_QUERY_SIZE> touched_descendent;
    for(int x=1;x<query_vertex_count+1;x++){
        Vertex u = order_[x];
        for(auto suc_u_depth : successor_neighbors_in_depth_[x]){
            touched_descendent.set(suc_u_depth);
        }
        touched_descendent.set(x);
        bool full_touched = true;
        for(int d=x;d<=query_vertex_count;++d){
            if(touched_descendent.test(d) == false){
                full_touched = false;
                break;
            }
        }
        if(full_touched == true){
            break;
        }else{
            full_descendent_.reset(x);
        }
    }

    // intialize the parent map
    query_vertex_count_ = query_graph_->getVerticesCount();
    parent_failing_set_map_.clear();
    parent_failing_set_map_.resize(query_vertex_count_+1);
    for(int i=0;i<=query_vertex_count_; ++i){
        parent_failing_set_map_[i].resize(query_vertex_count_+1);
    }
    for(int d=1;d<=query_vertex_count_; ++d){
        for(auto suc_depth : successor_neighbors_in_depth_[d]){
            parent_failing_set_map_[d][suc_depth] = ancestors_depth_[d] | ancestors_depth_[suc_depth];
        }
    }
}

void SubgraphEnum::order_adjustment(){
    // initialize the order_index
    order_index_ = vector<Vertex>(order_.size(), 0);
    for(int offset=0;offset<order_.size();++offset){
        order_index_[order_[offset]] = offset;
    }

    // initialize the successors and predecessors
    uint32_t query_vertex_count = query_graph_->getVerticesCount();
    successor_neighbors_in_depth_.clear();
    successor_neighbors_in_depth_.resize(query_vertex_count+1);
    for(uint32_t i=1; i<=query_vertex_count; ++i){
        uint32_t nbrs_count;
        const Vertex* nbrs = query_graph_->getVertexNeighbors(order_[i], nbrs_count);
        for(uint32_t j=0;j<nbrs_count;++j){
            Vertex n = nbrs[j];
            uint32_t depth = order_index_[n];
            if(depth > i){
                successor_neighbors_in_depth_[i].push_back(depth);
            }
        }
    }
}

bool debug(int depth, Vertex* emb, vector<Vertex>& order){
    //560,508855,97672,97662,97668,97652,508904,511522,97650,97648,190091,97655,510309,190092,515644,508857,80602,97664,511287,512715,97649,97653,522241,514799,16479,538317,80600,515790,522225,538839,80597,539943
    vector<Vertex> content = {0, 1446,1466,432,1624,2121,1100,2321,2458,2729,2771,9,133,1389,1146,250,1297,1187,2852,433,422,2537,2929,1620,964,1150,2666,468,1962,923,2581,2028,2607};
    // vector<Vertex> content = {0, 12, 7, 8, 11, 21803, 23953, 22209, 5462, 12};
    if(depth >= content.size()){
        return false;
    }
    for(int i=1;i<depth+1;++i){
        if(emb[i] != content[i]){
            return false;
        }
    }
    return true;
}

bool validate_correctness(Graph* query_graph, Graph* data_graph, vector<Vertex>& order_index, vector<Vertex>& emb_depth){
    for(Vertex u=0;u<query_graph->getVerticesCount();++u){
        uint32_t nbrs_count;
        const Vertex* nbrs = query_graph->getVertexNeighbors(u, nbrs_count);
        for(int i=0;i<nbrs_count;++i){
            Vertex u_n = nbrs[i];
            Vertex v = emb_depth[order_index[u]];
            Vertex v_n = emb_depth[order_index[u_n]];
            if(data_graph->checkEdgeExistence(v,v_n) == false){
                return false;
            }
        }
    }
    return true;
}

bool cmp_score(pair<Vertex, float>& p1, pair<Vertex, float>& p2){
    return p1.second > p2.second;
}

struct CandidateBuffer{
    Vertex* content;
    uint32_t content_size;
    uint32_t buffer_size;
};

void SubgraphEnum::match(Graph* query_graph, string ordering_method, long count_limit, uint32_t time_limit){
    query_graph_ = query_graph;
    // Execute Preprocessor
    query_time_ = 0;

#if PRINT_MEM_INFO == 1
    bool stop_thread = false;
    int current_pid = GetCurrentPid();
    thread mem_info_thread(thread_get_mem_info, ref(peak_memory_), ref(stop_thread));
    float starting_memory_cost = GetMemoryUsage(current_pid);
#endif

    pp_ = new preprocessor();
    storage_ = new catalog(query_graph_, data_graph_);
    pp_->execute(query_graph, data_graph_, storage_, true);
    preprocessing_time_ = NANOSECTOSEC(pp_->preprocess_time_);
    query_time_ += preprocessing_time_;

    // Generate Query Plan
    std::vector<std::vector<uint32_t>> spectrum;
    query_plan_generator::generate_query_plan_with_nd(query_graph, storage_, spectrum);
    ordering_time_ = NANOSECTOSEC(query_plan_generator::ordering_time_);

    // Order Adjustment
    auto start = std::chrono::high_resolution_clock::now();
    order_ = spectrum[0];
    // order_ = {22, 27, 25, 5, 23, 29, 21, 24, 26, 28, 31, 30, 3, 4, 17, 1, 11, 12, 15, 7, 2, 10, 16, 6, 8, 13, 14, 19, 18, 20, 9, 0};
    order_.insert(order_.begin(), 0); // padding
    // cout<<"original_order:";
    // for(auto n : order_){
    //     cout<<n<<", ";
    // }
    cout<<endl;
    // order_adjustment();
    initialization();
    // for(auto n : order_){
    //     cout<<n<<", ";
    // }
    // cout<<endl;
    // return;
    auto end = std::chrono::high_resolution_clock::now();
    order_adjust_time_ = NANOSECTOSEC(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    query_time_ += order_adjust_time_;

    // encoding the relations
    start = std::chrono::high_resolution_clock::now();
    // TrieEncoder* encoder = new TrieEncoder(storage_, order_, order_index_, query_graph_, data_graph_, successor_neighbors_in_depth_);
    TrieEncoder* encoder = new TrieEncoder(storage_, order_, order_index_, query_graph_, data_graph_);
    end = std::chrono::high_resolution_clock::now();
    query_time_ += NANOSECTOSEC(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());


#if PRINT_LEAF_STATE == 1
    leaf_states_counter_ = vector<uint64_t>(20, 0);
#endif

    // start enumeration
    // start timer
    stop_ = false;
    register_timer_enum(&stop_);
    start = std::chrono::high_resolution_clock::now();

    vector<CandidateBuffer> extending_candidates;
    vector<CandidateBuffer> extending_candidates_tmp;
    extending_candidates.resize(query_vertex_count_+1);
    extending_candidates_tmp.resize(query_vertex_count_+1);
    uint32_t max_degree = data_graph_->getGraphMaxDegree();
    for(int i=0;i<query_vertex_count_+1;++i){
        extending_candidates[i].content = new Vertex [max_degree];
        extending_candidates[i].buffer_size = max_degree;
        extending_candidates[i].content_size = 0;
        extending_candidates_tmp[i].content = new Vertex [max_degree];
        extending_candidates_tmp[i].buffer_size = max_degree;
        extending_candidates_tmp[i].content_size = 0;
    }

    query_vertex_count_ = query_graph_->getVerticesCount();
    data_vertex_count_ = data_graph_->getVerticesCount();


    searching_candidates_.clear();
    searching_candidates_.resize(query_vertex_count_+1);
    // initializing the candidates
    // encoder->get_candidates(1, searching_candidates_[1]);

    vector<Vertex> init_candidates;
    encoder->get_candidates(1, init_candidates);
    // realloc the initial candidate
    delete extending_candidates[1].content;
    delete extending_candidates_tmp[1].content;
    extending_candidates[1].content = new Vertex [init_candidates.size()+1];
    extending_candidates_tmp[1].content = new Vertex [init_candidates.size()+1];

    extending_candidates[1].content_size = init_candidates.size();
    memcpy(extending_candidates[1].content, &(init_candidates[0]), sizeof(Vertex)*init_candidates.size());


    embedding_depth_.clear();
    embedding_depth_.resize(query_vertex_count_+1);

    visited_query_depth_ = new Vertex [data_vertex_count_];
    candidates_offset_ = new Vertex [query_vertex_count_+1];
    memset(visited_query_depth_, 0, sizeof(Vertex)*data_vertex_count_);
    memset(candidates_offset_, 0, sizeof(Vertex)*(query_vertex_count_+1));

    int cur_depth = 1;
    candidates_offset_[cur_depth] = 0;
    find_matches_ = new bool [query_vertex_count_+1];
    find_matches_[query_vertex_count_] = true;

    vector<float> candidate_score;
    Vertex u, v;
    // Vertex conflicted_query_vertex;
    uint32_t conflicted_depth;
    state_count_ = 0;
    emb_count_ = 0;

    vector<bitset<MAX_QUERY_SIZE>> failing_set_depth;
    failing_set_depth.resize(query_vertex_count_+1);

    // bool debug_print=false;
    Vertex* intersection_buffer = new Vertex[data_graph_->getGraphMaxDegree()];
    uint32_t intersection_buffer_size = 0;
    Vertex* intersection_buffer_tmp = new Vertex[data_graph_->getGraphMaxDegree()];
    uint32_t intersection_buffer_size_tmp = 0;

#if INDEX_ORDER == 1
    vector<vector<pair<Vertex, float>>> ordered_candidates;
    ordered_candidates.resize(order_.size());
    unordered_map<Vertex, float>& score_map = encoder->scores[1];
    for(int x=0;x<extending_candidates[1].content_size;++x){
        Vertex vc = extending_candidates[1].content[x];
        ordered_candidates[1].push_back({vc, score_map[vc]});
    }
    sort(ordered_candidates[1].begin(), ordered_candidates[1].end(), cmp_score);
#endif

    while(true){
#if INDEX_ORDER==1
        while(candidates_offset_[cur_depth]<ordered_candidates[cur_depth].size())
#else
        while(candidates_offset_[cur_depth]<extending_candidates[cur_depth].content_size)
#endif
        {
            if(stop_==true){
                for(int i=1;i<=cur_depth;++i){
                    cout<<embedding_depth_[i]<<" ";
                }
                cout<<endl;
                goto EXIT;
            }
            

            Vertex* current_candidates = extending_candidates[cur_depth].content;
            state_count_ ++;
            find_matches_[cur_depth] = false;
            u = order_[cur_depth];
            uint32_t offset = candidates_offset_[cur_depth];
#if INDEX_ORDER == 1
            v = ordered_candidates[cur_depth][offset].first;
#else
            v = current_candidates[offset];
#endif
            embedding_depth_[cur_depth] = v;

            // cout<<"searching:"<<cur_depth<<":"<<u<<":"<<v<<":"<<candidates_offset_[cur_depth]<<endl;

            if(cur_depth == query_vertex_count_){
                for(int i=0;i<extending_candidates[cur_depth].content_size; ++i){
#if INDEX_ORDER == 1
                    v = ordered_candidates[cur_depth][i].first;
#else
                    v = current_candidates[i];
#endif
                    state_count_ ++;
                    if(visited_query_depth_[v] == 0){
                        find_matches_[cur_depth-1] = true;
                        find_matches_[cur_depth] = true;
                        emb_count_ ++;
                        embedding_depth_[cur_depth] = v;

#if PRINT_RESULT==1
                        matches_.push_back(embedding_depth_);
                        for(Vertex z=1;z<=query_vertex_count_; ++z){
                            cout<<embedding_depth_[z]<<" ";
                        }
                        // cout<<validate_correctness(query_graph_, data_graph_, order_index_, embedding_depth_);
                        cout<<endl;
                        // exit(0);
#endif
                        if(emb_count_ >= count_limit){
                            goto EXIT;
                        }
                    }else{

                        conflicted_depth = visited_query_depth_[v];
                    }
                    
                    // cout<<"conflict:"<<cur_depth<<":"<<v<<":"<<visited_query_depth_[v]<<":"<<state_count_<<endl;
                }
#if INDEX_ORDER==1
                candidates_offset_[cur_depth] = ordered_candidates[cur_depth].size();
#else
                candidates_offset_[cur_depth] = extending_candidates[cur_depth].content_size;
#endif
                // candidates_offset_[cur_depth] = searching_candidates_[cur_depth].size();
            }else{
                // normal enumeration
                candidates_offset_[cur_depth] ++;

                
                failing_set_depth[cur_depth].reset();

                if(visited_query_depth_[v] > 0){
                    conflicted_depth = visited_query_depth_[v];
         

                    // cout<<"conflict:"<<cur_depth<<":"<<v<<":"<<visited_query_depth_[v]<<":"<<state_count_<<endl;
                    continue;
                }

                // start extending
                visited_query_depth_[v] = cur_depth;
                // if(debug(cur_depth, &(embedding_depth_[0]), order_)){
                //     // if(cur_depth == 4)
                //     //     debug_print = true;
                //     cout<<"search:"<<cur_depth<<":"<<u<<":"<<v<<endl;
                // }

                // start intersection
                uint32_t next_depth = cur_depth+1;
                uint32_t pred_depth = predecessor_neighbors_in_depth_[next_depth][0];
                uint32_t count;
                Vertex* cans = encoder->get_edge_candidate(pred_depth, next_depth, embedding_depth_[pred_depth], count);
                memcpy(extending_candidates[next_depth].content, cans, sizeof(Vertex)*count);
                extending_candidates[next_depth].content_size = count;
                for(int x=1;x<predecessor_neighbors_in_depth_[next_depth].size();++x){
                    pred_depth = predecessor_neighbors_in_depth_[next_depth][x];
                    Vertex* cans = encoder->get_edge_candidate(pred_depth, next_depth, embedding_depth_[pred_depth], count);
                    ComputeSetIntersection::ComputeCandidates(cans, count, extending_candidates[next_depth].content, extending_candidates[next_depth].content_size, extending_candidates_tmp[next_depth].content, extending_candidates_tmp[next_depth].content_size);
                    swap(extending_candidates[next_depth].content, extending_candidates_tmp[next_depth].content);
                    swap(extending_candidates[next_depth].content_size, extending_candidates_tmp[next_depth].content_size);
                }
#if INDEX_ORDER == 1
                unordered_map<Vertex, float>& score_map = encoder->scores[next_depth];
                vector<pair<Vertex, float>>& current_ordered_candidates = ordered_candidates[next_depth];
                current_ordered_candidates.clear();
                for(int x=0;x<extending_candidates[next_depth].content_size;++x){
                    Vertex vc = extending_candidates[next_depth].content[x];
                    current_ordered_candidates.push_back({vc, score_map[vc]});
                }
                sort(current_ordered_candidates.begin(), current_ordered_candidates.end(), cmp_score);
#endif
                // for(uint32_t x=0;x<predecessor_neighbors_in_depth_[next_depth].size();++x){
                //     uint32_t pred_depth = predecessor_neighbors_in_depth_[next_depth][x];
                //     uint32_t count;
                //     Vertex* cans = encoder->get_edge_candidate(pred_depth, next_depth, embedding_depth_[pred_depth], count);
                //     if(x == 0){
                //         memcpy(intersection_buffer, cans, count*sizeof(Vertex));
                //         intersection_buffer_size = count;
                //     }else{
                //         swap(intersection_buffer, intersection_buffer_tmp);
                //         swap(intersection_buffer_size, intersection_buffer_size_tmp);
                //         ComputeSetIntersection::ComputeCandidates(cans, count, intersection_buffer_tmp, intersection_buffer_size_tmp, intersection_buffer, intersection_buffer_size);
                //     }
                // }
                // searching_candidates_[next_depth] = vector<Vertex>(intersection_buffer, intersection_buffer+intersection_buffer_size);
                cur_depth++;
                candidates_offset_[cur_depth] = 0;
            }
        }
#if INDEX_ORDER==1
        if(ordered_candidates[cur_depth].size() == 0)
#else
        if(extending_candidates[cur_depth].content_size == 0)
#endif

        cur_depth --;
        if(cur_depth == 0){
            break;
        }

        v = embedding_depth_[cur_depth];
        visited_query_depth_[v] = 0;

    }
EXIT:
    end = std::chrono::high_resolution_clock::now();
    enumeration_time_ = NANOSECTOSEC(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    query_time_ += enumeration_time_;

    if(stop_ == false){
        timer_delete(id);
    }

#if PRINT_MEM_INFO == 1
    stop_thread = true;
    mem_info_thread.join();
    // sleep(1);
    float cur_mem = GetMemoryUsage(current_pid);
    peak_memory_ = (peak_memory_ > cur_mem) ? peak_memory_ : cur_mem;
    peak_memory_ -= starting_memory_cost;
#endif

    delete pp_;
    delete storage_;
    delete candidates_offset_;
    delete visited_query_depth_;
    delete find_matches_;
    delete intersection_buffer;
    delete intersection_buffer_tmp;
    for(int i=0;i<query_vertex_count_+1;++i){
        delete extending_candidates[i].content;
        delete extending_candidates_tmp[i].content;
    }

    pp_ = NULL;
    storage_ = NULL;
}
