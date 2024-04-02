#include "trie_encoder.h"

#if COMPACT == 0
double calc_score(Value* query_vec, Value* data_vec, int column_size){
    double result = 0.0;
    double sum = 0.0;
    for(int i=0;i<column_size;++i){
        if(query_vec[i] > data_vec[i]){
            result = -10;
            break;
        }
        // assert(query_vec[i] <= data_vec[i]);
        result += (data_vec[i]-query_vec[i])*query_vec[i];
        // result += tanh(data_vec[i]-query_vec[i])*query_vec[i];
        sum += query_vec[i];
    }
    return result/sum;
}
#else
double calc_score(Value* query_vec, Value* data_vec, int column_size){
    double result = 0.0;
    double sum = 0.0;
    int content_size1 = query_vec[0];
    int content_size2 = data_vec[0];
    Value* start_key1 = query_vec+1;
    Value* start_key2 = data_vec+1;
    Value* start_val1 = query_vec+1+content_size1;
    Value* start_val2 = data_vec+1+content_size2;

    for(int i=0,j=0; i<content_size1 && j<content_size2;){
        if(start_key1[i] == start_key2[j]){
            if(start_val2[j]<start_val1[i]){
                result = -10;
                break;
            }
            result += (start_val2[j]-start_val1[i])*start_val1[i];
            i++;
            j++;
        }else if(start_key1[i] < start_key2[j]){
            return false;
        }else{
            j++;
        }
    }
    for(int i=0;i<content_size1;++i){
        sum += start_val1[i];
    }
    return result/sum;
}
#endif

TrieRelation::TrieRelation(catalog* storage, Vertex src, Vertex dst){
    Vertex src_tmp = std::min(src, dst);
    Vertex dst_tmp = std::max(src, dst);
    src_ = src;
    dst_ = dst;
    edge_relation& target_edge_relation = storage->edge_relations_[src_tmp][dst_tmp];
    edge* edges = target_edge_relation.edges_;
    uint32_t edge_size = target_edge_relation.size_;
    uint32_t src_idx = 0, dst_idx = 1;
    if(src > dst){
        std::sort(edges, edges + edge_size, [](edge& l, edge& r) -> bool {
            if (l.vertices_[1] == r.vertices_[1])
                return l.vertices_[0] < r.vertices_[0];
            return l.vertices_[1] < r.vertices_[1];
        });
        swap(src_idx, dst_idx);
    }

    // start encoding
    uint32_t start_offset=0;
    children_ = new Vertex [edge_size];
    Vertex current_vertex = edges[0].vertices_[src_idx];
    children_[0] = edges[0].vertices_[dst_idx];
// #if INDEX_ORDER == 1
//     offsets_ = new Vertex [edge_size];
//     vector<double> scores;
//     Value **query_vertex_content = query_vertex_emb->content;
//     Value **data_vertex_content = data_vertex_emb->content;
//     int val_dim = query_vertex_emb->column_size;
// #endif
    for(uint32_t i=1; i<edge_size; ++i){
        Vertex candidate_src = edges[i].vertices_[src_idx];
        Vertex candidate_dst = edges[i].vertices_[dst_idx];

        if(candidate_src != current_vertex){
// #if INDEX_ORDER == 1
//             uint32_t cnt = 0;
//             for(uint32_t j=start_offset;j<i;++j,++cnt){
//                 offsets_[j] = cnt;
//             }
// #endif
            offset_map[current_vertex] = {start_offset+children_, i-start_offset};
            start_offset = i;
            current_vertex = candidate_src;
// #if INDEX_ORDER == 1
//             Vertex* start = offset_map[current_vertex].first;
//             uint32_t n_size = offset_map[current_vertex].second;
//             scores.clear();
//             for(uint32_t j=0; j<n_size; ++j){
//                 scores.push_back(calc_score(query_vertex_content[src], data_vertex_content[start[j]], val_dim));
//             }

//             for(uint32_t x=0;x<scores.size();++x){
//                 for(uint32_t y=x+1;y<scores.size();++y){
//                     if(scores[x] < scores[y]){
//                         swap(scores[x], scores[y]);
//                         swap(start[x], start[y]);
//                     }
//                 }
//             }
// #endif
        }
        children_[i] = candidate_dst;
    }
    if(start_offset<edge_size){
// #if INDEX_ORDER == 1
//         uint32_t cnt = 0;
//         for(uint32_t j=start_offset;j<edge_size;++j,++cnt){
//             offsets_[j] = cnt;
//         }
// #endif
        Vertex candidate_src = edges[start_offset].vertices_[src_idx];
        offset_map[candidate_src] = {start_offset+children_, edge_size-start_offset};
// #if INDEX_ORDER == 1
//         Vertex* start = offset_map[current_vertex].first;
//         uint32_t n_size = offset_map[current_vertex].second;
//         scores.clear();
//         for(uint32_t j=0; j<n_size; ++j){
//             scores.push_back(calc_score(query_vertex_content[src], data_vertex_content[start[j]], val_dim));
//         }
//         for(uint32_t x=0;x<scores.size();++x){
//             for(uint32_t y=x+1;y<scores.size();++y){
//                 if(scores[x] < scores[y]){
//                     swap(scores[x], scores[y]);
//                     swap(start[x], start[y]);
//                 }
//             }
//         }
// #endif
    }
}

void TrieRelation::remove_candidate(Vertex src_u, Vertex src_v, Graph* data_graph){
    if(src_u == src_){
        offset_map.erase(src_v);
    }else{
        uint32_t nbrs_count;
        const Vertex* nbrs = data_graph->getVertexNeighbors(src_v, nbrs_count);
        for(int i=0;i<nbrs_count;++i){
            Vertex v_n = nbrs[i];
            auto itf = offset_map.find(v_n);
            if(itf != offset_map.end()){
                Vertex* start = itf->second.first;
                const uint32_t size = itf->second.second;
                uint32_t valid_size = 0;
                for(int j=0;j<size;++j){
                    if(start[j] == src_v){
                        start[valid_size] = start[j];
                        valid_size ++;
                    }
                }
                itf->second.second = valid_size;
            }
        }
    }
}

TrieRelation::~TrieRelation(){
    delete children_;
}

TrieEncoder::TrieEncoder(catalog* storage, vector<Vertex>& order, vector<Vertex>& order_index, Graph* query_graph, Graph* data_graph){
    order_ = order;
    storage_ = storage;
    candidate_edges.resize(order.size());
    for(int i=0;i<order.size();++i){
        candidate_edges[i] = vector<TrieRelation*>(order.size(), NULL);
    }
    order_index_ = order_index;
    query_graph_ = query_graph;
    data_graph_ = data_graph;
    // start encoding
    for(int i=1;i<order_.size();++i){
        Vertex src = order_[i];

        uint32_t nbrs_count;
        const Vertex* nbrs = query_graph->getVertexNeighbors(src, nbrs_count);

        for(int j=0;j<nbrs_count;++j){
            Vertex dst = nbrs[j];
            uint32_t dst_depth = order_index[dst];
            if(order_index[src] < order_index[dst]){
                candidate_edges[i][dst_depth] = new TrieRelation(storage, src, dst);
            }
        }
    }
#if INDEX_ORDER == 1
#if COMPACT == 0
    Value **query_vertex_content = query_vertex_emb->content;
    Value **data_vertex_content = data_vertex_emb->content;
    int val_dim = query_vertex_emb->column_size;
#else
    Value **query_vertex_content = query_vertex_emb_comp->content;
    Value **data_vertex_content = data_vertex_emb_comp->content;
    int val_dim = query_vertex_emb_comp->column_size;
#endif
    scores.resize(order_.size());
    for(uint32_t u_depth=1;u_depth<order_.size();++u_depth){
        Vertex u = order_[u_depth];
        for(int i=0;i<order_.size();++i){
            Vertex u_n = order_[i];
            if(candidate_edges[u_depth][i] != NULL){
                for(auto& p : candidate_edges[u_depth][i]->offset_map){
                    // cout<<u<<":"<<p.first<<endl;
                    scores[u_depth].insert({p.first, calc_score(query_vertex_content[u], data_vertex_content[p.first], val_dim)});
                }
                break;
            }
        }
    }
#endif
}

void TrieEncoder::get_candidates(uint32_t u_depth, vector<Vertex>& result){
    for(int i=0;i<order_.size();++i){
        if(candidate_edges[u_depth][i] != NULL){
            for(auto& p : candidate_edges[u_depth][i]->offset_map){
                result.push_back(p.first);
            }
            sort(result.begin(), result.end());
            return;
        }
    }
}

#if INDEX_ORDER == 1
    void TrieEncoder::get_candidates_ordered(uint32_t u_depth, vector<Vertex>& result){
        for(int i=0;i<order_.size();++i){
            if(candidate_edges[u_depth][i] != NULL){
                for(auto& p : candidate_edges[u_depth][i]->offset_map){
                    result.push_back(p.first);
                }
                sort(result.begin(), result.end());
                return;
            }
        }
    }
#endif

Vertex* TrieEncoder::get_edge_candidate(uint32_t src_depth, uint32_t dst_depth, Vertex src_v, uint32_t& count){
    auto& p = candidate_edges[src_depth][dst_depth]->offset_map[src_v];
    count = p.second;
    // if(count == 0){
    //     uint32_t nbrs_count;
    //     cout<<"remove:"<<src_depth<<":"<<src_v<<endl;
    //     const Vertex* nbrs = query_graph_->getVertexNeighbors(order_[src_depth], nbrs_count);
    //     for(int i=0;i<count;++i){
    //         Vertex u_n = nbrs[i];
    //         uint32_t depth_n = order_index_[u_n];
    //         if(depth_n>src_depth){
    //             candidate_edges[src_depth][depth_n]->remove_candidate(src_depth, src_v, data_graph_);
    //         }else{
    //             candidate_edges[depth_n][src_depth]->remove_candidate(depth_n, src_v, data_graph_);
    //         }
    //     }
    // }
    return p.first;
}

TrieEncoder::~TrieEncoder(){
    for(int i=0;i<candidate_edges.size();++i){
        for(int j=0;j<candidate_edges[i].size();++j){
            if(candidate_edges[i][j] != NULL){
                delete candidate_edges[i][j];
            }
        }
    }
}
