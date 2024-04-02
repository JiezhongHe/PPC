#include "nlf_filter.h"

void nlf_filter::execute(std::vector<std::vector<uint32_t>> &candidate_sets) {
    uint32_t n = query_graph_->getVerticesCount();
    candidate_sets.resize(n);

    for (uint32_t u = 0; u < n; ++u) {
        uint32_t label = query_graph_->getVertexLabel(u);
        uint32_t degree = query_graph_->getVertexDegree(u);
#if OPTIMIZED_LABELED_GRAPH == 1
        auto u_nlf = query_graph_->getVertexNLF(u);
#endif

        uint32_t data_vertex_num;
        const uint32_t* data_vertices = data_graph_->getVerticesByLabel(label, data_vertex_num);
        auto& candidate_set = candidate_sets[u];

        for (uint32_t j = 0; j < data_vertex_num; ++j) {
            uint32_t v = data_vertices[j];
            if (data_graph_->getVertexDegree(v) >= degree) {

                // NLF check
#if OPTIMIZED_LABELED_GRAPH == 1
                auto v_nlf = data_graph_->getVertexNLF(v);

                if (v_nlf->size() >= u_nlf->size()) {
                    bool is_valid = true;

                    for (auto element : *u_nlf) {
                        auto iter = v_nlf->find(element.first);
                        if (iter == v_nlf->end() || iter->second < element.second) {
                            is_valid = false;
                            break;
                        }
                    }

                    if (is_valid) {
                        candidate_set.push_back(v);
                    }
                }
#endif
            }
        }
    }
}

void nlf_filter::execute(catalog *storage) {
    for (uint32_t u = 0; u < query_graph_->getVerticesCount(); ++u) {
        uint32_t u_nbrs_cnt;
        const uint32_t* u_nbrs = query_graph_->getVertexNeighbors(u, u_nbrs_cnt);
        uint32_t uu = u_nbrs[u_nbrs_cnt - 1];

        if (u < uu) {
            filter_ordered_relation(u, &storage->edge_relations_[u][uu], uu);
        }
        else {
            filter_unordered_relation(u, &storage->edge_relations_[uu][u], uu);
        }
    }
// #if ENABLE_PRE_FILTERING == 1

// #if COMPACT == 0
//     Value **query_edge_content = query_edge_emb->content;
//     Value **data_edge_content = data_edge_emb->content;
//     int val_dim = data_edge_emb->column_size;
// #else
//     Value **query_edge_content = query_edge_emb_comp->content;
//     Value **data_edge_content = data_edge_emb_comp->content;
//     int val_dim = data_edge_emb_comp->column_size;
// #endif
//     // Value **query_edge_content = query_edge_emb->content;
//     // Value **data_edge_content = data_edge_emb->content;
    
//     for (uint32_t u = 0; u < query_graph_->getVerticesCount(); ++u) {
//         uint32_t u_nbrs_cnt;
//         const uint32_t* u_nbrs = query_graph_->getVertexNeighbors(u, u_nbrs_cnt);
//         for(int i=0;i<u_nbrs_cnt;++i){
//             uint32_t u_n = u_nbrs[i];
//             if(u<u_n){
//                 edge_relation* relation = &storage->edge_relations_[u][u_n];
//                 Vertex q_e_id = query_graph_->edge_id_map[u][u_n];

//                 uint32_t valid_edge_count = 0;
//                 for(uint j=0;j<relation->size_;++j){
//                     uint32_t v0 = relation->edges_[j].vertices_[0];
//                     uint32_t v1 = relation->edges_[j].vertices_[1];
//                     Vertex d_e_id;
//                     if(v0 < v1){
//                         d_e_id = data_graph_->edge_id_map[v0][v1];
//                     }else{
//                         d_e_id = data_graph_->edge_id_map[v1][v0];
//                     }
// #if COMPACT == 0
//                     if(vec_validation(query_edge_content[q_e_id], data_edge_content[d_e_id], val_dim) == true){
//                         relation->edges_[valid_edge_count] = relation->edges_[j];
//                         valid_edge_count ++;
//                     }
// #else
//                     if(compact_vec_validation(query_edge_content[q_e_id], data_edge_content[d_e_id]) == true){
//                         relation->edges_[valid_edge_count] = relation->edges_[j];
//                         valid_edge_count ++;
//                     }
// #endif
//                 }
//                 relation->size_ = valid_edge_count;
//             }
//         }
//     }
// #endif
}

void nlf_filter::filter_ordered_relation(uint32_t u, edge_relation *relation, uint32_t other) {
    uint32_t u_deg = query_graph_->getVertexDegree(u);

#if ENABLE_PRE_FILTERING == 1
#if COMPACT == 0
    Value **query_vertex_content = query_vertex_emb->content;
    Value **data_vertex_content = data_vertex_emb->content;
    int val_dim = query_vertex_emb->column_size;
#else
    Value **query_vertex_content = query_vertex_emb_comp->content;
    Value **data_vertex_content = data_vertex_emb_comp->content;
    int val_dim = query_vertex_emb_comp->column_size;
#endif
#endif

#if OPTIMIZED_LABELED_GRAPH == 1
    auto u_nlf = query_graph_->getVertexNLF(u);
    std::vector<std::pair<uint32_t, uint32_t>> nlf_array;
    for (auto element : *u_nlf) {
        nlf_array.emplace_back(element);
    }
#endif

    uint32_t valid_edge_count = 0;
    uint32_t prev_v = std::numeric_limits<uint32_t>::max();
    bool add = false;

    for (uint32_t i = 0; i < relation->size_; ++i) {
        uint32_t v = relation->edges_[i].vertices_[0];

        if (v != prev_v) {
            prev_v = v;
            add = false;
            uint32_t v_deg = data_graph_->getVertexDegree(v);

            if (v_deg >= u_deg) {
                add = true;
#if OPTIMIZED_LABELED_GRAPH == 1
                auto v_nlf = data_graph_->getVertexNLF(v);

                if (v_nlf->size() >= nlf_array.size()) {
                    for (auto element : nlf_array) {
                        auto iter = v_nlf->find(element.first);
                        if (iter == v_nlf->end() || iter->second < element.second) {
                            add = false;
                            break;
                        }
                    }
                }
#endif
// #if ENABLE_PRE_FILTERING == 1
//                 if(add == true){
// #if COMPACT == 0
//                     if(vec_validation(query_vertex_content[u], data_vertex_content[v], val_dim) == false){
//                         add = false;
//                     }
// #else
//                     if(compact_vec_validation(query_vertex_content[u], data_vertex_content[v]) == false){
//                         add = false;
//                     }
// #endif
//                 }
// #endif
            }
        }

        if (add) {
            if(valid_edge_count != i)
                relation->edges_[valid_edge_count] = relation->edges_[i];
            valid_edge_count += 1;
        }
    }

    relation->size_ = valid_edge_count;
}

void nlf_filter::filter_unordered_relation(uint32_t u, edge_relation *relation, uint32_t other) {
    uint32_t u_deg = query_graph_->getVertexDegree(u);

#if OPTIMIZED_LABELED_GRAPH == 1
    auto u_nlf = query_graph_->getVertexNLF(u);
    std::vector<std::pair<uint32_t, uint32_t>> nlf_array;
    for (auto element : *u_nlf) {
        nlf_array.emplace_back(element);
    }
#endif

#if ENABLE_PRE_FILTERING == 1
#if COMPACT == 0
    Value **query_vertex_content = query_vertex_emb->content;
    Value **data_vertex_content = data_vertex_emb->content;
    int val_dim = query_vertex_emb->column_size;
#else
    Value **query_vertex_content = query_vertex_emb_comp->content;
    Value **data_vertex_content = data_vertex_emb_comp->content;
    int val_dim = query_vertex_emb_comp->column_size;
#endif
#endif

    uint32_t valid_edge_count = 0;
    for (uint32_t i = 0; i < relation->size_; ++i) {
        uint32_t v = relation->edges_[i].vertices_[1];

        // v is not checked.
        if (status_[v] == 'u') {
            status_[v] = 'r';
            updated_.push_back(v);
            uint32_t v_deg = data_graph_->getVertexDegree(v);
            if (v_deg >= u_deg) {
                status_[v] = 'a';
#if OPTIMIZED_LABELED_GRAPH == 1
                auto v_nlf = data_graph_->getVertexNLF(v);

                if (v_nlf->size() >= nlf_array.size()) {
                    for (auto element : nlf_array) {
                        auto iter = v_nlf->find(element.first);
                        if (iter == v_nlf->end() || iter->second < element.second) {
                            status_[v] = 'r';
                            break;
                        }
                    }
                }
#endif
// #if ENABLE_PRE_FILTERING == 1
//                 if(status_[v] == 'a'){
// #if COMPACT == 0
//                     if(vec_validation(query_vertex_content[u], data_vertex_content[v], val_dim) == false){
//                         status_[v] = 'r';
//                     }
// #else
//                     if(compact_vec_validation(query_vertex_content[u], data_vertex_content[v]) == false){
//                         status_[v] = 'r';
//                     }
// #endif
//                     // if(status_[v] == 'a'){
//                     //     Vertex d_e_id;
//                     //     Vertex v_n = relation->edges_[i].vertices_[1];
//                     //     if(v<v_n){
//                     //         d_e_id = data_graph_->edge_id_map[v][v_n];
//                     //     }else{
//                     //         d_e_id = data_graph_->edge_id_map[v_n][v];
//                     //     }
//                     //     if(vec_validation(query_edge_content[q_e_id], data_edge_content[d_e_id], val_dim) == false){
//                     //         status_[v] = 'r';
//                     //     }
//                     // }
//                 }
// #endif
            }
        }

        if (status_[v] == 'a') {
            if (valid_edge_count != i)
                relation->edges_[valid_edge_count] = relation->edges_[i];
            valid_edge_count += 1;
        }
    }

    for (auto v : updated_)
        status_[v] = 'u';
    updated_.clear();

    relation->size_ = valid_edge_count;
}
