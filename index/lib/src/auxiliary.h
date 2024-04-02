#pragma once
#include <string.h>
#include <algorithm>
#include <cmath>

#include "../configuration/config.h"

#include "../utility/embedding.h"
#include "../utility/core_decomposition.h"
#include "../utility/utils.h"
#include "../graph/graph.h"

#include "../nd/nucleus_decomposition.h"


bool check_nlf(Graph* query, int query_anchor, Graph* data, int data_anchor);
bool cmp_score(const pair<Vertex, float>& f1, const pair<Vertex, float>& f2);
bool cmp_uint(const pair<Vertex, uint32_t>& f1, const pair<Vertex, uint32_t>& f2);

inline void sort_candidates_by_id(vector<Vertex>& candidate, vector<float>& score);
inline void sort_candidates_by_score(vector<Vertex>& candidate, vector<float>& score);

class tree_node{
public:
    vector<Vertex> candidate;
    vector<float> candidate_score;
    vector<vector<pair<vector<Vertex>, vector<float>>>> neighbor_candidates; // query_id, data_offset,pair<neighbor id, score>
};


class BitMap_dynamic{
private:
    uint32_t bit_size;
    uint8_t char_size;
    char* content;
public:
    BitMap_dynamic();

    BitMap_dynamic(uint32_t size);

    BitMap_dynamic(const BitMap_dynamic& bitmap);

    BitMap_dynamic& operator=(const BitMap_dynamic& bitmap);

    void resize(uint32_t size);

    void set();

    void reset();

    void set(uint32_t i);

    void reset(uint32_t i);

    bool test(uint32_t i);

    ~BitMap_dynamic();
};

class Candidate_node{
public:
    vector<Vertex> candidate;
    vector<vector<Vertex>> neighbor_candidate;
};

class Auxiliary{
private:
    Graph* data_graph;
    Graph* query_graph;
    Tensor* query_vertex_emb;
    Tensor* data_vertex_emb;
    Tensor* query_edge_emb;
    Tensor* data_edge_emb;

    
    vector<vector<Vertex>> candidate_query;

    uint32_t query_vertex_count, query_edge_count, data_vertex_count, data_edge_count;
    // to remove
    bool** candidate_bit_map;

    bool* flag;
    bool enable_order;

    vector<Vertex> edges_to_prune;

public:
    // Vertex query_vertex_count, data_vertex_count;
    vector<Vertex> order;
    vector<vector<Vertex>> predecessor_neighbors;
    vector<vector<Vertex>> successor_neighbors;

    vector<uint32_t> vertex_candidate_info;
    vector<unordered_map<Vertex, uint32_t>> edge_candidate_info;

#ifdef FAILING_SET_PRUNING
    vector<bitset<MAX_QUERY_SIZE>> ancestors;
#endif

#ifdef CORE_DECOMPOSITION
    vector<bool> cut_vertices;
    vector<int> parent_cut_vertices;
#endif

    tree_node* candidates;
    // vector<Candidate_node> candidates;

    // used for comparison between edge_embeddings
    vector<unordered_map<Vertex, Vertex>>* query_edge_id_map;
    vector<unordered_map<Vertex, Vertex>>* data_edge_id_map;
    Value** query_edge_emb_content, ** data_edge_emb_content;
    int edge_emb_column_size;
    // used for comparison between vertex_embeddings
    Value** query_vertex_emb_content, ** data_vertex_emb_content;
    int vertex_emb_column_size;

    Auxiliary(Graph* data_graph_, Graph* query_graph_, Tensor* query_vertex_emb_, Tensor* data_vertex_emb_, Tensor* query_edge_emb_, Tensor* data_edge_emb_, bool enable_order_);

    inline uint32_t get_query_edge_id_for_candidate_bitmap(Vertex u1, Vertex u2);

    inline uint32_t get_data_edge_id_for_candidate_bitmap(Vertex u1, Vertex u2);

    void construct();

    void LDF_NLF_initialization(vector<BitMap_dynamic>& candidate_vertex_bitmap);

    void link_and_prune(vector<BitMap_dynamic>& candidate_vertex_bitmap, vector<unordered_set<Vertex>>& filtered_edge_bitmap);

    double calc_score(Value* query_vec, Value* data_vec, int column_size);

    void print();

    inline bool validate_edge_emb(Vertex& u, Vertex& u_, Vertex& v, Vertex& v_);

    inline bool validate_vertex_emb(Vertex& u, Vertex& v);

    void get_candidate_info(uint32_t& candidate_vertices, uint32_t& candidate_edges);

    vector<pair<Vertex, uint32_t>> get_ordered_vertices_by_candidate_count();

    void update_pre_suc_neighbors(vector<Vertex>& search_order);

    void reorder_leaf_vertices();

    // GraphQL style
    void generate_order_in_GQL();

    void generate_order_by_nucleus_decompostion();

    // Generate ordering with predecessors
    void generate_order_by_predecessors();

    bool cmp_candidate(vector<pair<Vertex, int>>& candidate_size, vector<Vertex>& vec1, vector<Vertex>& vec2);

    Vertex pick_least_candidate(vector<pair<Vertex, int>>& candidate_size, unordered_set<Vertex>& vec);

#ifdef FAILING_SET_PRUNING
    void initilize_ancestors();
#endif

// #ifdef CORE_DECOMPOSITION
//     void query_core_decomposition();
// #endif

    void generate_search_order(vector<Vertex>& search_order, vector<Vertex>& search_order_offset);

    ~Auxiliary();

private:
    void generate_order_in();
};


// an backup ordering plan based on core decomposition
// #ifdef CORE_DECOMPOSITION
//     // generate order with failing set pruning
//     void generate_order_in_CORE_DECOMPOSITION_by_chunk(){
//         if(!order.empty()){
//             return;
//         }
//         vector<pair<Vertex, int>> candidates_size = get_ordered_vertices_by_candidate_count();

//         // decompose the query
//         Core query_cores(query_graph);
//         parent_check_point.resize(query_graph->label_map.size());

//         // order by the cores as well as the candidates
//         unordered_set<Vertex> next_vertices;
//         unordered_set<Vertex> next_cores;
//         unordered_set<Vertex> ordered_vertices;
//         unordered_set<Vertex> ordered_cores;
//         vector<Vertex> vertex_to_order_offset;
//         vertex_to_order_offset.resize(query_graph->label_map.size());
//         int current_core_id;
//         for(int i=0;i<query_graph->label_map.size();++i){
//             Vertex added_vertex;
//             bool update_cores = false;
//             int parent_core_id;
//             if(i == 0){
//                 update_cores = true;
//                 added_vertex = candidates_size[0].first;
//                 if(query_cores.cutVertices.find(added_vertex) == query_cores.cutVertices.end()){
//                     current_core_id = query_cores.reversed_index[added_vertex];
//                 }else{
//                     current_core_id = query_cores.reversed_index_for_cut_vertex[added_vertex][0];
//                     for(int j=1; j<query_cores.reversed_index_for_cut_vertex[added_vertex].size(); ++j){
//                         int id = query_cores.reversed_index_for_cut_vertex[added_vertex][j];
//                         if(cmp_candidate(candidates_size, query_cores.cores[current_core_id], query_cores.cores[id])){
//                             current_core_id = id;
//                         }
//                     }
//                 }
//             }else{
//                 if(next_vertices.empty()){
//                     // pick the core with the least candidates
//                     update_cores = true;
//                     parent_core_id = current_core_id;
//                     current_core_id = *(next_cores.begin());
//                     for(auto c_id : next_cores){
//                         if(cmp_candidate(candidates_size, query_cores.cores[current_core_id], query_cores.cores[c_id])){
//                             current_core_id = c_id;
//                         }
//                     }
//                     // start order a new core
//                     unordered_set<Vertex> can;
//                     for(auto v : query_cores.cores[current_core_id]){
//                         // connected to the ordered vertices
//                         for(auto m_v : ordered_vertices){
//                             if(query_graph->adj[v].find(m_v) != query_graph->adj[v].end() && ordered_vertices.find(v) == ordered_vertices.end()){
//                                 can.insert(v);
//                                 break;
//                             }
//                         }
//                     }
//                     added_vertex = pick_least_candidate(candidates_size, can);
//                 }else{
//                     added_vertex = pick_least_candidate(candidates_size, next_vertices);
//                 }
//             }
            
//             vertex_to_order_offset[added_vertex] = order.size();
//             order.push_back(added_vertex);
//             ordered_vertices.insert(added_vertex);
//             if(next_vertices.empty() && i>0){
//                 // parent_check_point[added_vertex] = query_cores.adj[current_core_id][parent_core_id];
//                 for(auto v : query_cores.cores[current_core_id]){
//                     if(query_cores.cutVertices.find(v) != query_cores.cutVertices.end() && ordered_vertices.find(v) != ordered_vertices.end()){
//                         parent_check_point[added_vertex] = vertex_to_order_offset[v];
//                         break;
//                     }
//                 }
//                 cut_check_point.push_back(true);
//             }else{
//                 cut_check_point.push_back(false);
//             }
            

//             // update the next_vertices and next_cores
//             if(query_cores.cutVertices.find(added_vertex) == query_cores.cutVertices.end()){
//                 for(auto v_n : query_graph->adj[added_vertex]){
//                     if(ordered_vertices.find(v_n) == ordered_vertices.end()){
//                         next_vertices.insert(v_n);
//                     }
//                 }
//                 next_vertices.erase(added_vertex);
//             }else{
//                 for(auto v_n : query_graph->adj[added_vertex]){
//                     bool whether_in_core = false;
//                     for(auto v : query_cores.cores[current_core_id]){
//                         if(v == v_n){
//                             whether_in_core = true;
//                             break;
//                         }
//                     }
//                     if(ordered_vertices.find(v_n) == ordered_vertices.end() && whether_in_core == true){
//                         next_vertices.insert(v_n);
//                     }
//                 }
//                 next_vertices.erase(added_vertex);
//             }
//             // update cores
//             if(update_cores == true){
//                 ordered_cores.insert(current_core_id);
//                 for(auto p : query_cores.adj[current_core_id]){
//                     if(ordered_cores.find(p.first) == ordered_cores.end()){
//                         next_cores.insert(p.first);
//                     }
//                 }
//                 next_cores.erase(current_core_id);
//             }
//         }
//     }

//     void generate_order_in_CORE_DECOMPOSITION_by_orginal_GQL(){
//         if(!order.empty()){
//             return;
//         }
//         Vertex query_vertex_count = query_graph->label_map.size();
//         vector<pair<Vertex, int>> candidates_size = get_ordered_vertices_by_candidate_count();

//         unordered_set<Vertex> next_candidates = query_graph->adj[candidates_size[0].first];
//         unordered_set<Vertex> added_candidates = {candidates_size[0].first};
//         vector<Vertex> vertex_to_order_offset;
//         vertex_to_order_offset.resize(query_vertex_count);

//         order.push_back(candidates_size[0].first);
//         vertex_to_order_offset[candidates_size[0].first] = 0;
//         for(int i=1;i<query_vertex_count;++i){
//             for(int j=0;j<query_vertex_count;++j){
//                 if(added_candidates.find(candidates_size[j].first) != added_candidates.end()){
//                     continue;
//                 }
//                 if(next_candidates.find(candidates_size[j].first) != next_candidates.end()){
//                     order.push_back(candidates_size[j].first);
//                     vertex_to_order_offset[candidates_size[j].first] = i;
//                     for(auto u:query_graph->adj[candidates_size[j].first]){
//                         next_candidates.insert(u);
//                     }
//                     added_candidates.insert(candidates_size[j].first);
//                     break;
//                 }
//             }
//         }
//         // decompose the query
//         Core query_cores(query_graph);

//         // order = {5, 20, 22, 27, 29, 19, 28, 23, 30, 36, 32, 34, 24, 6, 17, 26, 35, 15, 10, 11, 7, 4, 21, 9, 2, 13, 12, 33, 18, 0, 14, 37, 3, 16, 25, 8, 38, 1, 31, 39};
//         // for(int i=0;i<order.size();++i){
//         //     vertex_to_order_offset[order[i]] = i;
//         // }

//         // detect cut vertices
//         vector<vector<Vertex>> ordered_cores;
//         ordered_cores.resize(query_cores.cores.size());
//         for(int i=0; i<order.size(); ++i){
//             Vertex v = order[i];
//             if(query_cores.cutVertices.find(v) == query_cores.cutVertices.end()){
//                 int current_core_id = query_cores.reversed_index[v];
//                 ordered_cores[current_core_id].push_back(v);
//             }else{
//                 for(auto id : query_cores.reversed_index_for_cut_vertex[v]){
//                     ordered_cores[id].push_back(v);
//                 }
//             }
//         }
//         // cout<<"cout ordered cores:"<<endl;
//         // for(auto core : ordered_cores){
//         //     cout<<"{";
//         //     for(auto v : core){
//         //         cout<<v<<", ";
//         //     }
//         //     cout<<"}"<<endl;
//         // }
//         // cout<<"----------------------"<<endl;
//         // compute the parent
//         parent_check_point.resize(query_vertex_count);
//         cut_check_point = vector<bool>(query_vertex_count, false);
//         for(auto ordered_core : ordered_cores){
//             cut_check_point[vertex_to_order_offset[ordered_core[1]]] = true;
//             parent_check_point[ordered_core[1]] = vertex_to_order_offset[ordered_core[0]];
//         }
//     }
// #endif