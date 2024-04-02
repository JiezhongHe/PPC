#include "auxiliary.h"

bool check_nlf(Graph* query, int query_anchor, Graph* data, int data_anchor){
    unordered_map<Vertex, Vertex>& query_dist = query->nlf_data[query_anchor];
    unordered_map<Vertex, Vertex>& data_dist = data->nlf_data[data_anchor];
    for(auto p : query_dist){
        auto itf = data_dist.find(p.first);
        if(itf == data_dist.end()){
            return false;
        }
        if(p.second > itf->second){
            return false;
        }
    }
    return true;
}

bool cmp_score(const pair<Vertex, float>& f1, const pair<Vertex, float>& f2){
    return f1.second > f2.second;
}

bool cmp_uint(const pair<Vertex, uint32_t>& f1, const pair<Vertex, uint32_t>& f2){
    return f1.second < f2.second;
}

inline void sort_candidates_by_id(vector<Vertex>& candidate, vector<float>& score){
    int size = candidate.size();
    for(int i=0;i<size-1;++i){
        for(int j=i+1;j<size;++j){
            if(candidate[i]>candidate[j]){
                swap(candidate[i],candidate[j]);
                swap(score[i],score[j]);
            }
        }
    }
}

inline void sort_candidates_by_score(vector<Vertex>& candidate, vector<float>& score){
    int size = candidate.size();
    for(int i=0;i<size-1;++i){
        for(int j=i+1;j<size;++j){
            if(score[i]<score[j]){
                swap(candidate[i],candidate[j]);
                swap(score[i],score[j]);
            }
        }
    }
}

BitMap_dynamic::BitMap_dynamic(){
    content = NULL;
}

BitMap_dynamic::BitMap_dynamic(uint32_t size){
    char_size = sizeof(char);
    bit_size = size/char_size+1;
    content = new char [bit_size];
    memset(content, 0, char_size*bit_size);
}

void BitMap_dynamic::resize(uint32_t size){
    char_size = sizeof(char);
    bit_size = size/char_size+1;
    content = new char [bit_size];
    memset(content, 0, char_size*bit_size);
}

BitMap_dynamic::BitMap_dynamic(const BitMap_dynamic& bitmap){
    bit_size = bitmap.bit_size;
    content = new char [bit_size];
    char_size = sizeof(char);
    memcpy(content, bitmap.content, char_size*bit_size);
}

BitMap_dynamic& BitMap_dynamic::operator=(const BitMap_dynamic& bitmap){
    bit_size = bitmap.bit_size;
    content = new char [bit_size];
    char_size = sizeof(char);
    memcpy(content, bitmap.content, char_size*bit_size);
    return *this;
}

void BitMap_dynamic::set(){
    memset(content, 1, char_size*bit_size);
}

void BitMap_dynamic::reset(){
    memset(content, 0, char_size*bit_size);
}

void BitMap_dynamic::set(uint32_t i){
    content[i/char_size] |= 1<<(i%char_size);
}

void BitMap_dynamic::reset(uint32_t i){
    content[i/char_size] &= ~(1<<(i%char_size));
}

bool BitMap_dynamic::test(uint32_t i){
    if(content[i/char_size] & 1<<(i%char_size)>0){
        return true;
    }
    return false;
}

BitMap_dynamic::~BitMap_dynamic(){
    if(content != NULL)
        delete [] content;
}

///////////////////////
Auxiliary::Auxiliary(Graph* data_graph_, Graph* query_graph_, Tensor* query_vertex_emb_, Tensor* data_vertex_emb_, Tensor* query_edge_emb_, Tensor* data_edge_emb_, bool enable_order_){
    data_graph = data_graph_;
    query_graph = query_graph_;
    query_vertex_emb = query_vertex_emb_;
    data_vertex_emb = data_vertex_emb_;
    query_edge_emb = query_edge_emb_;
    data_edge_emb = data_edge_emb_;
    enable_order = enable_order_;
    data_graph->set_up_edge_id_map();
    query_graph->set_up_edge_id_map();
    data_edge_id_map = &(data_graph->edge_id_map);
    query_edge_id_map = &(query_graph->edge_id_map);
    query_vertex_count = query_graph->label_map.size();
    data_vertex_count = data_graph->label_map.size();
    query_edge_count = query_graph->get_edge_count();
    data_edge_count = data_graph->get_edge_count();
    if(query_edge_emb != NULL){
        data_edge_emb_content = data_edge_emb->content;
        query_edge_emb_content = query_edge_emb->content;
        edge_emb_column_size = data_edge_emb->column_size;
    }
    if(query_vertex_emb != NULL){
        data_vertex_emb_content = data_vertex_emb->content;
        query_vertex_emb_content = query_vertex_emb->content;
        vertex_emb_column_size = data_vertex_emb->column_size;
    }
    if(query_vertex_emb==NULL && query_edge_emb == NULL){
        enable_order = false;
    }
    construct();
}

inline uint32_t Auxiliary::get_query_edge_id_for_candidate_bitmap(Vertex u1, Vertex u2){
    if(u1<u2){
        return (*query_edge_id_map)[u1][u2];
    }else{
        return (*query_edge_id_map)[u2][u1]+query_edge_count;
    }
}

inline uint32_t Auxiliary::get_data_edge_id_for_candidate_bitmap(Vertex u1, Vertex u2){
    if(u1<u2){
        return (*data_edge_id_map)[u1][u2];
    }else{
        return (*data_edge_id_map)[u2][u1]+data_edge_count;
    }
}

void Auxiliary::construct(){
    // struct timeval start_t, end_t;
    vector<BitMap_dynamic> candidate_vertex_bitmap;
    vector<unordered_set<Vertex>> filtered_edges;
    // gettimeofday(&start_t, NULL);
    candidate_vertex_bitmap.resize(query_vertex_count);
    // filtered_edge_bitmap.resize(query_edge_count*2);
    for(Vertex u=0;u<query_vertex_count;++u){
        candidate_vertex_bitmap[u].resize(data_vertex_count);
    }
    // gettimeofday(&end_t, NULL);
    // cout<<"vertex-init:"<<get_time(start_t, end_t)<<endl;
    // gettimeofday(&start_t, NULL);
    // for(Vertex e_id=0; e_id<query_edge_count*2; ++e_id){
    //     filtered_edge_bitmap[e_id].resize(data_edge_count*2);
    // }
    filtered_edges.resize(query_edge_count*2);
    // gettimeofday(&end_t, NULL);
    // cout<<"edge-init:"<<get_time(start_t, end_t)<<endl;

    // gettimeofday(&start_t, NULL);
    LDF_NLF_initialization(candidate_vertex_bitmap);
    // gettimeofday(&end_t, NULL);
    // cout<<"initialization:"<<get_time(start_t, end_t)<<endl;
    // gettimeofday(&start_t, NULL);
    link_and_prune(candidate_vertex_bitmap, filtered_edges);
    // gettimeofday(&end_t, NULL);
    // cout<<"pruning:"<<get_time(start_t, end_t)<<endl;
    // gettimeofday(&start_t, NULL);
    // link(candidate_vertex_bitmap, candidate_edge_bitmap);
    // gettimeofday(&end_t, NULL);
    // cout<<"linking:"<<get_time(start_t, end_t)<<endl;
}

void Auxiliary::LDF_NLF_initialization(vector<BitMap_dynamic>& candidate_vertex_bitmap){
    query_graph->build_nlf();
    data_graph->build_nlf();
    vector<Label>& query_label_map = query_graph->label_map;
    vector<Label>& data_label_map = data_graph->label_map;
    candidates = new tree_node [query_vertex_count*2];
    // initially filter vertices
    for(Vertex u=0; u<query_vertex_count; ++u){
        for(Vertex v=0; v<data_vertex_count; ++v){
            if(query_label_map[u] == data_label_map[v] && check_nlf(query_graph, u, data_graph, v)){
                if(query_vertex_emb != NULL){
                    if(validate_vertex_emb(u, v)==true){
                        candidate_vertex_bitmap[u].set(v);
                    }
                }else{
                    candidate_vertex_bitmap[u].set(v);
                }
            }
        }
    }
}

void Auxiliary::link_and_prune(vector<BitMap_dynamic>& candidate_vertex_bitmap, vector<unordered_set<Vertex>>& filtered_edges){
    // first scan
    vector<unordered_set<Vertex>> to_check;
    uint32_t to_check_size = 0;
    to_check.resize(query_vertex_count);
    for(Vertex u=0; u<query_vertex_count; ++u){
        for(Vertex v=0; v<data_vertex_count; ++v){
            if(candidate_vertex_bitmap[u].test(v) == true){
                to_check[u].insert(v);
                to_check_size ++;
            }
        }
    }
    
    // iterative pruning
    while(to_check_size > 0){
        vector<unordered_set<Vertex>> to_check_next;
        to_check_next.resize(query_vertex_count);
        uint32_t to_check_next_size = 0;
        // geting auxiliary node (u,v) to check
        for(Vertex u=0; u<query_vertex_count; ++u){
            for(auto v : to_check[u]){
                bool whether_filter = false;
                // get the neighbor (u_n, v_n) of node (u,v)
                for(auto u_n : query_graph->adj[u]){
                    bool pass = false;
                    for(auto v_n : data_graph->adj[v]){
                        if(candidate_vertex_bitmap[u_n].test(v_n) == true){
                            pass = true;
                            break;
                        }
                    }
                    if(pass == false){
                        whether_filter = true;
                        break;
                    }
                }
                if(whether_filter == true){
                    candidate_vertex_bitmap[u].reset(v);
                    for(auto u_n : query_graph->adj[u]){
                        for(auto v_n : data_graph->adj[v]){
                            if(candidate_vertex_bitmap[u_n].test(v_n) == true){
                                to_check_next[u_n].insert(v_n);
                                to_check_next_size ++;
                            }
                        }
                    }
                }
            }
        }
        swap(to_check, to_check_next);
        to_check_size = to_check_next_size;
    }
    // link up the edges
    // struct timeval start_t, end_t;
    // gettimeofday(&start_t, NULL);
    vertex_candidate_info.resize(query_vertex_count);
    edge_candidate_info.resize(query_vertex_count);
    vector<unordered_map<Vertex, uint32_t>> candidate_offset_map;
    candidate_offset_map.resize(query_vertex_count);
    for(Vertex u=0; u<query_vertex_count; ++u){
        uint32_t vertex_candidate_count = 0;
        vector<Vertex>& cans = candidates[u].candidate;
        vector<pair<Vertex, float>> candidates_score;
        vector<float>& can_score = candidates[u].candidate_score;
        for(Vertex v=0; v<data_vertex_count; ++v){
            if(candidate_vertex_bitmap[u].test(v) == true){
                if(query_vertex_emb != NULL && enable_order == true){
                    float score = calc_score(query_vertex_emb_content[u], data_vertex_emb_content[v], vertex_emb_column_size);
                    can_score.push_back(score);
                    candidates_score.push_back({v, score});
                }else{
                    candidate_offset_map[u].insert({v, cans.size()});
                    cans.push_back(v);
                }
                // update the info
                ++ vertex_candidate_count;
                for(auto u_n : query_graph->adj[u]){
                    uint32_t edge_candidate_count = 0;
                    if(u < u_n){
                        for(auto v_n : data_graph->adj[v]){
                            if(candidate_vertex_bitmap[u_n].test(v_n) == true){
                                ++ edge_candidate_count;
                            }
                        }
                    }
                    edge_candidate_info[u].insert({u_n, edge_candidate_count});
                }
            }
        }
        can_score.shrink_to_fit();
        // reorder the candidates
        if(query_vertex_emb != NULL && enable_order == true){
            sort(candidates_score.begin(), candidates_score.end(), cmp_score);
            for(auto p : candidates_score){
                candidate_offset_map[u].insert({p.first, cans.size()});
                cans.push_back(p.first);
            }
        }
        cans.shrink_to_fit();
        vertex_candidate_info[u] = vertex_candidate_count;
    }
    // gettimeofday(&end_t, NULL);
    // cout<<"linkup-1:"<<get_time(start_t, end_t)<<endl;
    // generate the search order
    generate_order_in();
    update_pre_suc_neighbors(order);
    // link up the candidates and filter the invalid edges with edge-leveled index
    // gettimeofday(&start_t, NULL);
    to_check.clear();
    to_check.resize(query_vertex_count);
    to_check_size = 0;
    for(uint32_t i=0; i<order.size(); ++i){
        Vertex u = order[i];
        candidates[u].neighbor_candidates.resize(query_vertex_count);
        for(auto u_n : successor_neighbors[u]){
            candidates[u].neighbor_candidates[u_n].resize(candidates[u].candidate.size());
            for(uint32_t x=0; x<candidates[u].candidate.size(); ++x){
                Vertex v = candidates[u].candidate[x];
                vector<Vertex>& cans = candidates[u].neighbor_candidates[u_n][x].first;
                for(auto v_n : data_graph->adj[v]){
                    if(candidate_vertex_bitmap[u_n].test(v_n) && candidate_offset_map[u_n].find(v_n) != candidate_offset_map[u_n].end()){
                        // validate the edges
                        if(query_edge_emb != NULL){
                            if(validate_edge_emb(u, u_n, v, v_n) == true){
                                cans.push_back(candidate_offset_map[u_n][v_n]);
                            }else{
                                // both auxiliary nodes are nessary to be checked
                                Vertex q_e_id = get_query_edge_id_for_candidate_bitmap(u, u_n);
                                Vertex d_e_id = get_data_edge_id_for_candidate_bitmap(v, v_n);
                                // filtered_edge_bitmap[q_e_id].set(d_e_id);
                                filtered_edges[q_e_id].insert(d_e_id);
                                to_check[u].insert(v);
                                to_check[u_n].insert(v_n);
                                to_check_size += 2;
                            }
                        }else{
                            cans.push_back(candidate_offset_map[u_n][v_n]);
                        }
                    }
                }
                cans.shrink_to_fit();
                sort(cans.begin(), cans.end());
            }
        }
    }
    // gettimeofday(&end_t, NULL);
    // cout<<"linkup-2:"<<get_time(start_t, end_t)<<endl;
    // additionally iteratively filter invalid vertex
    if(query_edge_emb != NULL){
        // only check the vertex-level
        while(to_check_size > 0){
            // cout<<"addtional filtering"<<endl;
            vector<unordered_set<Vertex>> to_check_next;
            to_check_next.resize(query_vertex_count);
            uint32_t to_check_next_size = 0;
            for(Vertex u=0; u<query_vertex_count; ++u){
                for(auto v : to_check[u]){
                    // validate the auxiliary node
                    bool whether_filter = false;
                    // get the neighbor (u_n, v_n) for node (u, v)
                    for(auto u_n : query_graph->adj[u]){
                        bool pass = false;
                        for(auto v_n : data_graph->adj[v]){
                            if(candidate_vertex_bitmap[u_n].test(v_n) == true){
                                Vertex q_e_id = get_query_edge_id_for_candidate_bitmap(u, u_n);
                                Vertex d_e_id = get_data_edge_id_for_candidate_bitmap(v, v_n);
                                // if(filtered_edge_bitmap[q_e_id].test(d_e_id) == true){
                                //     continue;
                                // }
                                if(filtered_edges[q_e_id].find(d_e_id) != filtered_edges[q_e_id].end()){
                                    continue;
                                }
                                pass = true;
                                break;
                            }
                        }
                        if(pass = false){
                            whether_filter = true;
                            break;
                        }
                    }
                    if(whether_filter == true){
                        candidate_vertex_bitmap[u].reset(v);
                        for(auto u_n : query_graph->adj[u]){
                            for(auto v_n : data_graph->adj[v]){
                                if(candidate_vertex_bitmap[u_n].test(v_n) == true){
                                    to_check_next[u_n].insert(v_n);
                                    to_check_next_size ++;
                                }
                            }
                        }
                    }
                }
            }
            swap(to_check, to_check_next);
            to_check_size = to_check_next_size;
        }
    }
    // uint32_t can_vertex=0, can_edge=0;
    // get_candidate_info(can_vertex, can_edge);
    // cout<<"candidate vertices:"<<can_vertex<<" candidate edges:"<<can_edge<<endl;
}

double Auxiliary::calc_score(Value* query_vec, Value* data_vec, int column_size){
    double result = 0.0;
    double sum = 0.0;
    for(int i=0;i<column_size;++i){
        assert(query_vec[i] <= data_vec[i]);
        result += (data_vec[i]-query_vec[i])*query_vec[i];
        // result += tanh(data_vec[i]-query_vec[i])*query_vec[i];
        sum += query_vec[i];
    }
    return result/sum;
}

void Auxiliary::print(){
    for(int i=0;i<order.size();++i){
        Vertex u = order[i];
        cout<<"======== query id:"<<u<<endl;
        cout<<"candidates:";
        for(auto v : candidates[u].candidate){
            cout<<v<<", ";
        }
        cout<<endl;
        for(Vertex j=i+1;j<order.size();++j){
            Vertex u_n = order[j];
            if(query_graph->adj[u].find(u_n) != query_graph->adj[u].end()){
                cout<<"---------------------"<<endl;
                cout<<"query_neighbor:"<<u_n<<endl;
                for(int x=0;x<candidates[u].candidate.size();++x){
                    cout<<"data candidate:"<<candidates[u].candidate[x]<<endl;
                    for(auto c : candidates[u].neighbor_candidates[u_n][x].first){
                        cout<<c<<", ";
                    }
                    cout<<endl;
                }
            }
        }
    }
}

inline bool Auxiliary::validate_edge_emb(Vertex& u, Vertex& u_, Vertex& v, Vertex& v_){
    Vertex e_q_id = u<u_ ? (*query_edge_id_map)[u][u_] : (*query_edge_id_map)[u_][u];
    Vertex e_d_id = v<v_ ? (*data_edge_id_map)[v][v_] : (*data_edge_id_map)[v_][v];
    
    return vec_validation(query_edge_emb_content[e_q_id], data_edge_emb_content[e_d_id], edge_emb_column_size);
}

inline bool Auxiliary::validate_vertex_emb(Vertex& u, Vertex& v){
    return vec_validation(query_vertex_emb_content[u], data_vertex_emb_content[v], vertex_emb_column_size);
}

void Auxiliary::get_candidate_info(uint32_t& candidate_vertices, uint32_t& candidate_edges){
    candidate_vertices = 0;
    candidate_edges = 0;
    for(Vertex u=0;u<query_graph->label_map.size();++u){
        candidate_vertices += candidates[u].candidate.size();
    }
    for(Vertex i=0;i<order.size();++i){
        Vertex u = order[i];
        for(Vertex j=i+1;j<order.size();++j){
            Vertex u_n = order[j];
            if(query_graph->adj[u].find(u_n) != query_graph->adj[u].end()){
                for(int x=0;x<candidates[u].candidate.size();++x){
                    candidate_edges += candidates[u].neighbor_candidates[u_n][x].first.size();
                }
            }
        }
    }
}

vector<pair<Vertex, uint32_t>> Auxiliary::get_ordered_vertices_by_candidate_count(){
    vector<pair<Vertex, uint32_t> > candidates_size;
    for(Vertex u=0; u<query_vertex_count; ++u){
        // candidates_size.push_back({u, vertex_candidate_info[u]});
        candidates_size.push_back({u, candidates[u].candidate.size()});
    }
    // sort(candidates_size.begin(), candidates_size.end(), cmp_uint);
    // reorder
    for(int i=candidates_size.size();i>0;--i){
        for(int j=0;j<i-1;++j){
            if(candidates_size[j].second>candidates_size[j+1].second){
                swap(candidates_size[j], candidates_size[j+1]);
            }else if(candidates_size[j].second == candidates_size[j+1].second){
                Vertex d1 = candidates_size[j].first;
                Vertex d2 = candidates_size[j+1].first;
                if(query_graph->adj[d1].size() < query_graph->adj[d2].size()){
                    swap(candidates_size[j], candidates_size[j+1]);
                }
            }
        }
    }
    return candidates_size;
}

void Auxiliary::update_pre_suc_neighbors(vector<Vertex>& search_order){
    vector<Vertex> order_offset;
    order_offset.resize(search_order.size());
    for(int i=0; i<search_order.size(); ++i){
        order_offset[search_order[i]] = i;
    }
    predecessor_neighbors.clear();
    successor_neighbors.clear();
    predecessor_neighbors.resize(search_order.size());
    successor_neighbors.resize(search_order.size());
    for(int i=0; i<search_order.size(); ++i){
        Vertex u = search_order[i];
        for(auto n : query_graph->adj[u]){
            if(order_offset[u] < order_offset[n]){
                successor_neighbors[u].push_back(n);
            }else{
                predecessor_neighbors[u].push_back(n);
            }
        }
    }
}

void Auxiliary::reorder_leaf_vertices(){
    // initialize the preprocessors and sucessors
    update_pre_suc_neighbors(order);
    // put the leaf vertices behand
    vector<Vertex> leaves;
    for(Vertex u=0; u<order.size(); ++u){
        if(successor_neighbors[u].empty()){
            leaves.push_back(u);
        }
    }
    // delete and append the leaf vertices in the order
    for(auto l : leaves){
        for(auto it=order.begin(); it!=order.end(); ++it){
            if(*it == l){
                order.erase(it);
                break;
            }
        }
    }
    order.insert(order.end(), leaves.begin(), leaves.end());
    // reupdate the neighbors
    update_pre_suc_neighbors(order);
}

// GraphQL style
void Auxiliary::generate_order_in_GQL(){
    if(!order.empty()){
        return;
    }
    vector<pair<Vertex, uint32_t>> candidates_size = get_ordered_vertices_by_candidate_count();

    unordered_set<Vertex> next_candidates = query_graph->adj[candidates_size[0].first];
    unordered_set<Vertex> added_candidates = {candidates_size[0].first};
    order.push_back(candidates_size[0].first);
    for(uint32_t i=1;i<query_graph->label_map.size();++i){
        for(uint32_t j=0;j<query_graph->label_map.size();++j){
            if(added_candidates.find(candidates_size[j].first) != added_candidates.end()){
                continue;
            }
            if(next_candidates.find(candidates_size[j].first) != next_candidates.end()){
                order.push_back(candidates_size[j].first);
                for(auto u:query_graph->adj[candidates_size[j].first]){
                    next_candidates.insert(u);
                }
                added_candidates.insert(candidates_size[j].first);
                break;
            }
        }
    }
    reorder_leaf_vertices();
}

void Auxiliary::generate_order_by_nucleus_decompostion(){
    if(!order.empty()){
        return;
    }
    std::vector<std::vector<uint32_t>> generated_orders;
    query_plan_generator::generate_query_plan_with_nd(query_graph, vertex_candidate_info, edge_candidate_info, generated_orders);
    // if(generated_orders.size()==0){
    //     cout<<"error: No orderings"<<endl;
    // }else if(generated_orders.size()>1){
    //     cout<<"warning: multiple orderings"<<endl;
    // }
    order = generated_orders[0];
}

// Generate ordering with predecessors
void Auxiliary::generate_order_by_predecessors(){
    if(!order.empty()){
        return;
    }
    vector<pair<Vertex, uint32_t>> candidates_size = get_ordered_vertices_by_candidate_count();
    unordered_set<Vertex> next_candidates = query_graph->adj[candidates_size[0].first];
    unordered_set<Vertex> added_candidates = {candidates_size[0].first};
    vector<uint32_t> predecessor_count = vector<uint32_t>(query_vertex_count, 0);
    order.push_back(candidates_size[0].first);
    for(auto n : query_graph->adj[order[0]]){
        predecessor_count[n] = 1;
    }
    vector<uint32_t> candidate_card = vector<uint32_t>(query_vertex_count, 0);
    for(auto p : candidates_size){
        candidate_card[p.first] = p.second;
    }
    for(uint32_t i=1;i<query_vertex_count;++i){
        Vertex selected_next_vertex;
        uint32_t pred_count, card;
        auto it = next_candidates.begin();
        do{
            if(it == next_candidates.begin()){
                selected_next_vertex = *it;
                pred_count = predecessor_count[*it];
                card = candidate_card[*it];
            }else{
                if(pred_count < predecessor_count[*it]){
                    pred_count = predecessor_count[*it];
                    card = candidate_card[*it];
                    selected_next_vertex = *it;
                }else if(pred_count == predecessor_count[*it] && card > candidate_card[*it]){
                    card = candidate_card[*it];
                    selected_next_vertex = *it;
                }
            }
            it ++;
        }while(it != next_candidates.end());
        order.push_back(selected_next_vertex);
        next_candidates.erase(selected_next_vertex);
        added_candidates.insert(selected_next_vertex);
        for(auto u : query_graph->adj[selected_next_vertex]){
            if(added_candidates.find(u) == added_candidates.end()){
                next_candidates.insert(u);
            }
        }
    }
    reorder_leaf_vertices();
}

bool Auxiliary::cmp_candidate(vector<pair<Vertex, int>>& candidate_size, vector<Vertex>& vec1, vector<Vertex>& vec2){
    if(vec1.size() != vec2.size()){
        return vec1.size() > vec2.size();
    }else{
        uint32_t sum1=0, sum2=0;
        for(auto v : vec1){
            sum1 += candidate_size[v].second;
        }
        for(auto v : vec2){
            sum2 += candidate_size[v].second;
        }
        return sum1 > sum2;
    }
}

Vertex Auxiliary::pick_least_candidate(vector<pair<Vertex, int>>& candidate_size, unordered_set<Vertex>& vec){
    Vertex c = *(vec.begin());
    int s = candidate_size[c].second;
    for(auto can : vec){
        if(candidate_size[can].second < s){
            s = candidate_size[can].second;
            c = can;
        }
    }
    return c;
}

#ifdef FAILING_SET_PRUNING
void Auxiliary::initilize_ancestors(){
    if(order.empty()){
        cout<<"Error order is not initialized!"<<endl;
        return;
    }
    uint32_t query_vertex_num = order.size();
    // initialize the ancestors
    ancestors.resize(query_vertex_num);
    for(uint32_t i=0; i<query_vertex_num; ++i){
        ancestors[i].reset();
    }
    // update the ancestors
    vector<Vertex> order_offset; // vertex_id to offset
    order_offset.resize(query_vertex_num);
    for(uint32_t i=0; i<query_vertex_num; ++i){
        order_offset[order[i]] = i;
    }
    for(uint32_t i=0; i<query_vertex_num; ++i){
        ancestors[order[i]].set(order[i]);
        for(auto n : query_graph->adj[order[i]]){
            if(order_offset[n]>order_offset[order[i]]){
                ancestors[n].set(order[i]);
                ancestors[n] = ancestors[order[i]] | ancestors[n];
            }
        }
    }
}
#endif

// #ifdef CORE_DECOMPOSITION
// void Auxiliary::query_core_decomposition(){
//     Core composition(query_graph);

//     cut_vertices.resize(query_vertex_count);
//     memset(&cut_vertices[0], 0, sizeof(bool)*query_vertex_count);
//     parent_cut_vertices.resize(query_vertex_count);
//     memset(&parent_cut_vertices[0], -1, sizeof(int)*query_vertex_count);

//     for(auto v : composition.cutVertices){
//         cut_vertices[v] = true;
//     }

//     vector<Vertex> order_offset; // vertex_id to offset
//     order_offset.resize(query_vertex_count);
//     for(uint32_t i=0; i<query_vertex_count; ++i){
//         order_offset[order[i]] = i;
//     }

//     for(auto vec : composition.cores){

//     }

//     // print the decomposed query graph
//     // query_graph.print_graph();
//     // cout<<"cores:"<<endl;
//     // for(auto vec : composition.cores){
//     //     cout<<"{";
//     //     for(auto v : vec){
//     //         cout<<v<<", ";
//     //     }
//     //     cout<<"}"<<endl;
//     // }
//     // cout<<"cut vertices:"<<endl;
//     // cout<<"{";
//     // for(auto v : composition.cutVertices){
//     //     cout<<v<<", ";
//     // }
//     // cout<<"}"<<endl;   
// }
// #endif

void Auxiliary::generate_search_order(vector<Vertex>& search_order, vector<Vertex>& search_order_offset){
    generate_order_in();
    search_order = order;
    search_order_offset.resize(order.size());
    for(uint32_t i=0; i<order.size(); ++i){
        search_order_offset[order[i]] = i;
    }
    if(query_edge_emb != NULL && enable_order){
        for(Vertex i=1;i<search_order.size();++i){
            Vertex u = search_order[i];
            Vertex last_parent = *(predecessor_neighbors[search_order[i]].rbegin());
            vector<float>& score_u = candidates[u].candidate_score;
            Vertex q_e_id = last_parent<u ? (*query_edge_id_map)[last_parent][u] : (*query_edge_id_map)[u][last_parent];
            for(uint32_t y=0;y<candidates[last_parent].candidate.size();++y){
                Vertex c = candidates[last_parent].candidate[y];
                vector<float>& edge_score = candidates[last_parent].neighbor_candidates[u][y].second;
                edge_score.reserve(candidates[last_parent].neighbor_candidates[u][y].first.size());
                for(Vertex c_n_offset : candidates[last_parent].neighbor_candidates[u][y].first){
                    Vertex c_n = candidates[u].candidate[c_n_offset];
                    Vertex d_e_id = c_n<c ? (*data_edge_id_map)[c_n][c] : (*data_edge_id_map)[c][c_n];
                    edge_score.push_back(calc_score(query_edge_emb_content[q_e_id], data_edge_emb_content[d_e_id], edge_emb_column_size));
                    // edge_score.push_back(score_u[c_n_offset]+calc_score(query_edge_emb_content[q_e_id], data_edge_emb_content[d_e_id], edge_emb_column_size));
                }
            }
        }
    }
}

void Auxiliary::generate_order_in(){
#if ORDERING==0
    generate_order_in_GQL();
#elif ORDERING==1
    generate_order_by_predecessors();
#elif ORDERING==2
    generate_order_by_nucleus_decompostion();
#endif
#ifdef FAILING_SET_PRUNING
    initilize_ancestors();
#endif
}

Auxiliary::~Auxiliary(){
    delete [] candidates;
    // delete [] flag;
    // for(Vertex u=0;u<query_vertex_count;++u){
    //     delete [] candidate_bit_map[u];
    // }
    // delete [] candidate_bit_map;
}


