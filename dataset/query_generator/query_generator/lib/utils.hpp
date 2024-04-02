#pragma once
#include "graph.hpp"
#include <string.h>
#include <set>

typedef int VertexID;

bool check_nlf(graph_cpp* query, int query_anchor, graph_cpp* data, int data_anchor){
    if(query->label_map[query_anchor] != data->label_map[data_anchor]){
        return false;
    }
    unordered_map<int, int> query_dist;
    unordered_map<int, int> data_dist;
    for(auto neighbor : query->adj_list[query_anchor]){
        int label = query->label_map[neighbor];
        auto itf = query_dist.find(label);
        if(itf == query_dist.end()){
            query_dist.insert({label, 1});
        }else{
            itf->second ++;
        }
    }
    for(auto neighbor : data->adj_list[data_anchor]){
        int label = data->label_map[neighbor];
        auto itf = data_dist.find(label);
        if(itf == data_dist.end()){
            data_dist.insert({label, 1});
        }else{
            itf->second ++;
        }
    }
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

class auxiliary_entry{
public:
    VertexID** offset_start;
    VertexID* offset_size;
    VertexID* content;
    int size;
    VertexID qid;
    int max_id;
    auxiliary_entry(int u_neighbor_size, int v_neighbor_size, int max_neighbor_id){
        max_id = max_neighbor_id;
        offset_start = new VertexID* [max_neighbor_id+1];
        offset_size = new VertexID [max_neighbor_id+1];
        memset(offset_start, 0, sizeof(VertexID*)*(max_neighbor_id+1));
        memset(offset_size, 0, sizeof(VertexID)*(max_neighbor_id+1));
        content = new VertexID [v_neighbor_size*u_neighbor_size];
        size = 0;
    }

    void set_u_id(VertexID u_id){
        offset_start[u_id] = content+size;
        offset_size[u_id] = 0;
        qid = u_id;
    }

    void push_back(VertexID did){
        content[size] = did;
        size++;
        offset_size[qid]++;
    }

    void pop_back(VertexID qid, VertexID did){
        VertexID* start = offset_start[qid];
        for(int i=0;i<offset_size[qid]; ++i){
            if(start[i] == did){
                memcpy(start+i, start+i+1, sizeof(VertexID)*(offset_size[qid]-i-1));
                offset_size[qid] --;
                break;
            }
        }
    }

    VertexID* get_candidate(VertexID id, int& size){
        size = offset_size[id];
        return offset_start[id];
    }

    inline int top(){
        return offset_size[qid];
    }

    inline int top(VertexID id){
        return offset_size[id];
    }

    void print(){
        for(int i=0;i<max_id+1;i++){
            if(offset_start[i]!=NULL){
                cout<<i<<": ";
                for(int j=0;j<offset_size[i];j++){
                    cout<<*(offset_start[i]+j)<<" ";
                }
                cout<<endl;
            }
        }
    }

    ~auxiliary_entry(){
        delete [] offset_start;
        delete [] offset_size;
        delete [] content;
    }
};

class candidate_auxiliary{
private:
    graph_cpp* data_graph;
    graph_cpp* query_graph;
    vector<VertexID>* next;
    vector<VertexID>* pre;
    vector<VertexID> order;

    int max_query_id;
    int max_data_id;
    unordered_set<VertexID>* mask;
public:
    // unordered_map<VertexID, auxiliary_entry*>* content; // qid
    auxiliary_entry*** content; // qid + did

    void build_query_order(int query_anchor){
        // generate search order
        unordered_map<int, int> label_dist;
        for(auto it=query_graph->label_map.begin(); it!=query_graph->label_map.end(); ++it){
            int label = it->second;
            int v = it->first;
            auto itf = label_dist.find(label);
            if(itf == label_dist.end()){
                label_dist.insert({label, 1});
            }else{
                itf->second ++;
            }
        }
        order.push_back(query_anchor);
        unordered_map<VertexID, int> connected;
        for(auto neighbor : query_graph->adj_list[query_anchor]){
            connected.insert({neighbor, 1});
        }
        while(order.size()<query_graph->label_map.size()){
            // get the candidate with the max weight
            float max_weight = -1;
            VertexID candidate = -1;
            for(auto p : connected){
                float weight = p.second/label_dist[query_graph->label_map[p.first]];
                if(weight > max_weight){
                    max_weight = weight;
                    candidate = p.first;
                }
            }
            order.push_back(candidate);
            // update
            connected.erase(candidate);
            for(auto neighbor : query_graph->adj_list[candidate]){
                if(find(order.begin(), order.end(), neighbor) != order.end()){
                    continue;
                }
                int count = 0;
                for(auto c_n : query_graph->adj_list[neighbor]){
                    if(find(order.begin(), order.end(), c_n) != order.end()){
                        count ++;
                    }
                }
                connected[neighbor] = count;
            }
        }


    }

    candidate_auxiliary(graph_cpp* data_graph, graph_cpp* query_graph, int query_anchor, int data_anchor, unordered_set<VertexID>* mask = NULL){
        this->data_graph = data_graph;
        this->query_graph = query_graph;
        this->mask = mask;
        
        max_query_id = 0;
        for(auto it:query_graph->adj_list){
            if(max_query_id < it.first){
                max_query_id = it.first;
            }
        }
        if(mask == NULL){
            max_data_id = data_graph->label_map.size()-1;
        }else{
            max_data_id = 0;
            for(auto it=mask->begin(); it!=mask->end(); ++it){
                if(*it > max_data_id){
                    max_data_id = *it;
                }
            }
        }
        // initialize the next and pre
        next = new vector<VertexID> [max_query_id+1];
        pre = new vector<VertexID> [max_query_id+1];
        set<VertexID> explored;
        cout<<"start ordering"<<endl;
        build_query_order(query_anchor);
        cout<<"finish ordering"<<endl;
        for(auto u_id : order){
            for(auto neighbor : query_graph->adj_list[u_id]){
                if(explored.find(neighbor) == explored.end()){
                    next[u_id].push_back(neighbor);
                }else{
                    pre[u_id].push_back(neighbor);
                }
            }
            explored.insert(u_id);
        }
        cout<<"explored"<<endl;
        // initialize the content
        // content = new unordered_map<VertexID, auxiliary_entry*> [max_id+1];
        content = new auxiliary_entry** [max_query_id+1];
        memset(content, NULL, sizeof(auxiliary_entry**)*(max_query_id+1));

        construct();
    }

    void construct(){
        LDF_NLF();
        int card = -1, current_card = 0;
        while(card != current_card){
            cout<<"start backward_prune"<<endl;
            backward_prune();
            cout<<"start forward_prune"<<endl;
            forward_prune();
            card = current_card;
            cout<<"finish prune"<<endl;
            print();
            current_card = get_cardinality();
        }
    }

    void LDF_NLF(){
        VertexID* max_nid = new VertexID [max_query_id + 1];
        for(auto u : order){
            max_nid[u] = 0;
            for(auto u_n : query_graph->adj_list[u]){
                if(u_n > max_nid[u]){
                    max_nid[u] = u_n;
                }
            }
        }
        for(auto u:order){
            content[u] = new auxiliary_entry* [max_data_id+1];
            memset(content[u], NULL, sizeof(auxiliary_entry*)*(max_data_id+1));
            for(auto p : data_graph->adj_list){
                if(mask != NULL){
                    if(mask->find(p.first) == mask->end()){
                        continue;
                    }
                }
                VertexID v = p.first;
                if(check_nlf(query_graph, u, data_graph, v)){
                    auxiliary_entry* entry = new auxiliary_entry(query_graph->adj_list[u].size(), data_graph->adj_list[v].size(), max_nid[u]);
                    content[u][v] = entry;
                }
            }
        }

        for(auto u : order){
            vector<VertexID> to_delete;
            for(int candidate=0; candidate<=max_data_id; ++candidate){
                auxiliary_entry* entry = content[u][candidate];
                if(entry == NULL){
                    continue;
                }
                for(auto u_neighbor : query_graph->adj_list[u]){
                    entry->set_u_id(u_neighbor);
                    for(auto v_neighbor : data_graph->adj_list[candidate]){
                        if(content[u_neighbor][v_neighbor] != NULL){
                            entry->push_back(v_neighbor);
                        }
                    }
                }
            }
        }
        delete [] max_nid;
    }

    void backward_prune(){
        for(auto u : order){
            vector<VertexID> to_delete;
            for(VertexID candidate=0; candidate<=max_data_id; ++candidate){
                auxiliary_entry* entry = content[u][candidate];
                if(entry != NULL){
                    for(auto u_next : next[u]){
                        if(entry->top(u_next) == 0){
                            to_delete.push_back(candidate);
                            break;
                        }
                    }
                }
            }
            for(auto d:to_delete){
                auxiliary_entry* entry = content[u][d];
                for(auto u_neighbor : query_graph->adj_list[u]){
                    int candidate_size;
                    VertexID* start = entry->get_candidate(u_neighbor, candidate_size);
                    for(int i=0;i<candidate_size;++i){
                        auxiliary_entry* entry_inner = content[u_neighbor][start[i]];
                        if(entry_inner == NULL){
                            continue;
                        }
                        entry_inner->pop_back(u, d);
                    }
                }
                delete content[u][d];
                content[u][d] = NULL;
            }
        }
    }

    int get_cardinality(){
        int result = 0;
        for(auto q:order){
            for(int d=0;d<max_data_id;++d){
                if(get_entry(q, d) != NULL){
                    ++ result;
                }
            }
        }
        return result;
    }

    void forward_prune(){
        for(auto u_it=order.rbegin(); u_it!=order.rend(); u_it++){
            VertexID u = *u_it;
            vector<VertexID> to_delete;

            for(VertexID candidate = 0; candidate<=max_data_id; ++candidate){
                // cout<<candidate.first<<" "<<u->id<<endl;
                auxiliary_entry* entry = content[u][candidate];
                if(entry != NULL){
                    for(auto u_pre : pre[u]){
                        if(entry->top(u_pre) == 0){
                            to_delete.push_back(candidate);
                            break;
                        }
                    }
                }
                
            }
            for(auto d : to_delete){
                auxiliary_entry* entry = content[u][d];
                for(auto u_neighbor : query_graph->adj_list[u]){
                    int candidate_size;
                    VertexID* start = entry->get_candidate(u_neighbor, candidate_size);
                    for(int i=0;i<candidate_size;i++){
                        auxiliary_entry* entry_inner = content[u_neighbor][start[i]];
                        if(entry_inner == NULL){
                            continue;
                        }
                        entry_inner->pop_back(u, d);
                    }
                }
                delete content[u][d];
                content[u][d] = NULL;
            }
        }
    }

    inline auxiliary_entry* get_entry(VertexID qid, VertexID did){
        return content[qid][did];
    }

    void print(){
        for(auto u : order){
            cout<<"=======> candidate of query vertex u:"<<u<<endl;
            for(VertexID v=0;v<=max_data_id; ++v){
                if(content[u][v] != NULL){
                    cout<<"data id:"<<v<<endl;
                    content[u][v]->print();
                }
            }
        }
    }

    ~candidate_auxiliary(){
        for(auto u : order){
            for(VertexID v=0; v<=max_data_id; v++){
                if(content[u][v] != NULL){
                    delete content[u][v];
                }
            }
            delete [] content[u];
        }
        delete [] content;
        delete [] next;
        delete [] pre;
    };
};

class subgraph_matcher{
public:
    graph_cpp* data_graph;
    graph_cpp* query_graph;
    unordered_map<int, unordered_set<int> > candidates;
    subgraph_matcher(graph_cpp* data_graph_, graph_cpp* query_graph_){
        data_graph = data_graph_;
        query_graph = query_graph_;
    }

    void filtering(unordered_set<int>* mask){
        // init the candidates
        if(mask == NULL){
            for(auto p : data_graph->adj_list){
                unordered_set<int> can;
                for(auto q:query_graph->adj_list){
                    if(query_graph->label_map[q.first] == data_graph->label_map[p.first]){
                        can.insert(q.first);
                    }
                }
                candidates.insert({p.first, can});
            }
        }else{
            for(auto p_it=mask->begin(); p_it!=mask->end(); p_it++){
                unordered_set<int> can;
                int vertex = *p_it;
                for(auto q:query_graph->adj_list){
                    if(query_graph->label_map[q.first] == data_graph->label_map[vertex]){
                        can.insert(q.first);
                    }
                }
                candidates.insert({vertex, can});
            }
        }
        // start filterint iteratively
        bool filtered = true;
        while(filtered){
            filtered = false;
            vector<int> to_delete;
            for(auto c_p = candidates.begin(); c_p!=candidates.end() ; c_p++){
                // check whether candidate list c.second can match with c.first
                int data_vertex = c_p->first;
                unordered_set<int> neighbor_candidates;
                for(auto n : data_graph->adj_list[data_vertex]){
                    for(auto n_c : candidates[n]){
                        neighbor_candidates.insert(n_c);
                    }
                }
                // check whether match
                vector<int> inner_to_delete;
                for(auto can : c_p->second){
                    for(auto n_can : query_graph->adj_list[can]){
                        if(neighbor_candidates.find(n_can) == neighbor_candidates.end()){
                            filtered = true;
                            inner_to_delete.push_back(can);
                            break;
                        }
                    }
                }
                for(auto d : inner_to_delete){
                    c_p->second.erase(d);
                }
                if(c_p->second.size() == 0){
                    to_delete.push_back(data_vertex);
                }
            }
            for(auto d : to_delete){
                candidates.erase(d);
            }
        }
    }

    bool test_match_after_filtering(int query_anchor, int data_anchor){
        if(candidates[data_anchor].find(query_anchor) == candidates[data_anchor].end()){
            return false;
        }
        return true;
    }

    void print_candidates(){
        for(auto p : candidates){
            cout<<p.first<<":{";
            for(auto c : p.second){
                cout<<c<<", ";
            }
            cout<<"}"<<endl;
        }
    }

};

bool check_iso_in_k_hop(graph_cpp* data_graph, int data_anchor, graph_cpp* query_graph, int query_anchor, int hops){
    // unordered_set<VertexID> mask;
    // data_graph->get_k_hop_neihgborhood(data_anchor, hops, mask);
    // candidate_auxiliary auxiliary(data_graph, query_graph, query_anchor, data_anchor,  &mask);
    // auxiliary_entry* entry = auxiliary.get_entry(query_anchor, data_anchor);
    // if(entry == NULL){
    //     return false;
    // }
    // return true;

    // unordered_set<VertexID> mask;
    // cout<<"sample k-hop"<<endl;
    // data_graph->get_k_hop_neihgborhood(data_anchor, hops, mask);
    // cout<<"finish k-hop"<<endl;
    // subgraph_matcher sm(data_graph, query_graph);
    // cout<<"finish filtering"<<endl;
    // sm.filtering(&mask);
    // return sm.test_match_after_filtering(query_anchor, data_anchor);

    subgraph_matcher sm(data_graph, query_graph);
    sm.filtering(NULL);
    return sm.test_match_after_filtering(query_anchor, data_anchor);
}

void load_graph_list(string filename, vector<graph_cpp>& graph_list){
    ifstream in(filename);
    string line;

    int start = 0;
    unordered_map<int, unordered_set<int> > adj_list;
    unordered_map<int, int> label_map;
    while(getline(in, line)){
        char l;
        int ele1, ele2;
        stringstream ss;
        ss<<line;
        ss>>l;
        if(l == '#'){
            ss>>ele1;
            if(start != 0){
                graph_cpp graph(adj_list, label_map);
                graph_list.push_back(graph);
                adj_list.clear();
                label_map.clear();
            }
            start = 1;
        }else if(l == 'v'){
            ss >> ele1 >> ele2;
            label_map.insert({ele1, ele2});
            adj_list[ele1] = {};
            adj_list[ele2] = {};
        }else if(l == 'e'){
            ss >> ele1 >> ele2;
            adj_list[ele1].insert(ele2);
            adj_list[ele2].insert(ele1);
        }
    }
}



