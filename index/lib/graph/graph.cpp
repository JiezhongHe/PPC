#include "graph.h"

Graph::Graph(vector<vector<Vertex> >& adj_, vector<Label>& label_map_){
    for(auto it=adj_.begin(); it!=adj_.end(); it++){
        adj.push_back({});
        unordered_set<Vertex>& tail = adj.back();
        copy(it->begin(), it->end(), inserter(tail, tail.begin()));
    }
    common_edge_neighbor = NULL;
    copy(label_map_.begin(), label_map_.end(), inserter(label_map, label_map.begin()));
}

Graph::Graph(vector<unordered_set<Vertex> >& adj_, vector<Label>& label_map_){
    for(auto it=adj_.begin(); it!=adj_.end(); it++){
        adj.push_back({});
        unordered_set<Vertex>& tail = adj.back();
        copy(it->begin(), it->end(), inserter(tail, tail.begin()));
    }
    common_edge_neighbor = NULL;
    copy(label_map_.begin(), label_map_.end(), inserter(label_map, label_map.begin()));
}

Graph::Graph(string filename){
    common_edge_neighbor = NULL;
    string line;
    ifstream in(filename);
    unordered_map<Vertex, Label> label_map_unordered;
    unordered_map<Vertex, unordered_set<Vertex>> adj_unordered;
    while(getline(in, line)){
        if(line.c_str()[0] == 'v'){
            stringstream ss(line);
            Vertex v, n;
            char c;
            ss>>c>>v>>n;
            label_map_unordered[v] = n;
        }else if(line.c_str()[0] == 'e'){
            stringstream ss(line);
            Vertex v, n;
            char c;
            ss>>c>>v>>n;
            if(adj_unordered.find(v) == adj_unordered.end()){
                adj_unordered.insert({v, {n}});
            }else{
                adj_unordered[v].insert(n);
            }
            if(adj_unordered.find(n) == adj_unordered.end()){
                adj_unordered.insert({n, {v}});
            }else{
                adj_unordered[n].insert(v);
            }
        }
    }
    label_map.resize(label_map_unordered.size());
    for(auto v : label_map_unordered){
        label_map[v.first] = v.second;
    }
    adj.resize(label_map_unordered.size());
    for(auto v : adj_unordered){
        adj[v.first] = v.second;
    }
}

Graph::Graph(){common_edge_neighbor = NULL;};

Vertex Graph::get_edge_count(){
    if(edge_count == 0){
        for(int i=0;i<adj.size();++i){
            edge_count += adj[i].size();
        }
        edge_count = edge_count/2;
    }
    return edge_count;
}

void Graph::set_up_edge_id_map(){
    if(!edge_id_map.empty()){
        return;
    }
    Vertex edge_count = get_edge_count();
    edge_id_map.resize(adj.size());
    edge_set.resize(edge_count*2);
    estimated_common_neighbor_count.resize(edge_count);
    Vertex e_id = 0;
    for(Vertex v=0; v<adj.size(); ++v){
        vector<Vertex> neighbors;
        neighbors.assign(adj[v].begin(), adj[v].end());
        sort(neighbors.begin(), neighbors.end());
        for(auto e : neighbors){
            if(v < e){
                edge_id_map[v].insert({e, e_id});
                edge_set[e_id*2] = v;
                edge_set[e_id*2+1] = e;
                estimated_common_neighbor_count[e_id] = (adj[v].size() < adj[e].size()) ? adj[v].size() : adj[e].size();
                e_id ++;
            }
        }
    }
}

void Graph::construct_edge_common_neighbor(){
    set_up_edge_id_map();
    if(common_edge_neighbor != NULL){
        return;
    }
    common_edge_neighbor = new Vertex* [get_edge_count()];
    for(Vertex v=0; v<adj.size(); ++v){
        for(auto n : adj[v]){
            if(v < n){
                Vertex e_id = edge_id_map[v][n];
                common_edge_neighbor[e_id] = new Vertex [estimated_common_neighbor_count[e_id]*3+1];
                Vertex* vec = common_edge_neighbor[e_id];
                int offset = 1;
                int count = 0;
                for(auto v_n : adj[n]){
                    if(v_n !=v && adj[v].find(v_n) != adj[v].end()){
                        vec[offset++] = v_n;
                        count++;
                        // first push_back the e_id connecting source vertex
                        if(v_n<v){
                            // if(edge_id_map[v_n].find(v)==edge_id_map[v_n].end()){
                            //     cout<<"Error1:"<<v_n<<" is not connected to "<<v<<endl;
                            // }
                            vec[offset++] = edge_id_map[v_n][v];
                            // vec.push_back(edge_id_map[v_n][v]);
                        }else{
                            // if(edge_id_map[v].find(v_n)==edge_id_map[v].end()){
                            //     cout<<"Error2:"<<v<<" is not connected to "<<v_n<<endl;
                            // }
                            vec[offset++] = edge_id_map[v][v_n];
                            // vec.push_back(edge_id_map[v][v_n]);
                        }
                        // second push_back the e_id connecting target vertex
                        if(v_n<n){
                            // if(edge_id_map[v_n].find(n)==edge_id_map[v_n].end()){
                            //     cout<<"Error3:"<<v_n<<" is not connected to "<<n<<endl;
                            // }
                            vec[offset++] = edge_id_map[v_n][n];
                            // vec.push_back(edge_id_map[v_n][n]);
                        }else{
                            // if(edge_id_map[n].find(v_n)==edge_id_map[n].end()){
                            //     cout<<"Error4:"<<n<<" is not connected to "<<v_n<<endl;
                            // }
                            vec[offset++] = edge_id_map[n][v_n];
                            // vec.push_back(edge_id_map[n][v_n]);
                        }
                    }
                }
                vec[0] = count;
            }
        }
    }
}

void Graph::rebuild(vector<vector<Vertex> >& adj_, vector<Label>& label_map_){
    adj.clear();
    label_map.clear();
    nlf_data.clear();
    for(auto it=adj_.begin(); it!=adj_.end(); it++){
        adj.push_back({});
        unordered_set<Vertex>& tail = adj.back();
        copy(it->begin(), it->end(), inserter(tail, tail.begin()));
    }
    copy(label_map_.begin(), label_map_.end(), inserter(label_map, label_map.begin()));
}

void Graph::print_graph(){
    cout<<"vertex_count:"<<label_map.size()<<endl;
    cout<<"edge_count:"<<get_edge_count()<<endl;
    for(Vertex u=0;u<label_map.size();++u){
        cout<<u<<"-"<<label_map[u]<<":{";
        for(auto n : adj[u]){
            cout<<n<<", ";
        }
        cout<<"}"<<endl;
    }
}

void Graph::build_nlf(){
    if(nlf_data.size() == label_map.size()){
        return;
    }
    nlf_data.resize(label_map.size());
    for(int v=0;v<label_map.size();++v){
        for(auto n:adj[v]){
            Vertex label = label_map[n];
            auto itf = nlf_data[v].find(label);
            if(itf == nlf_data[v].end()){
                nlf_data[v].insert({label, 1});
            }else{
                itf->second ++;
            }
        }
    }
}

void Graph::subiso_candidate_generate(Graph& query, vector<pair<Vertex, vector<Vertex>>>& candidates){
    // build up the nlf
    candidates.clear();
    candidates.resize(query.label_map.size());
    for(int u=0; u<query.label_map.size(); ++u){
        candidates[u].first = u;
        candidates[u].second = {};
        for(int v=0;v<label_map.size(); ++v){
            if(query.label_map[u] == label_map[v]){
                bool can_match = true;
                for(auto l : query.nlf_data[u]){
                    auto itf = nlf_data[v].find(l.first);
                    if(itf == nlf_data[v].end() || itf->second < l.second){
                        can_match = false;
                        break;
                    }else if(itf->second < l.second){
                        can_match = false;
                        break;
                    }
                }
                if(can_match){
                    candidates[u].second.push_back(v);
                }
            }
        }
    }
    // rank the candidates
    for(int i=candidates.size();i>0;--i){
        for(int j=0;j<i-1;++j){
            if(candidates[j].second.size()>candidates[j+1].second.size()){
                swap(candidates[j], candidates[j+1]);
            }
        }
    }
}

Graph::~Graph(){
    if(common_edge_neighbor!=NULL){
        for(int i=0;i<get_edge_count();++i){
            delete [] common_edge_neighbor[i];
        }
        delete [] common_edge_neighbor;
    }
}

