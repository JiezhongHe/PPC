#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>

// #include <cstdlib>
// #include <ctime>
#include <stdlib.h>
#include <time.h>
#include "graph.hpp"
#include "utils.hpp"

using namespace std;

// extern bool check_nlf(graph_cpp* query, int query_anchor, graph_cpp* data, int data_anchor);
// extern bool check_iso_in_k_hop(graph_cpp* data_graph, int data_anchor, graph_cpp* query_graph, int query_anchor, int hops);
// extern void load_graph_list(string filename, vector<graph_cpp>& graph_list);

// default_random_engine e;

int random_range(int a, int b){
    if(a == b){
        return a;
    }
    // uniform_int_distribution<unsigned> u(a, b);
    // return u(e);
    return a+rand()%(b-a+1);
}

template<typename T>
T random_choice_vector(const vector<T>& vec){
    if(vec.size() == 0){
        cout<<"Error random choice in an empty vector"<<endl;
        exit(0);
    }
    int offset = random_range(0, vec.size()-1);
    auto it = vec.begin();
    advance(it, offset);
    return *it;
}

template<typename T1, typename T2>
T1 random_choice_map(const unordered_map<T1, T2>& vec){
    if(vec.size() == 0){
        cout<<"Error random choice in an empty vector"<<endl;
        exit(0);
    }
    int offset = random_range(0, vec.size()-1);
    auto it = vec.begin();
    advance(it, offset);
    return it->first;
}


class graph_generator{
public:
    // csr format
    vector<graph_cpp> graph_list;
    vector<unordered_map<int, vector<int> > > label_dist;

    // =======================
    graph_generator(string filename){
        srand(unsigned(time(NULL)));
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
                    add_graph(graph);
                    adj_list.clear();
                    label_map.clear();
                }
                start = 1;
            }else if(l == 'v'){
                ss >> ele1 >> ele2;
                cout<<"v"<<ele1<<" "<<ele2<<endl;
                label_map.insert({ele1, ele2});
                adj_list[ele1] = {};
            }else if(l == 'e'){
                ss >> ele1 >> ele2;
                cout<<"e"<<ele1<<" "<<ele2<<endl;
                adj_list[ele1].insert(ele2);
                adj_list[ele2].insert(ele1);
            }
        }
    }

    graph_generator(){srand(unsigned(time(NULL)));}

    void add_graph(const graph_cpp& graph){
        graph_list.push_back(graph);
        add_label_dist(graph);
    }

    void generate_pos_query(int hops, int query_size, graph_cpp& query, bool check_cycle){
        while(true){
            // randomly choose a graph
            int graph_id = random_range(0, graph_list.size()-1);
            // choose an anchor label evenly
            int offset = random_range(0, label_dist[graph_id].size()-1);
            auto it = label_dist[graph_id].begin();
            advance(it, offset);
            int anchor_label = it->first;
            if(it->second.size() < 2){
                continue;
            }

            // randomly generate a query anchor
            int query_anchor = random_choice_vector(it->second);

            // randomly generate a query graph
            vector<int> query_nei;
            // cout<<"start generate queries"<<endl;
            sample_neigh_with_root(graph_id, query_size, hops, query_anchor, query_nei);
            // cout<<"finish generate queries"<<endl;
            if(query_nei.size() == 0){
                continue;
            }
            get_induced_subgraph(graph_id, query_nei, query);
            if(query.adj_list[query_anchor].size()<=2){
                continue;
            }
            // randomly remove some edges
            vector<pair<int, int> > remove_edges;

            if(check_cycle){
                if(!query.has_cycle(query_anchor)){
                    continue;
                }
            }
            
            query.random_select_edge(remove_edges, random_range(0, query.get_edge_count()/2));
            
            for(auto e : remove_edges){
                query.test_remove_edge(e, query_anchor);
            }
            break;
        }
    }


    void generate_neg_query_on_multigraph(int hops, int query_size, graph_cpp& query, int& query_anchor, int& data_id, int& data_anchor, bool check_cycle){
        while(true){
            // randomly choose a graph
            int graph_id = random_range(0, graph_list.size()-1);
            // choose an anchor label evenly
            int offset = random_range(0, label_dist[graph_id].size()-1);
            auto it = label_dist[graph_id].begin();
            advance(it, offset);
            int anchor_label = it->first;
            if(it->second.size() < 2){
                continue;
            }

            // randomly generate a query anchor
            query_anchor = random_choice_vector(it->second);

            // randomly generate a query graph
            vector<int> query_nei;
            // cout<<"start generate queries"<<endl;
            sample_neigh_with_root(graph_id, query_size, hops, query_anchor, query_nei);
            // cout<<"finish generate queries"<<endl;
            if(query_nei.size() == 0){
                continue;
            }
            get_induced_subgraph(graph_id, query_nei, query);
            if(query.adj_list[query_anchor].size()<=2){
                continue;
            }
            // randomly remove some edges
            vector<pair<int, int> > remove_edges;

            if(check_cycle){
                if(!query.has_cycle(query_anchor)){
                    continue;
                }
            }
            
            query.random_select_edge(remove_edges, random_range(0, query.get_edge_count()/2));
            
            for(auto e : remove_edges){
                query.test_remove_edge(e, query_anchor);
            }

            for(int i=0;i<graph_list.size();++i){
                int data_graph_id = random_range(0, graph_list.size()-1);

                unordered_set<int> candidate_anchor;
                auto it_l = label_dist[data_graph_id].begin();
                copy(it_l->second.begin(), it_l->second.end(), inserter(candidate_anchor, candidate_anchor.begin()));
                int count = 0;
                while(candidate_anchor.size() > 0){
                    count ++;
                    int offset = random_range(0, candidate_anchor.size()-1);
                    auto itc = candidate_anchor.begin();
                    advance(itc, offset);
                    data_anchor = *itc;
                    candidate_anchor.erase(data_anchor);
                    
                    if(count > 10){
                        break;
                    }
                    // check by nlf
                    if(check_nlf(&query, query_anchor, &graph_list[data_graph_id], data_anchor)==false){
                        continue;
                    }
                    // check isomorphism
                    if(check_iso_in_k_hop(&graph_list[data_graph_id], data_anchor, &query, query_anchor, hops)){
                        continue;
                    }
                    data_id = data_graph_id;
                    return;
                }
            }
        }
    }

    void generate_neg_query_on_multigraph_edge(int hops, int query_size, graph_cpp& query, int& query_anchor, int& query_anchor_u, int& data_id, int& data_anchor, int& data_anchor_v, bool check_cycle){
        while(true){
            // randomly choose a graph
            int graph_id = random_range(0, graph_list.size()-1);
            // choose an anchor label evenly
            int offset = random_range(0, label_dist[graph_id].size()-1);
            auto it = label_dist[graph_id].begin();
            advance(it, offset);
            int anchor_label = it->first;
            if(it->second.size() < 2){
                continue;
            }

            // randomly generate a query anchor
            query_anchor = random_choice_vector(it->second);

            // randomly generate a query graph
            vector<int> query_nei;
            // cout<<"start generate queries"<<endl;
            sample_neigh_with_root(graph_id, query_size, hops, query_anchor, query_nei);
            // cout<<"finish generate queries"<<endl;
            if(query_nei.size() == 0){
                continue;
            }
            get_induced_subgraph(graph_id, query_nei, query);
            if(query.adj_list[query_anchor].size()<=2){
                continue;
            }
            // randomly remove some edges
            vector<pair<int, int> > remove_edges;

            if(check_cycle){
                if(!query.has_cycle(query_anchor)){
                    continue;
                }
            }
            
            query.random_select_edge(remove_edges, random_range(0, query.get_edge_count()/2));
            
            for(auto e : remove_edges){
                query.test_remove_edge(e, query_anchor);
            }

            for(int i=0;i<graph_list.size();++i){
                int data_graph_id = random_range(0, graph_list.size()-1);

                unordered_set<int> candidate_anchor;
                auto it_l = label_dist[data_graph_id].begin();
                copy(it_l->second.begin(), it_l->second.end(), inserter(candidate_anchor, candidate_anchor.begin()));
                int count = 0;
                while(candidate_anchor.size() > 0){
                    count ++;
                    int offset = random_range(0, candidate_anchor.size()-1);
                    auto itc = candidate_anchor.begin();
                    advance(itc, offset);
                    data_anchor = *itc;
                    candidate_anchor.erase(data_anchor);
                    
                    if(count > 10){
                        break;
                    }
                    // check by nlf
                    if(check_nlf(&query, query_anchor, &graph_list[data_graph_id], data_anchor)==false){
                        continue;
                    }
                    // check isomorphism
                    if(check_iso_in_k_hop(&graph_list[data_graph_id], data_anchor, &query, query_anchor, hops)){
                        continue;
                    }
                    // randomly pick up a neighbor
                    for(auto q_n : query.adj_list[query_anchor]){
                        for(auto d_n : graph_list[data_graph_id].adj_list[data_anchor]){
                            if(query.label_map[q_n] == graph_list[data_graph_id].label_map[d_n] && check_nlf(&query, q_n, &graph_list[data_graph_id], d_n)){
                                query_anchor_u = q_n;
                                data_anchor_v = d_n;
                                data_id = data_graph_id;
                                return;
                            }
                        }
                    }
                    // data_id = data_graph_id;
                    // return;
                }
            }
        }
    }

    void generate_query(int hops, int query_size, graph_cpp& query, int& query_anchor, int& data_id, int& data_anchor, bool check_cycle){
        while(true){
            // randomly choose a graph
            int graph_id = random_range(0, graph_list.size()-1);
            // choose an anchor label evenly
            int offset = random_range(0, label_dist[graph_id].size()-1);
            auto it = label_dist[graph_id].begin();
            advance(it, offset);
            int anchor_label = it->first;
            if(it->second.size() < 2){
                continue;
            }

            // randomly generate a query anchor
            query_anchor = random_choice_vector(it->second);

            // randomly generate a query graph
            vector<int> query_nei;
            // cout<<"start generate queries"<<endl;
            sample_neigh_with_root(graph_id, query_size, hops, query_anchor, query_nei);
            // cout<<"finish generate queries"<<endl;
            if(query_nei.size() == 0){
                continue;
            }
            get_induced_subgraph(graph_id, query_nei, query);
            if(query.adj_list[query_anchor].size()<=2){
                continue;
            }
            // randomly remove some edges
            vector<pair<int, int> > remove_edges;

            if(check_cycle){
                if(!query.has_cycle(query_anchor)){
                    continue;
                }
            }
            
            query.random_select_edge(remove_edges, random_range(0, query.get_edge_count()/2));
            
            for(auto e : remove_edges){
                query.test_remove_edge(e, query_anchor);
            }
        
            unordered_set<int> candidate_anchor;
            copy(it->second.begin(), it->second.end(), inserter(candidate_anchor, candidate_anchor.begin()));
            int count = 0;
            while(candidate_anchor.size() > 0){
                int offset = random_range(0, candidate_anchor.size()-1);
                auto itc = candidate_anchor.begin();
                advance(itc, offset);
                data_anchor = *itc;
                candidate_anchor.erase(data_anchor);
                if(data_anchor != query_anchor){
                    count ++;
                    if(count > 10){
                        break;
                    }
                    // check by nlf
                    if(check_nlf(&query, query_anchor, &graph_list[graph_id], data_anchor)==false){
                        // if(candidate_anchor.size()%1000 == 0){
                        //     cout<<"candidate size:"<<candidate_anchor.size()<<endl;
                        // }
                        continue;
                    }
                    // if(check_cycle){
                    //     if(query.has_cycle(query_anchor) == false){
                    //         continue;
                    //     }
                    // }
                    
                    // check isomorphism
                    if(check_iso_in_k_hop(&graph_list[graph_id], data_anchor, &query, query_anchor, hops)){
                        continue;
                    }
                    data_id = graph_id;
                    return;
                }
            }
        }
    }

    void generate_query_edge(int hops, int query_size, graph_cpp& query, int& query_anchor, int& query_anchor_u, int& data_id, int& data_anchor, int& data_anchor_v, bool check_cycle){
        while(true){
            // randomly choose a graph
            int graph_id = random_range(0, graph_list.size()-1);
            // choose an anchor label evenly
            int offset = random_range(0, label_dist[graph_id].size()-1);
            auto it = label_dist[graph_id].begin();
            advance(it, offset);
            int anchor_label = it->first;
            if(it->second.size() < 2){
                continue;
            }

            // randomly generate a query anchor
            query_anchor = random_choice_vector(it->second);

            // randomly generate a query graph
            vector<int> query_nei;
            // cout<<"start generate queries"<<endl;
            sample_neigh_with_root(graph_id, query_size, hops, query_anchor, query_nei);
            // cout<<"finish generate queries"<<endl;
            if(query_nei.size() == 0){
                continue;
            }
            get_induced_subgraph(graph_id, query_nei, query);
            if(query.adj_list[query_anchor].size()<=2){
                continue;
            }
            // randomly remove some edges
            vector<pair<int, int> > remove_edges;

            if(check_cycle){
                if(!query.has_cycle(query_anchor)){
                    continue;
                }
            }
            
            query.random_select_edge(remove_edges, random_range(0, query.get_edge_count()/2));
            
            for(auto e : remove_edges){
                query.test_remove_edge(e, query_anchor);
            }
        
            unordered_set<int> candidate_anchor;
            copy(it->second.begin(), it->second.end(), inserter(candidate_anchor, candidate_anchor.begin()));
            int count = 0;
            while(candidate_anchor.size() > 0){
                int offset = random_range(0, candidate_anchor.size()-1);
                auto itc = candidate_anchor.begin();
                advance(itc, offset);
                data_anchor = *itc;
                candidate_anchor.erase(data_anchor);
                if(data_anchor != query_anchor){
                    count ++;
                    if(count > 10){
                        break;
                    }
                    // check by nlf
                    if(check_nlf(&query, query_anchor, &graph_list[graph_id], data_anchor)==false){
                        // if(candidate_anchor.size()%1000 == 0){
                        //     cout<<"candidate size:"<<candidate_anchor.size()<<endl;
                        // }
                        continue;
                    }
                    // if(check_cycle){
                    //     if(query.has_cycle(query_anchor) == false){
                    //         continue;
                    //     }
                    // }
                    
                    // check isomorphism
                    if(check_iso_in_k_hop(&graph_list[graph_id], data_anchor, &query, query_anchor, hops)){
                        continue;
                    }
                    // randomly pick up a neighbor
                    for(auto q_n : query.adj_list[query_anchor]){
                        for(auto d_n : graph_list[graph_id].adj_list[data_anchor]){
                            if(query.label_map[q_n] == graph_list[graph_id].label_map[d_n] && check_nlf(&query, q_n, &graph_list[graph_id], d_n)){
                                query_anchor_u = q_n;
                                data_anchor_v = d_n;
                                data_id = graph_id;
                                return;
                            }
                        }
                    }
                    // data_id = graph_id;
                    // return;
                }
            }
        }
    }

    void sample_neigh_with_root(int g_id, int size, int hops, int& anchor, vector<int>& neighbor){
        graph_cpp& graph = graph_list[g_id];
        unordered_map<int, int> frontier = {{anchor, 0}};
        unordered_set<int> explored;
        while(explored.size()<size && frontier.size() > 0){
            int offset = random_range(0, frontier.size()-1);
            auto it = frontier.begin();
            int selected_v = it->first;
            advance(it, offset);
            if(it->second <= hops){
                explored.insert(selected_v);
                for(auto neighbor : graph.adj_list[selected_v]){
                    if(explored.find(neighbor) == explored.end()){
                        auto itf = frontier.find(neighbor);
                        if(itf == frontier.end()){
                            frontier.insert({neighbor, it->second+1});
                        }else if(itf->second > it->second+1){
                            itf->second = it->second+1;
                        }
                    }
                }
            }
            frontier.erase(it);
        }
        if(explored.size() == size){
            neighbor.assign(explored.begin(), explored.end());
        }
    }

    // void sample_neigh_with_root(int g_id, int size, int hops, int& anchor, vector<int>& neighbor){
    //     graph_cpp& graph = graph_list[g_id];
    //     for(int i=0; i<10; ++i){
    //         unordered_set<int> frontier = {anchor};
    //         unordered_set<int> explored;
    //         while(explored.size() < size*10 && frontier.size() > 0){
    //             int offset = random_range(0, frontier.size()-1);
    //             auto it = frontier.begin();
    //             advance(it, offset);
    //             int selected_v = *it;
    //             explored.insert(*it);
    //             frontier.erase(it);
    //             for(auto neighbor : graph.adj_list[selected_v]){
    //                 if(explored.find(neighbor) == explored.end()){
    //                     frontier.insert(neighbor);
    //                 }
    //             }
    //         }
    //         // get the distances
    //         unordered_map<int, int> distance;
    //         vector<pair<int, int> > inner_frontier = {{anchor, 0}};
    //         frontier = {anchor}; // used as another explored set
    //         while(inner_frontier.size()>0){
    //             auto it = inner_frontier.begin();
    //             distance.insert({it->first, it->second});
    //             int selected_v = it->first;
    //             int distance = it->second;
    //             inner_frontier.erase(it);
    //             frontier.insert(selected_v);

    //             for(auto neighbor : graph.adj_list[selected_v]){
    //                 if(frontier.find(neighbor)==frontier.end() && explored.find(neighbor)!=explored.end()){
    //                     inner_frontier.push_back({neighbor, distance+1});
    //                 }
    //             }
    //         }

    //         // randomly pickup vertices with size $size
    //         unordered_set<int> frontier_final = {anchor};
    //         unordered_set<int> to_add;
    //         while(to_add.size() < size && frontier_final.size() > 0){
    //             int offset = random_range(0, frontier_final.size()-1);
    //             auto it = frontier_final.begin();
    //             advance(it, offset);
    //             int selected_v = *it;
    //             if(distance[selected_v] <= hops){
    //                 to_add.insert(selected_v);
    //             }
    //             frontier_final.erase(it);

    //             for(auto neighbor : graph.adj_list[selected_v]){
    //                 if(to_add.find(neighbor) == to_add.end() && explored.find(neighbor)!=explored.end()){
    //                     frontier_final.insert(neighbor);
    //                 }
    //             }
    //         }
    //         if(to_add.size() == size){
    //             neighbor.assign(to_add.begin(), to_add.end());
    //             return;
    //         }
    //     }
    // }

    void get_induced_subgraph(int g_id, vector<int>& vertex_set, graph_cpp& graph){
        graph.clear();

        unordered_set<int> vertex_unordered_set;
        for(auto u : vertex_set){
            vertex_unordered_set.insert(u);
        }

        for(auto v : vertex_set){
            graph.label_map.insert({v, graph_list[g_id].label_map[v]});
            graph.adj_list.insert({v, {}});

            for(auto neighbor : graph_list[g_id].adj_list[v]){
                if(vertex_unordered_set.find(neighbor) != vertex_unordered_set.end()){
                    graph.adj_list[v].insert(neighbor);
                }
            }
        }
    }

private:
    // =======================
    void add_label_dist(const graph_cpp& graph){
        label_dist.push_back({});
        unordered_map<int, vector<int> >& local_label_dist = label_dist[label_dist.size()-1];
        for(auto it : graph.label_map){
            int label = it.second;
            int v = it.first;
            auto itf = local_label_dist.find(label);
            if(itf == local_label_dist.end()){
                local_label_dist.insert({label, {v}});
            }else{
                itf->second.push_back(v);
            }
        }
    }
};


