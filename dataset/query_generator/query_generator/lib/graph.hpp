#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#include <cstdlib>
#include <ctime>

using namespace std;

class graph_cpp{
public:
    unordered_map<int, unordered_set<int> > adj_list;
    unordered_map<int, int> label_map;
    unordered_map<int, unordered_set<int> > anchor_cycle_edges;

    graph_cpp(unordered_map<int, unordered_set<int> >& adj_list_, unordered_map<int, int>& label_map_){
        copy(adj_list_.begin(), adj_list_.end(), inserter(adj_list, adj_list.begin()));
        copy(label_map_.begin(), label_map_.end(), inserter(label_map, label_map.begin()));
    }
    graph_cpp(const graph_cpp& g){
        copy(g.adj_list.begin(), g.adj_list.end(), inserter(adj_list, adj_list.begin()));
        copy(g.label_map.begin(), g.label_map.end(), inserter(label_map, label_map.begin()));
    }
    graph_cpp(){};

    void clear(){
        label_map.clear();
        adj_list.clear();
    }

    bool has_cycle(int anchor){
        unordered_map<int, int> explored;
        vector<pair<int, int> > frontier = {{anchor, -1}};
        while(frontier.size() > 0){
            auto p = frontier[0];
            frontier.erase(frontier.begin());
            if(p.second == -1){
                explored.insert({p.first, p.second});
                for(auto n : adj_list[p.first]){
                    frontier.push_back({n, n});
                }
            }else{
                explored.insert({p.first, p.second});
                for(auto n : adj_list[p.first]){
                    if(explored.find(n)==explored.end()){
                        frontier.push_back({n, p.second});
                    }
                }
            }
        }
        bool has_c = false;
        for(auto p : adj_list){
            for(auto n : p.second){
                if(p.first == anchor || n == anchor){
                    continue;
                }
                if(explored[p.first] != explored[n]){
                    return true;
                }
            }
        }
        return false;
    }

    void random_select_edge(vector<pair<int, int> >& samples, int sample_size){
        vector<pair<int, int> > edges;
        for(auto v_p : adj_list){
            for(auto n : v_p.second){
                if(v_p.first < n){
                    edges.push_back({v_p.first, n});
                }
            }
        }
        if(sample_size > edges.size()){
            sample_size = edges.size();
        }
        random_shuffle(edges.begin(), edges.end());
        samples.assign(edges.begin(), edges.begin()+sample_size);
    }

    int get_edge_count(){
        int count = 0;
        for(auto p : adj_list){
            count += p.second.size();
        }
        return count/2;
    }

    void test_remove_edge(pair<int, int>& edge, int anchor){
        // avoid removing the circle
        if(anchor_cycle_edges[edge.first].find(edge.second) != anchor_cycle_edges[edge.first].end()){
            return;
        }

        // check the connectivity of the removed graph
        unordered_set<int> explored{edge.first};
        unordered_set<int> frontier(adj_list[edge.first]);
        frontier.erase(edge.second);

        // remove edge
        adj_list[edge.first].erase(edge.second);
        adj_list[edge.second].erase(edge.first);

        // check if there is another path connecting edge.first and edge.second
        bool connected = false;
        while(frontier.size() > 0){
            int vertex = *(frontier.begin());
            frontier.erase(frontier.begin());
            explored.insert(vertex);

            for(auto n : adj_list[vertex]){
                if(n == edge.second){
                    connected = true;
                    goto out_loop;
                }
                if(explored.find(n) == explored.end()){
                    frontier.insert(n);
                }
            }

        }
        out_loop:
        // check whether has cycle
        if(connected == true){
            if(has_cycle(anchor) == false){
                connected = false;
                goto out_loop;
            }
        }

        // recover the graph
        if(connected == false){
            adj_list[edge.first].insert(edge.second);
            adj_list[edge.second].insert(edge.first);
        }
    }

    void get_k_hop_neihgborhood(int anchor, int hops, unordered_set<int>& explored){
        explored.clear();
        vector<pair<int, int>> frontier = {{anchor, 0}};

        while(frontier.size() > 0){
            int new_vertex = (frontier.begin())->first;
            int distance = (frontier.begin())->second;
            frontier.erase(frontier.begin());
            explored.insert(new_vertex);
            if(distance >= hops){
                continue;
            }

            for(auto neighbor : adj_list[new_vertex]){
                if(explored.find(neighbor) == explored.end()){
                    frontier.push_back({neighbor, distance+1});
                }
            }
        }
    }

    void print(){
        for(auto p : adj_list){
            cout<<p.first<<":"<<label_map[p.first]<<"{";
            for(auto n : p.second){
                cout<<n<<", ";
            }
            cout<<"}"<<endl;
        }
    }
};