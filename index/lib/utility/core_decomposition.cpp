#include "core_decomposition.h"

Core::Core(Graph* g){
    graph = g;
    splitGraph();
}

void Core::splitGraph(){
    tarjan_explored_vertices.clear();
    Vertex vertex_count = graph->label_map.size();
    visited = new bool [vertex_count];
    dfn = new int [vertex_count];
    low = new int [vertex_count];
    parent = new Vertex [vertex_count];
    memset(visited, false, sizeof(bool)*(vertex_count));
    memset(parent, -1, sizeof(Vertex)*(vertex_count));
    depth = 0;
    cutVertices.clear();
    // pick up a start vertex
    tarjanDFS(0);
    if(tarjan_explored_vertices.size()>0){
        cores.push_back(tarjan_explored_vertices);
    }
    reversed_index.resize(vertex_count);
    reversed_index_for_cut_vertex.resize(vertex_count);
    for(int i=0; i<cores.size(); ++i){
        for(auto v : cores[i]){
            reversed_index[v] = i;
            if(cutVertices.find(v) != cutVertices.end()){
                reversed_index_for_cut_vertex[v].push_back(i);
            }
        }
    }
    for(int i=0; i<cores.size(); ++i){
        adj[i] = {};
    }
    for(int i=0; i<cores.size(); ++i){
        for(int j=i+1; j<cores.size(); ++j){
            for(auto v1 : cores[i]){
                for(auto v2 : cores[j]){
                    if(v1 == v2){
                        adj[i][j] = v1;
                        adj[j][i] = v1;
                        // adj[i].push_back({j, v1});
                        // adj[j].push_back({i, v1});
                        goto next_itr;
                    }
                }
            }
            next_itr:
            i=i;
        }
    }
    delete [] visited;
    delete [] dfn;
    delete [] low;
    delete [] parent;
}

void Core::tarjanDFS(Vertex u){
    dfn[u] = low[u] = ++ depth;
    visited[u] = true;
    tarjan_explored_vertices.push_back(u);
    // num of the sub-tree
    int children = 0;
    for (auto v : graph->adj[u]){
        // a tree edge
        if (!visited[v])
        {
            children++;
            parent[v] = u;
            // invoke recursively
            int explored_vertices_size = tarjan_explored_vertices.size();
            tarjanDFS(v);
            low[u] = min(low[u], low[v]);
            // root vertex
            if (parent[u] == -1 && children >= 2){
                cutVertices.insert(u);
                vector<Vertex> core;
                bool has_u = false;
                for(int j=tarjan_explored_vertices.size()-1; j>=explored_vertices_size; --j){
                    core.push_back(tarjan_explored_vertices[j]);
                    tarjan_explored_vertices.pop_back();
                    if(tarjan_explored_vertices[j] == u){
                        has_u = true;
                    }
                }
                if(has_u == false){
                    core.push_back(u);
                }
                cores.push_back(core);
            }
            // other vertices
            else if (parent[u] != -1 && low[v] >= dfn[u]){
                cutVertices.insert(u);
                vector<Vertex> core;
                bool has_u = false;
                for(int j=tarjan_explored_vertices.size()-1; j>=explored_vertices_size; --j){
                    core.push_back(tarjan_explored_vertices[j]);
                    tarjan_explored_vertices.pop_back();
                    if(tarjan_explored_vertices[j] == u){
                        has_u = true;
                    }
                }
                if(has_u == false){
                    core.push_back(u);
                }
                cores.push_back(core);
            }
        }
        // update the low[u]
        else if (v != parent[u])
            low[u] = min(low[u], dfn[v]);
    }
}