#pragma once
#include "string.h"
#include "../configuration/config.h"
#include "../graph/graph.h"

class Core{
public:
    Graph* graph;
    vector<vector<Vertex>> cores;
    vector<Vertex> reversed_index;
    vector<vector<Vertex>> reversed_index_for_cut_vertex;
    unordered_set<Vertex> cutVertices;
    unordered_map<Vertex, unordered_map<Vertex, Vertex>> adj; // core_id -> neighboring cores (core_id, connected cut vertex)

    Core(Graph* g);

private:
    // Tarjan algorithm in order to calculate the cut vertices
    bool* visited;
    int depth;
    int* dfn, *low;
    Vertex* parent;
    vector<Vertex> tarjan_explored_vertices;

    void splitGraph();

    void tarjanDFS(Vertex u);
};
