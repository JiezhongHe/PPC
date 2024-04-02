#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <assert.h>
#include <pthread.h>

#include "../configuration/config.h"


using namespace std;


class Graph{
public:
    vector<Label> label_map;
    vector<unordered_set<Vertex>> adj;
    vector<unordered_map<Vertex, Vertex>> nlf_data;

    vector<unordered_map<Vertex, Vertex>> edge_id_map;
    vector<Vertex> estimated_common_neighbor_count;
    vector<Vertex> edge_set;
    Vertex edge_count = 0;

    Vertex** common_edge_neighbor; // e_id -> list of neighbors

    Graph(vector<vector<Vertex> >& adj_, vector<Label>& label_map_);
    Graph(vector<unordered_set<Vertex> >& adj_, vector<Label>& label_map_);
    Graph(string filename);
    Graph();

    Vertex get_edge_count();
    void set_up_edge_id_map();

    void construct_edge_common_neighbor();
    void rebuild(vector<vector<Vertex> >& adj_, vector<Label>& label_map_);

    void print_graph();
    void build_nlf();
    void subiso_candidate_generate(Graph& query, vector<pair<Vertex, vector<Vertex>>>& candidates);

    ~Graph();
};

