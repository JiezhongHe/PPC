#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include <cstdlib>
#include <ctime>


#include "../graph/graph.h"
#include "../utility/embedding.h"

class feature_counter{
public:
    virtual void count_for_edges(Graph& graph, Tensor**& result, int& result_size) = 0;
    virtual void count_for_vertices(Graph& graph, Tensor**& result, int& result_size) = 0;
    virtual vector<vector<Label>> get_features() = 0;
    virtual int get_feature_type() = 0;
    virtual bool get_residual() = 0;
};