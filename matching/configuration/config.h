#pragma once
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <bitset>

using namespace std;
// #define ENABLE_TIME_INFO 0

#define TIME_LIMIT 300 // enumeration time limit, unit: s

typedef unsigned int Label;
typedef uint16_t Value;
typedef unsigned int Vertex;
typedef uint64_t EdgeHash;
typedef uint8_t QUERY_SIZE_TYPE;


#define MAX_VERTEX_ID 0xffffffff // mush be consistent with the type Vertex

#define MAX_QUERY_SIZE 60

#define ENABLE_PRE_FILTERING 1

#define COMPACT 1

#define INDEX_ORDER 1


#define PRINT_RESULT 0

#define PRINT_MEM_INFO 0

#define PRINT_LEAF_STATE 0

struct edge {
    uint32_t vertices_[2];
};

class TreeNode {
public:
    Vertex id_;
    Vertex parent_;
    uint32_t level_;
    uint32_t under_level_count_;
    uint32_t children_count_;
    uint32_t bn_count_;
    uint32_t fn_count_;
    Vertex* under_level_;
    Vertex* children_;
    Vertex* bn_;
    Vertex* fn_;
    size_t estimated_embeddings_num_;
public:
    TreeNode() {
        id_ = 0;
        under_level_ = NULL;
        bn_ = NULL;
        fn_ = NULL;
        children_ = NULL;
        parent_ = 0;
        level_ = 0;
        under_level_count_ = 0;
        children_count_ = 0;
        bn_count_ = 0;
        fn_count_ = 0;
        estimated_embeddings_num_ = 0;
    }

    ~TreeNode() {
        delete[] under_level_;
        delete[] bn_;
        delete[] fn_;
        delete[] children_;
    }

    void initialize(const uint32_t size) {
        under_level_ = new Vertex[size];
        bn_ = new Vertex[size];
        fn_ = new Vertex[size];
        children_ = new Vertex[size];
    }
};

class Edges {
public:
    uint32_t* offset_;
    uint32_t* edge_;
    uint32_t vertex_count_;
    uint32_t edge_count_;
    uint32_t max_degree_;
public:
    Edges() {
        offset_ = NULL;
        edge_ = NULL;
        vertex_count_ = 0;
        edge_count_ = 0;
        max_degree_ = 0;
    }

    ~Edges() {
        delete[] offset_;
        delete[] edge_;
    }
};

#define NANOSECTOSEC(elapsed_time) ((elapsed_time)/(double)1000000000)

#define OPTIMIZED_LABELED_GRAPH 1

#define ON_FILTERING 0

#define GNN_PRUNING_MARGIN 0
#if GNN_PRUNING_MARGIN==1
#define ENABLE_PRE_FILTERING 0
#define INDEX_ORDER 0
#endif