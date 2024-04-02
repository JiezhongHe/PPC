#pragma once
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

using namespace std;
// #define ENABLE_TIME_INFO 0
#define PRINT_BUILD_PROGRESS 0

typedef unsigned int Label;
typedef unsigned int Vertex;
#define MAX_VERTEX_ID 0xffffffff // mush be consistent with the type Vertex

// type of the frequencies/supports recorded by PPC-index
typedef uint16_t Value;


// #define CORE_DECOMPOSITION 
// -------------------- index construction ------------------------------------
#define MAX_SAMPLE_NUM 20000

// AVX instructions are used to optimize the index construction/validation as well as the subgraph enumeration algorithm
// 0 stands for no optimization; 1; 2 for AVX512
// note as there are some compatibility issues about intersection using AVX512, hence intersection are implemented in AVX256
#define AVX 0
#define HYBRID 0 // related to the intersection operation using AVX

// comment to disable OVERFLOW_CHECK. When the frequencies/supports are recorded by 16-bit short integers, it is likely to be overflow
// during the counting process. Turn on the OVERFLOW_CHECK mechanism at a cost of index construction speed.
#define OVERFLOW_CHECK 0

// ------------------ subgraph enumeration/retrieval --------------------------
#define MAX_QUERY_SIZE 60

// optimization for subgraph enumeration
#define FAILING_SET_PRUNING 0

// enumeration time limit
#define TIME_LIMIT 300 // enumeration time limit, unit: s

// 0 refers to GQL ordering methods, 1 refers to ordering by number of predecessors, 2 refers to ordering by nucleus-decomposition
#define ORDERING 2
