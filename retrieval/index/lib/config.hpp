#pragma once

// #define ENABLE_TIME_INFO 0

#define TIME_LIMIT 300 // enumeration time limit, unit: s

typedef unsigned int Label;
typedef uint16_t Value;

typedef unsigned int Vertex;
#define MAX_VERTEX_ID 0xffffffff // mush be consistent with the type Vertex

inline void sum_safe_without_overflow(Value& a, Value& b){
    Value tmp = a+b;
    if(tmp<a || tmp<b){
        a = 0xffffffff;
    }else{
        a = tmp;
    }
}

#define OVERFLOW_CHECK 0

#define FAILING_SET_PRUNING 0
