#pragma once

#include "../configuration/config.h"
#include <immintrin.h>
#include <x86intrin.h>

#define SI 1

typedef unsigned int ui;
typedef ui LabelID;

/*
 * Because the set intersection is designed for computing common neighbors, the target is uieger.
 */

class ComputeSetIntersection {
public:
    static size_t galloping_cnt_;
    static size_t merge_cnt_;

    static void ComputeCandidates(const Vertex* larray, ui l_count, const Vertex* rarray,
                                  ui r_count, Vertex* cn, ui &cn_count);
    static void ComputeCandidates(const Vertex* larray, ui l_count, const Vertex* rarray,
                                  ui r_count, ui &cn_count);

// #if AVX==2
//     static void ComputeCNGallopingAVX512(const Vertex* larray, const ui l_count,
//                                          const Vertex* rarray, const ui r_count, Vertex* cn,
//                                          ui &cn_count);
//     static void ComputeCNGallopingAVX512(const Vertex* larray, const ui l_count,
//                                          const Vertex* rarray, const ui r_count, ui &cn_count);

//     static void ComputeCNMergeBasedAVX512(const Vertex* larray, const ui l_count, const Vertex* rarray,
//                                           const ui r_count, Vertex* cn, ui &cn_count);
//     static void ComputeCNMergeBasedAVX512(const Vertex* larray, const ui l_count, const Vertex* rarray,
//                                           const ui r_count, ui &cn_count);
// #endif

#if AVX > 0
    static void ComputeCNGallopingAVX2(const Vertex* larray, ui l_count,
                                       const Vertex* rarray, ui r_count, Vertex* cn,
                                       ui &cn_count);
    static void ComputeCNGallopingAVX2(const Vertex* larray, ui l_count,
                                       const Vertex* rarray, ui r_count, ui &cn_count);

    static void ComputeCNMergeBasedAVX2(const Vertex* larray, ui l_count, const Vertex* rarray,
                                        ui r_count, Vertex* cn, ui &cn_count);
    static void ComputeCNMergeBasedAVX2(const Vertex* larray, ui l_count, const Vertex* rarray,
                                        ui r_count, ui &cn_count);
    static const ui BinarySearchForGallopingSearchAVX2(const Vertex*  array, ui offset_beg, ui offset_end, ui val);
    static const ui GallopingSearchAVX2(const Vertex*  array, ui offset_beg, ui offset_end, ui val);
#else
    static void ComputeCNNaiveStdMerge(const Vertex* larray, ui l_count, const Vertex* rarray,
                                       ui r_count, Vertex* cn, ui &cn_count);
    static void ComputeCNNaiveStdMerge(const Vertex* larray, ui l_count, const Vertex* rarray,
                                       ui r_count, ui &cn_count);

    static void ComputeCNGalloping(const Vertex * larray, ui l_count, const Vertex * rarray,
                                   ui r_count, Vertex * cn, ui& cn_count);
    static void ComputeCNGalloping(const Vertex * larray, ui l_count, const Vertex * rarray,
                                   ui r_count, ui& cn_count);
    static const ui GallopingSearch(const Vertex *src, ui begin, ui end, ui target);
    static const ui BinarySearch(const Vertex *src, ui begin, ui end, ui target);

// #elif AVX==2
//     static void ComputeCNGallopingAVX512(const Vertex* larray, const ui l_count,
//                                          const Vertex* rarray, const ui r_count, Vertex* cn,
//                                          ui &cn_count);
//     static void ComputeCNGallopingAVX512(const Vertex* larray, const ui l_count,
//                                          const Vertex* rarray, const ui r_count, ui &cn_count);

//     static void ComputeCNMergeBasedAVX512(const Vertex* larray, const ui l_count, const Vertex* rarray,
//                                           const ui r_count, Vertex* cn, ui &cn_count);
//     static void ComputeCNMergeBasedAVX512(const Vertex* larray, const ui l_count, const Vertex* rarray,
//                                           const ui r_count, ui &cn_count);

#endif
};

