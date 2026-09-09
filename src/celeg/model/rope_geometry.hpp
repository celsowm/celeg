#pragma once

#include "celeg/model/definition.hpp"

namespace celeg::rope_geometry {

#if defined(__CUDACC__)
#define CELEG_ROPE_GEOMETRY_INLINE __host__ __device__ __forceinline__
#else
#define CELEG_ROPE_GEOMETRY_INLINE inline
#endif

struct PairComponents {
    int first;
    int second;
};

CELEG_ROPE_GEOMETRY_INLINE int rotary_pairs(int rotary_dimension) {
    return rotary_dimension / 2;
}

CELEG_ROPE_GEOMETRY_INLINE PairComponents pair_components(
    int pair, int pair_count, RopePairingKind pairing) {
    if (pairing == RopePairingKind::AdjacentPairs) {
        return {2 * pair, 2 * pair + 1};
    }
    return {pair, pair_count + pair};
}

CELEG_ROPE_GEOMETRY_INLINE int mrope_axis_for_pair(
    int pair, int section0, int section1, bool interleaved) {
    if (interleaved) return pair % 3;
    if (pair < section0) return 0;
    if (pair < section0 + section1) return 1;
    return 2;
}

#undef CELEG_ROPE_GEOMETRY_INLINE

}
