#include <metal_stdlib>

using namespace metal;

struct CelegRopePairComponents {
    uint first;
    uint second;
};

inline CelegRopePairComponents celeg_rope_pair_components(
    uint pair, uint pair_count, uint pairing_mode) {
    if (pairing_mode == 1u) {
        return {2u * pair, 2u * pair + 1u};
    }
    return {pair, pair_count + pair};
}

inline uint celeg_mrope_axis_for_pair(
    uint pair, uint section0, uint section1, uint interleaved) {
    if (interleaved != 0u) return pair % 3u;
    if (pair < section0) return 0u;
    if (pair < section0 + section1) return 1u;
    return 2u;
}

kernel void celeg_rope_geometry_semantics_probe(
    device uint* pair_components [[buffer(0)]],
    device uint* axes [[buffer(1)]],
    constant uint& pair [[buffer(2)]],
    constant uint& pair_count [[buffer(3)]],
    constant uint& pairing_mode [[buffer(4)]],
    constant uint& section0 [[buffer(5)]],
    constant uint& section1 [[buffer(6)]],
    constant uint& interleaved [[buffer(7)]]) {
    const CelegRopePairComponents components =
        celeg_rope_pair_components(pair, pair_count, pairing_mode);
    pair_components[0] = components.first;
    pair_components[1] = components.second;
    axes[0] = celeg_mrope_axis_for_pair(pair, section0, section1, interleaved);
}
