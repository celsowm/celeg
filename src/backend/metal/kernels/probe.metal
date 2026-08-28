#include <metal_stdlib>

using namespace metal;

kernel void celeg_probe(device uint* output [[buffer(0)]],
                        uint index [[thread_position_in_grid]]) {
    if (index == 0) output[0] = 0xCE1E9u;
}
