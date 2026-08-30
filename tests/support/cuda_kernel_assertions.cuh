#pragma once

#include "utils.cuh"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>

namespace celeg::cuda_test {

inline float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

inline __nv_bfloat16 to_bf16(float value) {
    return __float2bfloat16(value);
}

inline void expect_near(float actual, float expected, float tolerance = 0.03f) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "expected " << expected << ", got " << actual << '\n';
        std::abort();
    }
}

}
