#pragma once

#include "celeg/backend/cuda/utils.cuh"

namespace celeg {

struct PackedExecutionHandles {
    PackedExecutionHandles() : stream(), cublas(stream.get()) {}

    PackedExecutionHandles(const PackedExecutionHandles&) = delete;
    PackedExecutionHandles& operator=(const PackedExecutionHandles&) = delete;

    CudaStream stream;
    CublasHandle cublas;
};

}
