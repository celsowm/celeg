#pragma once

#include "celeg/backend/cuda/utils.cuh"

namespace celeg {

// CUDA objects shared by the packed workspace and its compiled operators.
// Keeping their lifetime in one value makes stream/handle ownership explicit
// and prevents individual operators from creating runtime resources.
struct PackedExecutionHandles {
    PackedExecutionHandles() : stream(), cublas(stream.get()) {}

    PackedExecutionHandles(const PackedExecutionHandles&) = delete;
    PackedExecutionHandles& operator=(const PackedExecutionHandles&) = delete;

    CudaStream stream;
    CublasHandle cublas;
};

} // namespace celeg
