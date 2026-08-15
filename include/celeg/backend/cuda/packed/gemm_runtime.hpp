#pragma once

#include "celeg/backend/cuda/gemm_dispatcher.hpp"

#include <memory>

namespace celeg {

class PackedGemmRuntime {
public:
    void ensure(cudaStream_t stream, const CudaModelOptions& options);
    GemmDispatcher& dispatcher();

private:
    std::unique_ptr<GemmDispatcher> dispatcher_;
    CudaModelOptions options_{};
};

}
