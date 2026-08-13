#pragma once

#include "celeg/backend/cuda/gemm_dispatcher.hpp"

#include <memory>

namespace celeg {

// Owns packed execution's lazily-created GEMM/autotuning runtime. The
// compiled packed program only borrows the dispatcher during a launch.
class PackedGemmRuntime {
public:
    void ensure(cudaStream_t stream, const CudaModelOptions& options);
    GemmDispatcher& dispatcher();

private:
    std::unique_ptr<GemmDispatcher> dispatcher_;
    CudaModelOptions options_{};
};

} // namespace celeg
