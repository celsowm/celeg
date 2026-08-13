#include "celeg/backend/cuda/packed/gemm_runtime.hpp"

#include <stdexcept>

namespace celeg {

void PackedGemmRuntime::ensure(cudaStream_t stream,
                               const CudaModelOptions& options) {
    if (dispatcher_ && options_.gemm_backend == options.gemm_backend &&
        options_.lt_workspace_bytes == options.lt_workspace_bytes &&
        options_.lt_heuristics == options.lt_heuristics &&
        options_.lt_autotune == options.lt_autotune) {
        return;
    }
    options_ = options;
    dispatcher_ = std::make_unique<GemmDispatcher>(stream, options_);
}

GemmDispatcher& PackedGemmRuntime::dispatcher() {
    if (!dispatcher_) throw std::logic_error("packed GEMM runtime is not initialized");
    return *dispatcher_;
}

} // namespace celeg
