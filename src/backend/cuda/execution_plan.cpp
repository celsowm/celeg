#include "celeg/backend/cuda/execution_plan.hpp"

#include <sstream>
#include <stdexcept>

namespace celeg {

CudaExecutionPlan CudaExecutionPlan::compile(CudaModelOptions requested, int max_context) {
    if (max_context <= 0) {
        throw std::invalid_argument("max_context must be positive");
    }
    if (requested.lt_heuristics <= 0 || requested.lt_heuristics > 64) {
        throw std::invalid_argument("lt_heuristics must be between 1 and 64");
    }
    if (requested.attention_chunk_tokens <= 0) {
        throw std::invalid_argument("attention_chunk_tokens must be positive");
    }
    const bool segmented_capable =
        requested.attention_mode != AttentionMode::Single;
    if (segmented_capable && requested.attention_chunk_tokens > max_context) {
        throw std::invalid_argument(
            "attention_chunk_tokens must not exceed max_context when segmented attention is enabled");
    }
    if (segmented_capable && !requested.fast_attention) {
        throw std::invalid_argument(
            "segmented or automatic attention requires --fast-attention");
    }
    if (requested.attention_mode == AttentionMode::Auto &&
        requested.attention_auto_threshold <= 0) {
        throw std::invalid_argument(
            "attention_auto_threshold must be positive");
    }
#if defined(CELEG_DEBUG_SYNC) && CELEG_DEBUG_SYNC
    if (requested.cuda_graph) {
        throw std::invalid_argument(
            "CUDA Graph capture is incompatible with CELEG_DEBUG_SYNC; use --no-cuda-graph");
    }
#endif

    CudaExecutionPlan plan;
    plan.options_ = requested;

    // MoE expert offload resolves residency at decode time using host-roundtrip
    // reads of the router output and cross-stream event synchronization, neither
    // of which is capturable into a CUDA graph. Force graph capture off when
    // offload is enabled.
    if (requested.expert_offload.enabled()) {
        plan.options_.cuda_graph = false;
    }

    switch (requested.weight_mode) {
        case WeightMode::Bf16:
            plan.linear_kernel_ = requested.gemm_backend == GemmBackend::CublasLt
                ? LinearKernelKind::Bf16CublasLt
                : LinearKernelKind::Bf16Cublas;
            break;
        case WeightMode::Int8:
            plan.linear_kernel_ = LinearKernelKind::W8A16;
            break;
        case WeightMode::Int4:
            plan.linear_kernel_ = LinearKernelKind::W4A16;
            break;
        case WeightMode::NativeGguf:
            // Phase 1.4: a native-GGUF weight mode mixes BF16 (norms/conv) and
            // GGUF MMQ (linear blocks). Do not report this path as plain
            // BF16 cuBLASLt; the per-tensor dispatcher switches to MMQ for
            // the GGUF-quantized tensors at run time.
            plan.linear_kernel_ = LinearKernelKind::MixedBf16AndGgufMmq;
            break;
    }
    if (segmented_capable) {
        plan.attention_chunks_ =
            (max_context + requested.attention_chunk_tokens - 1) /
            requested.attention_chunk_tokens;
    }
    return plan;
}

bool CudaExecutionPlan::segmented_attention(int position) const {
    return select_segmented_attention(
        options_.attention_mode, position, options_.attention_auto_threshold);
}

std::string CudaExecutionPlan::description() const {
    std::ostringstream out;
    out << "linear=";
    switch (linear_kernel_) {
        case LinearKernelKind::Bf16Cublas: out << "bf16-cublas"; break;
        case LinearKernelKind::Bf16CublasLt: out << "bf16-cublaslt"; break;
        case LinearKernelKind::W8A16: out << "w8a16"; break;
        case LinearKernelKind::W4A16: out << "w4a16"; break;
        case LinearKernelKind::Q4kMmq: out << "q4k-mmq"; break;
        case LinearKernelKind::Q6kMmq: out << "q6k-mmq"; break;
        case LinearKernelKind::MixedBf16AndGgufMmq:
            out << "mixed-bf16-and-gguf-mmq"; break;
    }
    out << ", sampling=fused, attention=";
    switch (options_.attention_mode) {
        case AttentionMode::Single: out << "single"; break;
        case AttentionMode::Segmented: out << "segmented"; break;
        case AttentionMode::Auto: out << "auto"; break;
    }
    return out.str();
}

} // namespace celeg
