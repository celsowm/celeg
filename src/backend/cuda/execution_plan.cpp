#include "celeg/backend/cuda/execution_plan.hpp"

#include <sstream>
#include <stdexcept>
#include <functional>
#include <type_traits>
#include <atomic>

namespace celeg {

namespace {
std::atomic<uint64_t> g_compile_count = 0;
}

uint64_t CudaExecutionPlan::compile_count() {
    return g_compile_count.load(std::memory_order_relaxed);
}

CudaExecutionPlan CudaExecutionPlan::compile(
    CudaModelOptions requested, int max_context, CudaDeviceCapabilities device) {
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
    if (requested.mtp_speculative_tokens <= 0) {
        throw std::invalid_argument("mtp_speculative_tokens must be positive");
    }
    if (requested.enable_mtp && requested.cuda_graph) {
        throw std::invalid_argument(
            "MTP requires --no-cuda-graph until proposal metrics are graph-safe");
    }
    if (requested.enable_mtp && requested.mtp_speculative_tokens != 1) {
        throw std::invalid_argument(
            "MTP currently supports exactly one speculative token");
    }
#if defined(CELEG_DEBUG_SYNC) && CELEG_DEBUG_SYNC
    if (requested.cuda_graph) {
        throw std::invalid_argument(
            "CUDA Graph capture is incompatible with CELEG_DEBUG_SYNC; use --no-cuda-graph");
    }
#endif

    CudaExecutionPlan plan;
    g_compile_count.fetch_add(1, std::memory_order_relaxed);
    plan.options_ = requested;
    device.mmq_tensor_core_enabled =
        (requested.mmq_tensor_cores.has_value() ? *requested.mmq_tensor_cores
                                                 : device.mmq_tensor_core_supported) &&
        device.mmq_tensor_core_supported;
    plan.device_ = device;
    plan.max_context_ = max_context;

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
            plan.linear_kernel_ = LinearKernelKind::MixedBf16AndGgufMmq;
            break;
    }
    if (segmented_capable) {
        plan.attention_chunks_ =
            (max_context + requested.attention_chunk_tokens - 1) /
            requested.attention_chunk_tokens;
    }
    std::size_t hash = 1469598103934665603ULL;
    const auto mix = [&hash](auto value) {
        hash ^= std::hash<std::decay_t<decltype(value)>>{}(value);
        hash *= 1099511628211ULL;
    };
    mix(plan.max_context_);
    mix(static_cast<int>(plan.linear_kernel_));
    mix(static_cast<int>(plan.options_.weight_mode));
    mix(static_cast<int>(plan.options_.kv_cache_mode));
    mix(static_cast<int>(plan.options_.gemm_backend));
    mix(static_cast<int>(plan.options_.attention_mode));
    mix(plan.options_.attention_chunk_tokens);
    mix(plan.options_.attention_auto_threshold);
    mix(plan.options_.fast_attention);
    mix(plan.options_.fused_projections);
    mix(plan.options_.fused_residuals);
    mix(plan.options_.cuda_graph);
    mix(plan.options_.lt_workspace_bytes);
    mix(plan.options_.lt_heuristics);
    mix(plan.options_.lt_autotune);
    mix(plan.options_.allocate_local_kv_cache);
    mix(plan.options_.enable_mtp);
    mix(plan.options_.mtp_speculative_tokens);
    mix(plan.options_.expert_offload.fingerprint());
    mix(plan.device_.device_ordinal);
    mix(plan.device_.compute_major);
    mix(plan.device_.compute_minor);
    mix(plan.device_.mmq_tensor_core_supported);
    mix(plan.device_.mmq_tensor_core_enabled);
    plan.fingerprint_ = static_cast<uint64_t>(hash);
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
        case LinearKernelKind::Fp8W8A8: out << "fp8-w8a8"; break;
        case LinearKernelKind::Nvfp4W4A4: out << "nvfp4-w4a4-bf16dequant"; break;
    }
    out << ", sampling=fused, attention=";
    switch (options_.attention_mode) {
        case AttentionMode::Single: out << "single"; break;
        case AttentionMode::Segmented: out << "segmented"; break;
        case AttentionMode::Auto: out << "auto"; break;
    }
    out << ", device=" << device_.device_ordinal << "."
        << device_.compute_major << device_.compute_minor
        << ", mmq-tensor-cores="
        << (device_.mmq_tensor_core_enabled ? "on" : "off")
        << " (override=";
    if (options_.mmq_tensor_cores.has_value()) {
        out << (*options_.mmq_tensor_cores ? "on" : "off");
    } else {
        out << "auto";
    }
    out << ")"
        << ", flash-attn=" << (options_.flash_attn ? "on" : "off")
        << ", managed-weights=" << (options_.managed_weights ? "on" : "off")
        << ", fingerprint=" << fingerprint_;
    return out.str();
}

}
