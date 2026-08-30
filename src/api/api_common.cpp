#include "api_internal.hpp"

#include <stdexcept>

namespace celeg::api {

thread_local std::string global_error;

void require_size(uint32_t actual, size_t expected, const char* name) {
    if (actual < expected) {
        throw std::invalid_argument(std::string(name) + " struct is too small");
    }
}

void validate_backend_request(const BackendCreateRequest& request,
                             BackendId expected) {
    if (!request.runtime || !request.options || request.backend_id != expected ||
        request.options->backend_id() != expected) {
        throw std::invalid_argument("backend request has invalid options");
    }
}

GenerationConfig generation(const celeg_generation_options& source) {
    GenerationConfig result;
    result.temperature = source.temperature;
    result.top_k = source.top_k;
    result.top_p = source.top_p;
    result.repetition_penalty = source.repetition_penalty;
    result.seed = source.seed;
    result.validate();
    return result;
}

CpuIsa cpu_isa(int value) {
    switch (value) {
        case CELEG_CPU_ISA_AUTO: return CpuIsa::Auto;
        case CELEG_CPU_ISA_SCALAR: return CpuIsa::Scalar;
        case CELEG_CPU_ISA_AVX2: return CpuIsa::Avx2;
        case CELEG_CPU_ISA_AVX_VNNI: return CpuIsa::AvxVnni;
        case CELEG_CPU_ISA_AVX512_VNNI: return CpuIsa::Avx512Vnni;
        case CELEG_CPU_ISA_AMX_INT8: return CpuIsa::AmxInt8;
        case CELEG_CPU_ISA_NEON: return CpuIsa::Neon;
        case CELEG_CPU_ISA_DOTPROD: return CpuIsa::DotProd;
        case CELEG_CPU_ISA_I8MM: return CpuIsa::I8mm;
        case CELEG_CPU_ISA_SVE2: return CpuIsa::Sve2;
        case CELEG_CPU_ISA_SME2: return CpuIsa::Sme2;
        default: throw std::invalid_argument("invalid CPU ISA");
    }
}

CpuAffinityPolicy cpu_affinity(int value) {
    switch (value) {
        case CELEG_CPU_AFFINITY_NONE: return CpuAffinityPolicy::None;
        case CELEG_CPU_AFFINITY_COMPACT: return CpuAffinityPolicy::Compact;
        case CELEG_CPU_AFFINITY_SCATTER: return CpuAffinityPolicy::Scatter;
        default: throw std::invalid_argument("invalid CPU affinity policy");
    }
}

CpuKvCacheMode cpu_kv_cache_mode(int value) {
    switch (value) {
        case CELEG_CPU_KV_CACHE_FP32: return CpuKvCacheMode::Fp32;
        case CELEG_CPU_KV_CACHE_BF16: return CpuKvCacheMode::Bf16;
        default: throw std::invalid_argument("invalid CPU KV cache mode");
    }
}

CpuNumaMode cpu_numa_mode(int value) {
    switch (value) {
        case CELEG_CPU_NUMA_DISABLED: return CpuNumaMode::Disabled;
        case CELEG_CPU_NUMA_LOCAL: return CpuNumaMode::Local;
        case CELEG_CPU_NUMA_REPLICATE_WEIGHTS: return CpuNumaMode::ReplicateWeights;
        default: throw std::invalid_argument("invalid CPU NUMA mode");
    }
}

CpuModelOptions cpu_options(const celeg_cpu_model_config& input) {
    if (input.q4_group_size != 32 && input.q4_group_size != 64) {
        throw std::invalid_argument("CPU Q4 group size must be 32 or 64");
    }
    CpuModelOptions result;
    result.isa = cpu_isa(input.isa);
    result.threads = input.threads > 0 ? static_cast<size_t>(input.threads) : 0;
    result.weight_format = input.q4_group_size == 64
        ? CpuWeightFormat::Q4Group64 : CpuWeightFormat::Q4Group32;
    result.use_pack_cache = input.use_pack_cache != 0;
    if (input.pack_cache_directory) result.pack_cache_directory = input.pack_cache_directory;
    result.affinity = cpu_affinity(input.affinity);
    result.kv_cache_mode = cpu_kv_cache_mode(input.kv_cache_mode);
    result.kv_page_tokens = input.kv_page_tokens;
    result.prefill_chunk_tokens = input.prefill_chunk_tokens;
    result.prefill_chunk_threshold = input.prefill_chunk_threshold;
    result.attention_parallel_threshold = input.attention_parallel_threshold;
    result.attention_page_tile = input.attention_page_tile;
    result.numa_mode = cpu_numa_mode(input.numa_mode);
    return result;
}

CpuModelOptions cpu_options(const celeg_cpu_model_options& source) {
    return cpu_options(source.cpu);
}

CpuConcurrentEngineOptions cpu_engine_options(const celeg_cpu_engine_options& input) {
    CpuConcurrentEngineOptions result;
    result.max_active_requests = input.max_active_requests;
    result.max_batched_tokens = input.max_batched_tokens;
    result.max_prefill_batch = input.max_prefill_batch;
    result.max_decode_batch = input.max_decode_batch;
    result.decode_first = input.decode_first != 0;
    result.long_prefill_chunk_tokens = input.long_prefill_chunk_tokens;
    result.long_prefill_threshold = input.long_prefill_threshold;
    result.prefix_cache = input.prefix_cache != 0;
    result.prefix_cache_max_entries = input.prefix_cache_max_entries;
    result.prefix_cache_max_bytes = input.prefix_cache_max_bytes;
    return result;
}

celeg_request_status status(serve::RequestStatus source) {
    switch (source) {
        case serve::RequestStatus::Queued: return CELEG_REQUEST_QUEUED;
        case serve::RequestStatus::Prefill: return CELEG_REQUEST_PREFILLING;
        case serve::RequestStatus::Decoding: return CELEG_REQUEST_DECODING;
        case serve::RequestStatus::Finished: return CELEG_REQUEST_COMPLETED;
        case serve::RequestStatus::Cancelled: return CELEG_REQUEST_CANCELLED;
        case serve::RequestStatus::Failed: return CELEG_REQUEST_FAILED;
    }
    return CELEG_REQUEST_FAILED;
}

#ifdef CELEG_API_WITH_CUDA
namespace {
WeightMode cuda_weight_mode(int value) {
    switch (value) {
        case CELEG_WEIGHT_MODE_BF16: return WeightMode::Bf16;
        case CELEG_WEIGHT_MODE_INT8: return WeightMode::Int8;
        case CELEG_WEIGHT_MODE_INT4: return WeightMode::Int4;
        case CELEG_WEIGHT_MODE_NATIVE_GGUF: return WeightMode::NativeGguf;
        default: throw std::invalid_argument("invalid CUDA weight mode");
    }
}

KvCacheMode cuda_kv_cache_mode(int value) {
    switch (value) {
        case CELEG_CUDA_KV_CACHE_BF16: return KvCacheMode::Bf16;
        case CELEG_CUDA_KV_CACHE_INT8: return KvCacheMode::Int8;
        default: throw std::invalid_argument("invalid CUDA KV cache mode");
    }
}

GemmBackend cuda_gemm_backend(int value) {
    switch (value) {
        case CELEG_CUDA_GEMM_CUBLAS: return GemmBackend::Cublas;
        case CELEG_CUDA_GEMM_CUBLASLT: return GemmBackend::CublasLt;
        default: throw std::invalid_argument("invalid CUDA GEMM backend");
    }
}

AttentionMode cuda_attention_mode(int value) {
    switch (value) {
        case CELEG_CUDA_ATTENTION_SINGLE: return AttentionMode::Single;
        case CELEG_CUDA_ATTENTION_SEGMENTED: return AttentionMode::Segmented;
        case CELEG_CUDA_ATTENTION_AUTO: return AttentionMode::Auto;
        default: throw std::invalid_argument("invalid CUDA attention mode");
    }
}

SchedulerPolicy cuda_scheduler_policy(int value) {
    switch (value) {
        case CELEG_CUDA_SCHEDULER_GUARANTEED_NO_EVICT:
            return SchedulerPolicy::GuaranteedNoEvict;
        case CELEG_CUDA_SCHEDULER_MAX_UTILIZATION:
            return SchedulerPolicy::MaxUtilization;
        default: throw std::invalid_argument("invalid CUDA scheduler policy");
    }
}
}

CudaModelOptions cuda_options(const celeg_cuda_model_options& input) {
    if (input.flags != 0) {
        throw std::invalid_argument("CUDA model option flags are reserved and must be zero");
    }
    CudaModelOptions result;
    result.weight_mode = cuda_weight_mode(input.weight_mode);
    result.kv_cache_mode = cuda_kv_cache_mode(input.kv_cache_mode);
    result.gemm_backend = cuda_gemm_backend(input.gemm_backend);
    result.attention_mode = cuda_attention_mode(input.attention_mode);
    result.attention_chunk_tokens = input.attention_chunk_tokens;
    result.attention_auto_threshold = input.attention_auto_threshold;
    if (input.lt_workspace_mb > 0) {
        result.lt_workspace_bytes = static_cast<size_t>(input.lt_workspace_mb) * 1024ULL * 1024ULL;
    }
    result.lt_heuristics = input.lt_heuristics;
    return result;
}

ConcurrentEngineOptions cuda_engine_options(const celeg_cuda_engine_options& input) {
    ConcurrentEngineOptions result;
    result.max_active_requests = input.max_active_requests;
    result.max_batched_tokens = input.max_batched_tokens;
    result.prefill_chunk_tokens = input.prefill_chunk_tokens;
    result.page_tokens = input.page_tokens;
    result.logical_kv_pages = input.logical_kv_pages;
    result.scheduler_policy = cuda_scheduler_policy(input.scheduler_policy);
    return result;
}
#endif

#ifdef CELEG_API_WITH_METAL
MetalModelOptions metal_options(const celeg_metal_model_options& input) {
    if (input.weight_mode != CELEG_METAL_WEIGHT_BF16) {
        throw std::invalid_argument("invalid Metal weight mode");
    }
    if (input.kv_cache_mode != CELEG_METAL_KV_CACHE_BF16) {
        throw std::invalid_argument("invalid Metal KV cache mode");
    }
    if (input.storage_mode != CELEG_METAL_STORAGE_SHARED &&
        input.storage_mode != CELEG_METAL_STORAGE_PRIVATE) {
        throw std::invalid_argument("invalid Metal storage mode");
    }
    if (input.kv_page_tokens <= 0) {
        throw std::invalid_argument("Metal KV page size must be positive");
    }
    return {
        MetalWeightMode::Bf16,
        MetalKvCacheMode::Bf16,
        input.storage_mode == CELEG_METAL_STORAGE_PRIVATE
            ? MetalStorageMode::Private
            : MetalStorageMode::Shared,
        input.kv_page_tokens,
    };
}

MetalEngineOptions metal_engine_options(const celeg_metal_engine_options& input) {
    if (input.max_active_requests <= 0 || input.max_batched_tokens <= 0 ||
        input.prefill_chunk_tokens <= 0 || input.kv_page_tokens <= 0 ||
        input.prefix_cache_max_entries == 0 ||
        (input.prefix_cache != 0 && input.prefix_cache != 1)) {
        throw std::invalid_argument("Metal engine limits must be positive");
    }
    return {input.max_active_requests, input.max_batched_tokens,
            input.prefill_chunk_tokens, input.kv_page_tokens,
            input.prefix_cache != 0, input.prefix_cache_max_entries};
}
#endif

}
