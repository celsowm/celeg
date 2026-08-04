#include "api_internal.hpp"

#include <stdexcept>

namespace celeg::api {

thread_local std::string global_error;

void require_size(uint32_t actual, size_t expected, const char* name) {
    if (actual < expected) {
        throw std::invalid_argument(std::string(name) + " struct is too small");
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
        case 0: return CpuIsa::Auto;
        case 1: return CpuIsa::Scalar;
        case 2: return CpuIsa::Avx2;
        case 3: return CpuIsa::AvxVnni;
        case 4: return CpuIsa::Avx512Vnni;
        case 5: return CpuIsa::AmxInt8;
        case 6: return CpuIsa::Neon;
        case 7: return CpuIsa::DotProd;
        case 8: return CpuIsa::I8mm;
        case 9: return CpuIsa::Sve2;
        case 10: return CpuIsa::Sme2;
        default: throw std::invalid_argument("invalid CPU ISA");
    }
}

CpuAffinityPolicy cpu_affinity(int value) {
    switch (value) {
        case 0: return CpuAffinityPolicy::None;
        case 1: return CpuAffinityPolicy::Compact;
        case 2: return CpuAffinityPolicy::Scatter;
        default: throw std::invalid_argument("invalid CPU affinity policy");
    }
}

CpuKvCacheMode cpu_kv_cache_mode(int value) {
    switch (value) {
        case 0: return CpuKvCacheMode::Fp32;
        case 1: return CpuKvCacheMode::Bf16;
        default: throw std::invalid_argument("invalid CPU KV cache mode");
    }
}

CpuNumaMode cpu_numa_mode(int value) {
    switch (value) {
        case 0: return CpuNumaMode::Disabled;
        case 1: return CpuNumaMode::Local;
        case 2: return CpuNumaMode::ReplicateWeights;
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

CpuModelOptions cpu_options(const celeg_engine_model_options& source) {
    if (source.backend != CELEG_BACKEND_CPU) {
        throw std::invalid_argument("CPU options require CPU backend");
    }
    return cpu_options(source.backend_options.cpu);
}

CpuConcurrentEngineOptions cpu_engine_options(const celeg_engine_options& source) {
    const auto& input = source.backend_options.cpu;
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
        case 0: return WeightMode::Bf16;
        case 1: return WeightMode::Int8;
        case 2: return WeightMode::Int4;
        default: throw std::invalid_argument("invalid CUDA weight mode");
    }
}

KvCacheMode cuda_kv_cache_mode(int value) {
    switch (value) {
        case 0: return KvCacheMode::Bf16;
        case 1: return KvCacheMode::Int8;
        default: throw std::invalid_argument("invalid CUDA KV cache mode");
    }
}

GemmBackend cuda_gemm_backend(int value) {
    switch (value) {
        case 0: return GemmBackend::Cublas;
        case 1: return GemmBackend::CublasLt;
        default: throw std::invalid_argument("invalid CUDA GEMM backend");
    }
}

AttentionMode cuda_attention_mode(int value) {
    switch (value) {
        case 0: return AttentionMode::Single;
        case 1: return AttentionMode::Segmented;
        case 2: return AttentionMode::Auto;
        default: throw std::invalid_argument("invalid CUDA attention mode");
    }
}

SchedulerPolicy cuda_scheduler_policy(int value) {
    switch (value) {
        case 0: return SchedulerPolicy::GuaranteedNoEvict;
        case 1: return SchedulerPolicy::MaxUtilization;
        default: throw std::invalid_argument("invalid CUDA scheduler policy");
    }
}
} // namespace

CudaModelOptions cuda_options(const celeg_engine_model_options& source) {
    const auto& input = source.backend_options.cuda;
    if (source.backend != CELEG_BACKEND_CUDA) {
        throw std::invalid_argument("CUDA options require CUDA backend");
    }
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

ConcurrentEngineOptions cuda_engine_options(const celeg_engine_options& source) {
    const auto& input = source.backend_options.cuda;
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

} // namespace celeg::api
