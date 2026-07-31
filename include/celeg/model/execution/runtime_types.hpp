#pragma once

#include "celeg/runtime/concurrency/policy_types.hpp"
#include "celeg/runtime/moe/offload.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace celeg {

// Upper bound on the top-k sampler buffer. This is a kernel/workspace limit,
// not a model-topology property, so it lives here instead of the resolved model.
inline constexpr int kMaxTopK = 128;

enum class GemmBackend {
    Cublas,
    CublasLt,
};

enum class WeightMode {
    Bf16,
    Int8,
    Int4,
    // Keep GGUF block-quantized weights in their native on-disk format
    // (Q4_K / Q6_K) on the device instead of dequantizing to BF16 at load
    // time. The matmul kernels dequantize on the fly, reading ~3x less
    // memory per decode token than the BF16 path.
    NativeGguf,
};

enum class KvCacheMode {
    Bf16,
    Int8,
};

struct ModelOptions {
    bool fused_residuals = false;
    bool fast_attention = false;
    bool fused_projections = false;
    bool fused_sampling = true;
    bool cuda_graph = true;
    bool legacy_prefill = false;
    GemmBackend gemm_backend = GemmBackend::Cublas;
    size_t lt_workspace_bytes = 64ULL * 1024ULL * 1024ULL;
    int lt_heuristics = 8;
    bool lt_autotune = false;
    WeightMode weight_mode = WeightMode::Bf16;
    KvCacheMode kv_cache_mode = KvCacheMode::Bf16;
    // Segmented (chunked, parallel-reduce) decode attention beats the
    // single serial per-KV-token kernel across the whole range measured
    // (position 16 through 1500+, chunk 32-64): decode has far less
    // baseline block parallelism than prefill (q_heads blocks, not
    // rows*q_heads), so splitting the KV history into chunks increases
    // occupancy rather than just reshuffling already-hidden latency.
    // Callers that want it must also set fast_attention = true (Auto and
    // Segmented both require it - see ExecutionPlan::compile); the mode
    // itself stays Single by default so a default-constructed ModelOptions
    // remains valid without that extra requirement. The CLI opts in
    // explicitly (attention_mode "auto" + fast_attention true by default).
    AttentionMode attention_mode = AttentionMode::Single;
    int attention_chunk_tokens = 32;
    int attention_auto_threshold = 1;
    bool allocate_local_kv_cache = true;
    ExpertOffloadOptions expert_offload;
};

struct ModelMemoryStats {
    size_t weights = 0;
    size_t kv_cache = 0;
    size_t conv_state = 0;
    size_t rope_tables = 0;
    size_t activations = 0;
    size_t sampling = 0;
    size_t matmul_workspace = 0;
    size_t attention_workspace = 0;

    size_t total() const {
        return weights + kv_cache + conv_state + rope_tables + activations +
               sampling + matmul_workspace + attention_workspace;
    }
};

struct RuntimeMetrics {
    double last_prefill_ms = 0.0;
    uint64_t prefill_tokens = 0;
    double cumulative_decode_ms = 0.0;
    uint64_t decoded_tokens = 0;

    double prefill_tokens_per_second() const {
        return last_prefill_ms > 0.0
            ? static_cast<double>(prefill_tokens) * 1000.0 / last_prefill_ms
            : 0.0;
    }
    double decode_tokens_per_second() const {
        return cumulative_decode_ms > 0.0
            ? static_cast<double>(decoded_tokens) * 1000.0 / cumulative_decode_ms
            : 0.0;
    }
};

struct DecodeBenchmark {
    int warmup_steps = 0;
    int measured_steps = 0;
    float elapsed_ms = 0.0f;

    float milliseconds_per_token() const {
        return measured_steps > 0 ? elapsed_ms / measured_steps : 0.0f;
    }
    float tokens_per_second() const {
        return elapsed_ms > 0.0f ? measured_steps * 1000.0f / elapsed_ms : 0.0f;
    }
};

struct PrefixState {
    int position = 0;
    std::vector<uint8_t> seen_tokens;
    std::vector<uint16_t> logits_bf16;
    std::vector<uint16_t> conv_state_bf16;
};

struct GenerationConfig {
    float temperature = 0.1f;
    int top_k = 50;
    float top_p = 1.0f;
    float repetition_penalty = 1.05f;
    uint64_t seed = 1;

    bool greedy() const { return temperature <= 0.0f || top_k == 1; }
    void validate() const;
};

enum class SessionPhase : uint8_t {
    Empty,
    Prefilling,
    Ready,
    DecodePending,
};

} // namespace celeg
