#pragma once

#include "lfm/policy.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lfm {

struct LfmConfig {
    static constexpr int hidden = 1024;
    static constexpr int intermediate = 2560;
    static constexpr int layers = 14;
    static constexpr int q_heads = 16;
    static constexpr int kv_heads = 8;
    static constexpr int head_dim = 64;
    static constexpr int q_width = q_heads * head_dim;
    static constexpr int kv_width = kv_heads * head_dim;
    static constexpr int qkv_width = q_width + 2 * kv_width;
    static constexpr int vocab = 65536;
    static constexpr int conv_cache = 3;
    static constexpr int rope_pairs = head_dim / 2;
    static constexpr float norm_eps = 1e-5f;
    static constexpr float rope_theta = 1'000'000.0f;
    static constexpr int max_top_k = 128;
};

enum class GemmBackend {
    Cublas,
    CublasLt,
};

enum class WeightMode {
    Bf16,
    Int8,
    Int4,
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
    AttentionMode attention_mode = AttentionMode::Single;
    int attention_chunk_tokens = 256;
    int attention_auto_threshold = 4096;
    bool allocate_local_kv_cache = true;
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

} // namespace lfm
