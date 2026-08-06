#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace celeg {

// Raw multimodal inputs replace selected token positions during prefill.
// Values are row-major by override, with one vector per position.
struct PromptEmbedding {
    int width = 0;
    std::vector<std::size_t> positions;
    std::vector<float> values;

    bool empty() const { return positions.empty(); }

    const float* at_position(std::size_t position) const {
        for (std::size_t index = 0; index < positions.size(); ++index) {
            if (positions[index] == position) {
                return values.data() + index * static_cast<std::size_t>(width);
            }
        }
        return nullptr;
    }
};

// Public generation limit shared by CPU and CUDA samplers.
inline constexpr int kMaxTopK = 128;

struct ModelMemoryStats {
    size_t weights = 0;
    size_t kv_cache = 0;
    size_t conv_state = 0;
    size_t activations = 0;
    size_t sampling = 0;
    size_t matmul_workspace = 0;
    size_t attention_workspace = 0;

    size_t total() const {
        return weights + kv_cache + conv_state + activations +
               sampling + matmul_workspace + attention_workspace;
    }
};

struct RuntimeMetrics {
    double last_prefill_ms = 0.0;
    uint64_t prefill_tokens = 0;
    double cumulative_decode_ms = 0.0;
    uint64_t decoded_tokens = 0;

    double prefill_tokens_per_second() const;
    double decode_tokens_per_second() const;
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
    std::vector<uint16_t> mamba_state_bf16;
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
