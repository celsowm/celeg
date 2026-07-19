#pragma once

#include "lfm/cpu_kernels.hpp"
#include "lfm/cpu_model.hpp"
#include "lfm/cpu_paged_kv.hpp"
#include "lfm/cpu_prefix_cache.hpp"
#include "lfm/cpu_quantization.hpp"
#include "lfm/cpu_thread_pool.hpp"
#include "lfm/model_shape.hpp"
#include "lfm/model_variant.hpp"

#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace lfm {

struct CpuModel::Impl {
    struct CommonWeights {
        std::vector<float> operator_norm;
        std::vector<float> ffn_norm;
        Q4GroupMatrix w13;
        Q4GroupMatrix w2;
    };
    struct AttentionWeights {
        CommonWeights common;
        Q4GroupMatrix qkv;
        Q4GroupMatrix out;
        std::vector<float> q_norm;
        std::vector<float> k_norm;
    };
    struct ConvolutionWeights {
        CommonWeights common;
        Q4GroupMatrix in;
        std::vector<float> weight;
        Q4GroupMatrix out;
    };
    using WeightLayer = std::variant<AttentionWeights, ConvolutionWeights>;

    struct Shared {
        Shared(const std::string& path, int context, CpuModelOptions requested);

        static CpuIsa resolve_isa(CpuIsa requested);
        void prepare_pack_path();
        void load_weights();
        CommonWeights load_common(class SafeTensorFile* file,
                                  class CpuPackReader* reader,
                                  class CpuPackWriter* writer,
                                  int layer);
        Q4GroupMatrix load_matrix(class SafeTensorFile* file,
                                  class CpuPackReader* reader,
                                  class CpuPackWriter* writer,
                                  const std::string& name,
                                  const std::vector<int64_t>& expected);
        Q4GroupMatrix load_concat(
            class SafeTensorFile* file,
            class CpuPackReader* reader,
            class CpuPackWriter* writer,
            const std::string& synthetic,
            const std::vector<std::pair<std::string, std::vector<int64_t>>>& parts);
        std::vector<float> load_vector(class SafeTensorFile* file,
                                       class CpuPackReader* reader,
                                       class CpuPackWriter* writer,
                                       const std::string& name,
                                       const std::vector<int64_t>& expected);
        size_t weights_memory_bytes() const;

        std::string safetensors_path;
        int max_context = 0;
        CpuModelOptions options;
        CpuCapabilities capabilities;
        CpuThreadPool pool;
        CpuLinearEngine linear;
        size_t group_size = 32;
        std::filesystem::path pack_file;
        std::string source_id;
        bool loaded_pack = false;
        ModelShape shape;
        const IModelVariant* variant = nullptr;
        Q4GroupMatrix embedding;
        std::vector<float> final_norm;
        std::vector<WeightLayer> layers;
        std::vector<std::shared_ptr<CpuKvPagePool>> kv_pools;
        std::vector<int> layer_to_kv_pool;
    };

    struct AttentionState {
        size_t pool_index = 0;
        std::vector<CpuKvPageId> pages;
        size_t token_count = 0;
    };
    struct ConvolutionState {
        std::vector<float> state;
    };
    using LayerState = std::variant<AttentionState, ConvolutionState>;

    Impl(std::shared_ptr<Shared> shared_weights,
         GenerationConfig generation_config,
         int preferred_numa_node = -1);
    ~Impl();

    void allocate_state();
    void allocate_activations();
    void reset();
    void forward_token(int32_t token, bool compute_logits);
    void forward_chunk(std::span<const int32_t> tokens, bool compute_logits);
    int32_t sample();
    void set_generation(GenerationConfig config);
    CpuModelMemoryStats memory_stats() const;

    const CommonWeights& common_weights(size_t layer) const;
    AttentionState& attention_state(size_t layer);
    const AttentionState& attention_state(size_t layer) const;
    ConvolutionState& convolution_state(size_t layer);
    const ConvolutionState& convolution_state(size_t layer) const;

    void store_kv(AttentionState& state, int position,
                  const float* key, const float* value);
    void run_attention(const AttentionState& state, const float* q,
                       float* output, int sequence_length) const;
    void release_attention_pages(AttentionState& state) noexcept;

    CpuPrefixSnapshot export_prefix_snapshot() const;
    void restore_prefix_snapshot(CpuPrefixSnapshot snapshot,
                                 bool ready_for_decode);

    std::shared_ptr<Shared> shared;
    GenerationConfig generation;
    std::vector<LayerState> states;

    std::vector<float> hidden;
    std::vector<float> residual;
    std::vector<float> normed;
    std::vector<float> op_output;
    std::vector<float> qkv;
    std::vector<float> conv_projected;
    std::vector<float> gate_up;
    std::vector<float> activated;
    std::vector<float> mlp_output;
    std::vector<float> logits;
    std::vector<uint8_t> seen;

    // Lazily sized and reused by layer-major single-prompt prefill.
    std::vector<float> chunk_hidden;
    std::vector<float> chunk_residual;
    std::vector<float> chunk_normed;
    std::vector<float> chunk_op;
    std::vector<float> chunk_qkv;
    std::vector<float> chunk_conv;
    std::vector<float> chunk_gate_up;
    std::vector<float> chunk_activated;
    std::vector<float> chunk_mlp;

    int position_value = 0;
    int preferred_numa_node = -1;
    uint64_t attention_parallel_calls = 0;
    SessionPhase phase = SessionPhase::Empty;
    uint64_t rng_state = 1;
    RuntimeMetrics metrics;
};

} // namespace lfm
