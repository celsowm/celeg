#pragma once

#include "lfm/config.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lfm {

// Runtime model topology descriptor. Replaces the former LfmConfig constexpr
// struct so multiple checkpoints (230M, 1.2B-Instruct, ..., 8B-A1B MoE) can
// coexist.
//
// Invariants are validated once at construction; hot paths read derived fields
// directly. ModelShape is value-semantic and cheap to copy.
struct ModelShape {
    int hidden = 0;
    int intermediate = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    int conv_cache = 0;
    int conv_dim = 0;
    int max_position_embeddings = 0;
    int bos_token_id = -1;
    int eos_token_id = -1;
    int pad_token_id = -1;
    float norm_eps = 0.0f;
    float rope_theta = 0.0f;
    std::string rope_type;
    std::vector<LayerType> layer_types;

    // ---- MoE topology (populated when architecture == MoeLfm2) ----
    ArchitectureKind architecture = ArchitectureKind::DenseLfm2;
    // Dense FFN intermediate size (used by the first `num_dense_layers` layers).
    int dense_intermediate = 0;
    // MoE expert FFN intermediate size (used by the remaining layers).
    int moe_intermediate = 0;
    int num_dense_layers = 0;
    int num_experts = 0;
    int experts_per_token = 0;
    bool normalize_topk = false;
    bool use_expert_bias = false;
    float routed_scaling_factor = 1.0f;

    // True when the layer at `layer_index` uses the MoE feed-forward path.
    // For the MoE architecture this is every layer at or beyond
    // `num_dense_layers`; dense architectures never use MoE.
    bool layer_uses_moe(int layer_index) const {
        return architecture == ArchitectureKind::MoeLfm2 &&
               layer_index >= num_dense_layers;
    }

    // Derived fields populated by compute_derived().
    int q_width = 0;            // num_attention_heads * head_dim
    int kv_width = 0;           // num_key_value_heads * head_dim
    int qkv_width = 0;          // q_width + 2 * kv_width
    int rope_pairs = 0;         // head_dim / 2
    int attention_layer_count = 0;
    int conv_layer_count = 0;
    // layer_index -> attention slot in [0, attention_layer_count).
    // -1 for convolution layers.
    std::vector<int> attention_slot_for_layer;
    // attention slot -> model layer index.
    std::vector<int> layer_for_attention_slot;

    static ModelShape from_config(const ModelConfig& config);

    void compute_derived();
    void validate() const;

    // Compact fingerprint (e.g. "hidden2048-layers16-qh32-kvh8-hd64-voc65536")
    // used in cache filenames and session headers.
    std::string fingerprint() const;

    std::string summary() const;
};

} // namespace lfm
