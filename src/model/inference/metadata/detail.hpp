#pragma once

#include "celeg/checkpoint/metadata.hpp"
#include "celeg/model/inference.hpp"

#include <string_view>
#include <vector>

namespace celeg {

/// Local facts produced by `normalize_rope_position_facts` and consumed by
/// `normalize_rope_pairing`; the original single function carried them as
/// locals between the two statement clusters.
struct RopePositionFacts {
    std::optional<double> rope_theta;
    std::optional<float> rotary_fraction;
    std::vector<int> mrope_sections;
    bool mrope_interleaved = false;
    bool architecture_never_applies_rope = false;
};

/// Core-dimension and policy clusters of `normalize_model_metadata`.
/// Each helper takes (const CheckpointMetadata&, NormalizedModelMetadata&)
/// unless noted; statement order, mutation order and error paths match the
/// original single body.
void normalize_core_dims(const CheckpointMetadata& metadata, NormalizedModelMetadata& result);
void normalize_mamba2_facts(const CheckpointMetadata& metadata, NormalizedModelMetadata& result);
void normalize_token_vocab_policy(const CheckpointMetadata& metadata, NormalizedModelMetadata& result);
void normalize_norm_type_gate(const CheckpointMetadata& metadata, NormalizedModelMetadata& result);
void normalize_logit_multipliers(const CheckpointMetadata& metadata, NormalizedModelMetadata& result);

/// Attention/recurrent/MoE clusters of `normalize_model_metadata`.
RopePositionFacts normalize_rope_position_facts(const CheckpointMetadata& metadata,
                                                NormalizedModelMetadata& result);
void normalize_qk_norm_facts(const CheckpointMetadata& metadata, NormalizedModelMetadata& result);
void normalize_gated_delta_facts(const CheckpointMetadata& metadata, NormalizedModelMetadata& result);
void normalize_latent_attention_facts(const CheckpointMetadata& metadata,
                                      NormalizedModelMetadata& result);
void normalize_moe_facts(const CheckpointMetadata& metadata, NormalizedModelMetadata& result);
void normalize_xsa_ties(const CheckpointMetadata& metadata, NormalizedModelMetadata& result);
void normalize_defaults(const CheckpointMetadata& metadata, NormalizedModelMetadata& result);
void normalize_position_embedding_gate(const CheckpointMetadata& metadata,
                                       NormalizedModelMetadata& result);
void normalize_rope_pairing(const CheckpointMetadata& metadata, NormalizedModelMetadata& result,
                            const RopePositionFacts& facts);

}
