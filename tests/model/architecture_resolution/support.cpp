#include "support.hpp"

#include "celeg/runtime/context.hpp"
#include "support/assertions.hpp"

#include <utility>

namespace celeg::architecture_resolution_test {

celeg::CheckpointMetadata structural_metadata(std::string model_type) {
    celeg::CheckpointMetadata metadata;
    metadata.values["model_type"] = std::move(model_type);
    metadata.values["hidden_size"] = int64_t(576);
    metadata.values["intermediate_size"] = int64_t(1728);
    metadata.values["num_hidden_layers"] = int64_t(1);
    metadata.values["num_attention_heads"] = int64_t(9);
    metadata.values["num_key_value_heads"] = int64_t(3);
    metadata.values["head_dim"] = int64_t(64);
    metadata.values["vocab_size"] = int64_t(32770);
    metadata.values["max_position_embeddings"] = int64_t(8192);
    metadata.values["bos_token_id"] = int64_t(0);
    metadata.values["eos_token_id"] = int64_t(0);
    metadata.values["pad_token_id"] = int64_t(1);
    metadata.values["rms_norm_eps"] = 1.0e-6;
    metadata.values["rope_theta"] = 100000.0;
    metadata.values["tie_word_embeddings"] = true;
    return metadata;
}

celeg::CheckpointMetadata postnorm_evidence_metadata(std::string model_type) {
    celeg::CheckpointMetadata metadata;
    metadata.values["model_type"] = std::move(model_type);
    metadata.values["hidden_size"] = int64_t(8);
    metadata.values["intermediate_size"] = int64_t(16);
    metadata.values["num_hidden_layers"] = int64_t(4);
    metadata.values["num_attention_heads"] = int64_t(2);
    metadata.values["num_key_value_heads"] = int64_t(2);
    metadata.values["head_dim"] = int64_t(4);
    metadata.values["vocab_size"] = int64_t(32);
    metadata.values["max_position_embeddings"] = int64_t(32768);
    metadata.values["bos_token_id"] = int64_t(1);
    metadata.values["eos_token_id"] = int64_t(2);
    metadata.values["pad_token_id"] = int64_t(0);
    metadata.values["norm_eps"] = 1.0e-6;
    metadata.values["rope_theta"] = 500000.0;
    metadata.values["tie_word_embeddings"] = false;
    metadata.values["use_qk_norm"] = true;
    metadata.values["use_pre_attn_norm"] = false;
    metadata.values["use_post_attn_norm"] = true;
    metadata.values["use_pre_mlp_norm"] = false;
    metadata.values["use_post_mlp_norm"] = true;
    metadata.values["layer_types"] = std::vector<std::string>{
        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"};
    metadata.values["layer_layouts"] = std::vector<std::string>{
        "decoder_postnorm", "decoder_postnorm", "decoder_postnorm", "decoder_postnorm"};
    metadata.values["sliding_window"] = int64_t(4096);
    metadata.values["rope_scaling.rope_type"] = std::string("yarn");
    metadata.values["rope_scaling.factor"] = 8.0;
    metadata.values["rope_scaling.original_max_position_embeddings"] = int64_t(8192);
    metadata.values["rope_scaling.attention_factor"] = 1.2079441541679836;
    metadata.values["rope_scaling.beta_fast"] = 32.0;
    metadata.values["rope_scaling.beta_slow"] = 1.0;
    return metadata;
}

celeg::CheckpointMetadata hybrid_kda_mla_metadata(std::string model_type) {
    celeg::CheckpointMetadata metadata;
    metadata.values["model_type"] = std::move(model_type);
    metadata.values["hidden_size"] = int64_t(8);
    metadata.values["num_hidden_layers"] = int64_t(4);
    metadata.values["num_attention_heads"] = int64_t(2);
    metadata.values["head_dim"] = int64_t(4);
    metadata.values["v_head_dim"] = int64_t(4);
    metadata.values["q_lora_rank"] = int64_t(4);
    metadata.values["kv_lora_rank"] = int64_t(2);
    metadata.values["qk_nope_head_dim"] = int64_t(4);
    metadata.values["qk_rope_head_dim"] = int64_t(2);
    metadata.values["vocab_size"] = int64_t(32);
    metadata.values["max_position_embeddings"] = int64_t(128);
    metadata.values["bos_token_id"] = int64_t(1);
    metadata.values["eos_token_id"] = int64_t(2);
    metadata.values["pad_token_id"] = int64_t(0);
    metadata.values["norm_eps"] = 1.0e-6;
    metadata.values["rope_theta"] = 100000.0;
    metadata.values["tie_word_embeddings"] = false;
    metadata.values["short_conv_kernel_size"] = int64_t(4);
    metadata.values["kda_safe_gate"] = true;
    metadata.values["kda_lower_bound"] = -5.0;
    /// Ling's exact hybrid key set: schedule + projection style + gate
    /// granularity are consumed and verified, the rest are proven inert.
    metadata.values["layer_group_size"] = int64_t(4);
    metadata.values["no_kda_lora"] = true;
    metadata.values["gated_attention_proj_granularity_type"] = std::string("head_wise");
    metadata.values["linear_silu"] = true;
    metadata.values["group_norm_size"] = int64_t(1);
    metadata.values["num_kv_heads_for_linear_attn"] = int64_t(0);
    metadata.values["embedding_dropout"] = 0.0;
    metadata.values["use_mla_nope"] = false;
    return metadata;
}

celeg::ResolvedModel resolve_postnorm_evidence(
    const celeg::ArchitectureCatalog& catalog,
    std::string model_type) {
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = postnorm_evidence_metadata(std::move(model_type));
    checkpoint.repository = std::make_shared<PostnormEvidenceRepository>();
    const auto& architecture = catalog.select(checkpoint.metadata);
    CELEG_TEST_CHECK(architecture.id() == "automatic");
    return architecture.resolve(checkpoint);
}

celeg::ResolvedModel resolve_hybrid_kda_mla(
    const celeg::ArchitectureCatalog& catalog,
    celeg::CheckpointMetadata metadata) {
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = std::move(metadata);
    checkpoint.repository = std::make_shared<HybridKdaMlaRepository>();
    const auto& architecture = catalog.select(checkpoint.metadata);
    CELEG_TEST_CHECK(architecture.id() == "automatic");
    return architecture.resolve(checkpoint);
}

bool hybrid_resolve_fails_with(celeg::CheckpointMetadata metadata,
                               celeg::ResolutionFailureKind expected) {
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = std::move(metadata);
    checkpoint.repository = std::make_shared<HybridKdaMlaRepository>();
    try {
        const auto runtime = celeg::create_builtin_runtime_context();
        (void)runtime->architectures().select(checkpoint.metadata).resolve(checkpoint);
    } catch (const celeg::ResolutionError& error) {
        return error.kind() == expected;
    }
    return false;
}

bool equivalent_weight_plan(const celeg::WeightPlan& a, const celeg::WeightPlan& b) {
    if (a.requests.size() != b.requests.size()) return false;
    for (std::size_t index = 0; index < a.requests.size(); ++index) {
        const auto& lhs = a.requests[index];
        const auto& rhs = b.requests[index];
        if (lhs.role != rhs.role || lhs.layer != rhs.layer || lhs.expert != rhs.expert ||
            lhs.expected_shape != rhs.expected_shape || lhs.source_name != rhs.source_name ||
            lhs.physical_layer != rhs.physical_layer ||
            lhs.norm_weight_kind != rhs.norm_weight_kind) {
            return false;
        }
    }
    return true;
}

bool has_weight_role(const celeg::WeightPlan& plan, celeg::TensorRole role, int layer) {
    for (const auto& request : plan.requests) {
        if (request.role == role && request.layer == layer) return true;
    }
    return false;
}

celeg::ResolvedModel resolve_structural_identity(const celeg::ArchitectureCatalog& catalog,
                                                 std::string model_type) {
    auto metadata = structural_metadata(std::move(model_type));
    metadata.repository_hint = "synthetic/identity-invariance";
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = std::move(metadata);
    checkpoint.repository = std::make_shared<GptxRepository>();
    const auto& architecture = catalog.select(checkpoint.metadata);
    CELEG_TEST_CHECK(architecture.id() == "automatic");
    return architecture.resolve(checkpoint);
}

celeg::ResolvedModel resolve_norm_layout(const celeg::ArchitectureCatalog& catalog,
                                         NormFixtureLayout norms) {
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = structural_metadata("unknown_norm_layout");
    checkpoint.repository = std::make_shared<GptxRepository>(norms);
    const auto& architecture = catalog.select(checkpoint.metadata);
    CELEG_TEST_CHECK(architecture.id() == "automatic");
    return architecture.resolve(checkpoint);
}

bool resolution_fails_with(const celeg::ArchitectureCatalog& catalog,
                           celeg::CheckpointMetadata metadata,
                           NormFixtureLayout norms,
                           bool query_key_norms,
                           celeg::ResolutionFailureKind expected) {
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = std::move(metadata);
    checkpoint.repository = std::make_shared<GptxRepository>(norms, query_key_norms);
    try {
        (void)catalog.select(checkpoint.metadata).resolve(checkpoint);
    } catch (const celeg::ResolutionError& error) {
        return error.kind() == expected;
    }
    return false;
}

bool normalize_fails_with(celeg::CheckpointMetadata metadata,
                          celeg::ResolutionFailureKind expected) {
    try {
        (void)celeg::normalize_model_metadata(metadata);
    } catch (const celeg::ResolutionError& error) {
        return error.kind() == expected;
    }
    return false;
}

bool inference_input_fails_with(celeg::CheckpointMetadata metadata,
                                celeg::ResolutionFailureKind expected) {
    celeg::CheckpointView checkpoint;
    checkpoint.metadata = std::move(metadata);
    checkpoint.repository = std::make_shared<GptxRepository>();
    try {
        (void)celeg::build_inference_input(checkpoint);
    } catch (const celeg::ResolutionError& error) {
        return error.kind() == expected;
    }
    return false;
}

}
