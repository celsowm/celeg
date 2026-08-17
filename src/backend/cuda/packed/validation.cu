#include "celeg/backend/cuda/packed/executor.hpp"
#include "celeg/detail/model/layer_state.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace celeg {

PackedEligibility validate_packed_batch_common(
    const std::vector<PackedSessionContext>& sessions,
    size_t maximum_batch, PackedOperation operation) {
    (void)operation;
    PackedEligibility result;
    if (sessions.empty()) {
        result.reason = "packed batch is empty";
        return result;
    }
    if (sessions.size() > maximum_batch) {
        result.reason = "packed batch exceeds executor capacity";
        return result;
    }
    for (const PackedSessionContext& session : sessions) {
        if (session.owner == nullptr) {
            result.reason = "null packed session";
            return result;
        }
        for (const Layer& layer : session.layers()) {
            if (const auto* attention = as_attention(layer);
                attention && attention->layout.uses_latent_state()) {
                result.reason =
                    "packed CUDA execution currently requires ordinary KV attention";
                return result;
            }
        }
    }
    result.accepted = true;
    return result;
}

PackedEligibility PackedBatchValidator::validate_session(
    const PackedSessionContext& model,
    PackedOperation operation,
    PackedExecutorCapabilities capabilities) const {
    PackedEligibility result;
    if (model.phase() == SessionPhase::DecodePending) {
        result.reason = "session already has a pending decode";
        return result;
    }
    if (operation == PackedOperation::Decode &&
        model.phase() != SessionPhase::Ready) {
        result.reason = "session has not completed prefill";
        return result;
    }
    if (model.position() >= model.max_context()) {
        result.reason = "session reached max_context";
        return result;
    }
    if (!capabilities.physical_paged_kv && operation == PackedOperation::Decode &&
        model.use_segmented_attention(model.position())) {
        result.reason = "segmented attention requires physical paged KV in packed decode";
        return result;
    }
    if (!capabilities.physical_paged_kv && !model.local_kv_cache_available()) {
        result.reason = "session released its local KV cache";
        return result;
    }
    result.accepted = true;
    return result;
}

PackedEligibility PackedCompatibilityPolicy::validate_executor(
    const PackedSessionContext& session) const {
    PackedEligibility result;
    if (session.compatibility_key.execution_plan_fingerprint !=
            execution_plan_fingerprint_ ||
        session.compatibility_key.device_ordinal != device_ordinal_) {
        result.reason = "packed session plan does not match executor device policy";
        return result;
    }
    result.accepted = true;
    return result;
}

PackedEligibility PackedCompatibilityPolicy::validate_batch(
    const PackedSessionContext& reference,
    const std::vector<PackedSessionContext>& sessions) const {
    PackedEligibility result;
    for (size_t row = 0; row < sessions.size(); ++row) {
        for (size_t previous = 0; previous < row; ++previous) {
            if (sessions[previous].owner == sessions[row].owner) {
                result.reason = "duplicate session in packed batch";
                return result;
            }
        }
        if (!(reference.compatibility_key == sessions[row].compatibility_key)) {
            result.reason = "sessions use incompatible packed execution keys";
            return result;
        }
    }
    result.accepted = true;
    return result;
}

PackedWorkspaceRequirements PackedWorkspaceRequirements::derive(
    size_t maximum_batch,
    size_t maximum_prefill_tokens,
    size_t page_table_stride,
    const ExecutionTopology& shape,
    const CompiledModelProgram& program) {
    if (maximum_batch == 0 || maximum_prefill_tokens == 0) {
        throw std::invalid_argument("packed capacities must be positive");
    }
    if (program.hidden <= 0 || shape.num_hidden_layers <= 0) {
        throw std::invalid_argument("packed topology has invalid workspace dimensions");
    }
    PackedWorkspaceRequirements result;
    result.maximum_batch = maximum_batch;
    result.maximum_prefill_tokens = maximum_prefill_tokens;
    for (const CompiledLayerProgram& layer : program.layers) {
        std::visit([&](const auto& mixer) {
            using Mixer = std::decay_t<decltype(mixer)>;
            if constexpr (std::is_same_v<Mixer, CompiledAttentionProgram>) {
                result.maximum_projection_width = std::max(
                    result.maximum_projection_width,
                    static_cast<size_t>(mixer.semantics.projection_width()));
                result.attention_query_heads = std::max(
                    result.attention_query_heads, mixer.semantics.query_heads);
                result.attention_head_dim = std::max(
                    result.attention_head_dim, mixer.semantics.head_dim);
            } else if constexpr (std::is_same_v<Mixer, ShortConvolutionSpec>) {
            } else if constexpr (std::is_same_v<Mixer, GatedDeltaNetSpec>) {
                result.max_gated_delta_qkv = std::max(
                    result.max_gated_delta_qkv, static_cast<size_t>(mixer.qkv_width()));
                result.max_gated_delta_output = std::max(
                    result.max_gated_delta_output, static_cast<size_t>(mixer.value_width()));
                result.max_gated_delta_gate = std::max(
                    result.max_gated_delta_gate,
                    static_cast<size_t>(std::max(mixer.value_heads, mixer.decay_width())));
            } else if constexpr (std::is_same_v<Mixer, Mamba2Spec>) {
                result.maximum_mamba_projection_width = std::max(
                    result.maximum_mamba_projection_width,
                    static_cast<size_t>(2 * mixer.intermediate_size +
                                        2 * mixer.group_count * mixer.state_size + mixer.num_heads));
                result.maximum_mamba_intermediate = std::max(
                    result.maximum_mamba_intermediate, static_cast<size_t>(mixer.intermediate_size));
                result.maximum_projection_width = std::max(
                    result.maximum_projection_width, result.maximum_mamba_projection_width);
            } else if constexpr (std::is_same_v<Mixer, MlpBlockSpec>) {
                result.maximum_ffn_intermediate = std::max(
                    result.maximum_ffn_intermediate, static_cast<size_t>(mixer.intermediate_size));
            }
        }, layer.mixer);

        std::visit([&](const auto& ff) {
            using FF = std::decay_t<decltype(ff)>;
            if constexpr (std::is_same_v<FF, std::monostate>) {
                return;
            } else if constexpr (std::is_same_v<FF, CompiledDenseFeedForwardProgram>) {
                result.maximum_ffn_intermediate = std::max(
                    result.maximum_ffn_intermediate, static_cast<size_t>(ff.intermediate_size));
            } else if constexpr (std::is_same_v<FF, MoeLayerProgram>) {
                result.moe_intermediate = std::max(
                    result.moe_intermediate, static_cast<size_t>(ff.routed.mlp.intermediate_size));
                result.maximum_ffn_intermediate = std::max(
                    result.maximum_ffn_intermediate, static_cast<size_t>(ff.routed.mlp.intermediate_size));
            }
        }, layer.feed_forward);
    }
    result.maximum_mamba_projection_width = std::max<size_t>(1, result.maximum_mamba_projection_width);
    result.maximum_mamba_intermediate = std::max<size_t>(1, result.maximum_mamba_intermediate);
    result.max_gated_delta_qkv = std::max<size_t>(1, result.max_gated_delta_qkv);
    result.max_gated_delta_output = std::max<size_t>(1, result.max_gated_delta_output);
    result.max_gated_delta_gate = std::max<size_t>(1, result.max_gated_delta_gate);
    result.maximum_projection_width = std::max<size_t>(1, result.maximum_projection_width);
    result.maximum_ffn_intermediate = std::max<size_t>(1, result.maximum_ffn_intermediate);
    result.layer_slots = maximum_batch * static_cast<size_t>(shape.num_hidden_layers);
    result.page_table_entries = maximum_prefill_tokens * page_table_stride;
    return result;
}

}
