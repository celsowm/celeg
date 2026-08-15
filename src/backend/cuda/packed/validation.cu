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
    if (program.hidden <= 0 ||
        shape.num_hidden_layers <= 0 ||
        std::max(shape.maximum_attention_projection_width(),
                 shape.maximum_mamba_projection_width()) <= 0 ||
        shape.max_feed_forward_intermediate <= 0) {
        throw std::invalid_argument("packed topology has invalid workspace dimensions");
    }
    PackedWorkspaceRequirements result;
    result.maximum_batch = maximum_batch;
    result.maximum_prefill_tokens = maximum_prefill_tokens;
    result.maximum_projection_width = static_cast<size_t>(std::max(
        shape.maximum_attention_projection_width(),
        shape.maximum_mamba_projection_width()));
    result.maximum_mamba_projection_width = static_cast<size_t>(std::max(
        1, shape.maximum_mamba_projection_width()));
    result.maximum_mamba_intermediate = static_cast<size_t>(std::max(
        1, shape.mamba2_intermediate));
    result.maximum_ffn_intermediate =
        static_cast<size_t>(shape.max_feed_forward_intermediate);
    for (const CompiledLayerProgram& layer : program.layers) {
        if (const auto* moe = std::get_if<MoeLayerProgram>(&layer.feed_forward)) {
            result.moe_intermediate = std::max(
                result.moe_intermediate,
                static_cast<size_t>(moe->routed.mlp.intermediate_size));
        }
    }
    result.layer_slots = maximum_batch * static_cast<size_t>(shape.num_hidden_layers);
    result.page_table_entries = maximum_prefill_tokens * page_table_stride;
    return result;
}

} // namespace celeg
