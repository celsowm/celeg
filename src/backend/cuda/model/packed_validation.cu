#include "celeg/backend/cuda/packed.hpp"

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

PackedWorkspaceRequirements PackedWorkspaceRequirements::derive(
    size_t maximum_batch,
    size_t maximum_prefill_tokens,
    size_t page_table_stride,
    const RuntimeTopology& shape) {
    if (maximum_batch == 0 || maximum_prefill_tokens == 0) {
        throw std::invalid_argument("packed capacities must be positive");
    }
    if (shape.hidden <= 0 || shape.vocab_size <= 0 ||
        shape.num_hidden_layers <= 0 || shape.maximum_attention_projection_width() <= 0 ||
        shape.max_feed_forward_intermediate <= 0) {
        throw std::invalid_argument("packed topology has invalid workspace dimensions");
    }
    for (size_t layer = 0; layer < shape.feed_forward_intermediates.size(); ++layer) {
        if (shape.feed_forward_intermediates[layer] <= 0 ||
            static_cast<size_t>(shape.feed_forward_intermediates[layer]) >
                static_cast<size_t>(shape.max_feed_forward_intermediate)) {
            throw std::invalid_argument(
                "packed topology layer " + std::to_string(layer) +
                " exceeds max_feed_forward_intermediate");
        }
    }
    PackedWorkspaceRequirements result;
    result.maximum_batch = maximum_batch;
    result.maximum_prefill_tokens = maximum_prefill_tokens;
    result.maximum_projection_width =
        static_cast<size_t>(shape.maximum_attention_projection_width());
    result.maximum_ffn_intermediate =
        static_cast<size_t>(shape.max_feed_forward_intermediate);
    result.moe_intermediate = static_cast<size_t>(std::max(0, shape.moe_intermediate));
    result.layer_slots = maximum_batch * static_cast<size_t>(shape.num_hidden_layers);
    result.page_table_entries = maximum_prefill_tokens * page_table_stride;
    return result;
}

} // namespace celeg
