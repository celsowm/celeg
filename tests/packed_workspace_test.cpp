#include "celeg/backend/cuda/packed.hpp"

#include <iostream>
#include <stdexcept>

int main() {
    try {
        celeg::RuntimeTopology shape;
        shape.hidden = 128;
        shape.vocab_size = 256;
        shape.num_hidden_layers = 2;
        shape.max_feed_forward_intermediate = 192;
        shape.moe_intermediate = 64;
        shape.attention_layouts.resize(2);
        shape.attention_layouts[0].query_heads = 2;
        shape.attention_layouts[0].key_value_heads = 1;
        shape.attention_layouts[0].head_dim = 64;
        shape.attention_layouts[1] = shape.attention_layouts[0];
        shape.attention_layouts[1].query_heads = 3;
        shape.feed_forward_intermediates = {128, 192};

        const auto requirements = celeg::PackedWorkspaceRequirements::derive(
            4, 16, 8, shape);
        if (requirements.maximum_projection_width != 320 ||
            requirements.maximum_ffn_intermediate != 192 ||
            requirements.layer_slots != 8 ||
            requirements.page_table_entries != 128) {
            throw std::runtime_error("packed workspace maxima were derived incorrectly");
        }

        shape.feed_forward_intermediates[1] = 256;
        bool rejected = false;
        try {
            (void)celeg::PackedWorkspaceRequirements::derive(4, 16, 8, shape);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("oversized layer was not rejected");

        celeg::SessionPhase phase = celeg::SessionPhase::Empty;
        int position = 0;
        bool local_kv = true;
        celeg::PackedSessionContext session;
        session.phase_state = &phase;
        session.position_state = &position;
        session.max_context_value = 16;
        session.local_kv_cache_available_state = &local_kv;
        const celeg::PackedBatchValidator validator;
        const auto prefill = validator.validate_session(
            session, celeg::PackedOperation::Prefill,
            celeg::PackedExecutorCapabilities{true});
        if (!prefill.accepted) throw std::runtime_error("empty session rejected for prefill");
        const auto before_decode = validator.validate_session(
            session, celeg::PackedOperation::Decode,
            celeg::PackedExecutorCapabilities{true});
        if (before_decode.accepted || before_decode.reason.empty()) {
            throw std::runtime_error("incomplete session accepted for decode");
        }
        phase = celeg::SessionPhase::Ready;
        const auto decode = validator.validate_session(
            session, celeg::PackedOperation::Decode,
            celeg::PackedExecutorCapabilities{true});
        if (!decode.accepted) throw std::runtime_error("ready session rejected for decode");
        phase = celeg::SessionPhase::DecodePending;
        if (validator.validate_session(session, celeg::PackedOperation::Decode,
                                       celeg::PackedExecutorCapabilities{true}).accepted) {
            throw std::runtime_error("pending decode was accepted");
        }
        std::cout << "packed_workspace_test: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "packed_workspace_test: " << error.what() << '\n';
        return 1;
    }
}
