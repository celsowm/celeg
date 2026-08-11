#include "celeg/backend/cuda/packed.hpp"
#include "celeg/backend/cuda/packed_metadata_cache.hpp"

#include <array>
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
            throw std::runtime_error(
                "packed workspace maxima were derived incorrectly: projection=" +
                std::to_string(requirements.maximum_projection_width) +
                " ffn=" + std::to_string(requirements.maximum_ffn_intermediate) +
                " slots=" + std::to_string(requirements.layer_slots) +
                " pages=" + std::to_string(requirements.page_table_entries));
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

        celeg::PackedMetadataCache metadata_cache(2);
        session.owner = reinterpret_cast<void*>(0x1);
        session.storage_generation_value = 7;
        const std::array<celeg::PackedSessionContext, 1> bound = {session};
        if (!metadata_cache.changed(bound)) {
            throw std::runtime_error("new metadata binding was not detected");
        }
        metadata_cache.update(bound);
        if (metadata_cache.changed(bound)) {
            throw std::runtime_error("unchanged metadata binding was rejected");
        }
        session.storage_generation_value = 8;
        const std::array<celeg::PackedSessionContext, 1> replaced = {session};
        if (!metadata_cache.changed(replaced)) {
            throw std::runtime_error("replaced backing storage was not detected");
        }

        celeg::PackedCompatibilityKey compatibility;
        compatibility.execution_plan_fingerprint = 11;
        compatibility.device_ordinal = 3;
        compatibility.expert_residency_fingerprint = 17;
        const auto same_compatibility = compatibility;
        if (!(compatibility == same_compatibility)) {
            throw std::runtime_error("packed compatibility key is not value comparable");
        }
        compatibility.expert_residency_fingerprint++;
        compatibility.execution_plan_fingerprint = 99;
        if (compatibility == same_compatibility) {
            throw std::runtime_error("packed compatibility key ignored residency identity");
        }

        celeg::PackedSessionContext compatible_a;
        celeg::PackedSessionContext compatible_b;
        celeg::PackedSessionContext incompatible;
        compatible_a.owner = reinterpret_cast<void*>(0x10);
        compatible_b.owner = reinterpret_cast<void*>(0x20);
        incompatible.owner = reinterpret_cast<void*>(0x30);
        compatible_a.compatibility_key = same_compatibility;
        compatible_b.compatibility_key = same_compatibility;
        incompatible.compatibility_key = compatibility;
        const celeg::PackedCompatibilityPolicy policy(11, 3);
        if (!policy.validate_executor(compatible_a).accepted) {
            throw std::runtime_error("matching packed executor identity was rejected");
        }
        if (policy.validate_executor(incompatible).accepted) {
            throw std::runtime_error("mismatched packed executor identity was accepted");
        }
        const std::array<celeg::PackedSessionContext, 2> compatible_batch = {
            compatible_a, compatible_b};
        if (!policy.validate_batch(compatible_a, std::vector<celeg::PackedSessionContext>(
                compatible_batch.begin(), compatible_batch.end())).accepted) {
            throw std::runtime_error("matching packed batch was rejected");
        }
        const std::array<celeg::PackedSessionContext, 2> duplicate_batch = {
            compatible_a, compatible_a};
        if (policy.validate_batch(compatible_a, std::vector<celeg::PackedSessionContext>(
                duplicate_batch.begin(), duplicate_batch.end())).accepted) {
            throw std::runtime_error("duplicate packed session was accepted");
        }
        std::cout << "packed_workspace_test: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "packed_workspace_test: " << error.what() << '\n';
        return 1;
    }
}
