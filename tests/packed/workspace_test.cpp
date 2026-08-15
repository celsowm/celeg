#include "celeg/backend/cuda/packed/executor.hpp"
#include "celeg/backend/cuda/packed/metadata_cache.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        celeg::ModelGraph graph;
        graph.hidden = 128;
        graph.layers.resize(2);
        for (int index = 0; index < 2; ++index) {
            celeg::AttentionSpec attention;
            attention.query_heads = index == 0 ? 2 : 3;
            attention.key_value_heads = 1;
            attention.head_dim = 64;
            graph.layers[index].mixer = attention;
            graph.layers[index].feed_forward = celeg::DenseFeedForwardSpec{
                index == 0 ? 128 : 192, celeg::ActivationKind::SwiGLU};
        }
        const celeg::RuntimeTopology shape = celeg::compose_runtime_topology({}, graph);
        celeg::CompiledModelProgram program;
        program.hidden = graph.hidden;
        program.layers.resize(graph.layers.size());
        for (size_t index = 0; index < graph.layers.size(); ++index) {
            program.layers[index].mixer = celeg::CompiledAttentionProgram{
                std::get<celeg::AttentionSpec>(graph.layers[index].mixer), {}};
            program.layers[index].feed_forward =
                celeg::CompiledDenseFeedForwardProgram{
                    index == 0 ? 128 : 192, celeg::ActivationKind::SwiGLU};
        }

        const auto requirements = celeg::PackedWorkspaceRequirements::derive(
            4, 16, 8, shape.exec, program);
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

        celeg::SessionPhase phase = celeg::SessionPhase::Empty;
        int position = 0;
        bool local_kv = true;
        celeg::SessionState identity;
        celeg::PackedSessionContext session;
        session.owner.session_identity = &identity;
        session.session.phase_state = &phase;
        session.session.position_state = &position;
        session.immutable.max_context_value = 16;
        session.session.local_kv_cache_available_state = &local_kv;
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

        celeg::SessionState identity_a;
        celeg::SessionState identity_b;
        celeg::SessionState identity_c;
        celeg::PackedSessionContext compatible_a;
        celeg::PackedSessionContext compatible_b;
        celeg::PackedSessionContext incompatible;
        compatible_a.owner.session_identity = &identity_a;
        compatible_b.owner.session_identity = &identity_b;
        incompatible.owner.session_identity = &identity_c;
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
