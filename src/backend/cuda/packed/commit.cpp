#include "celeg/backend/cuda/packed/commit.hpp"

#include <stdexcept>

namespace celeg {

void commit_packed_decode(std::span<const PackedSessionContext> sessions,
                          std::span<const int32_t> sampled,
                          double gpu_elapsed_ms,
                          std::span<int32_t> output) {
    if (sampled.size() < sessions.size() || output.size() < sessions.size()) {
        throw std::invalid_argument("packed decode commit buffers are too small");
    }
    for (size_t row = 0; row < sessions.size(); ++row) {
        const PackedSessionContext& model = sessions[row];
        output[row] = sampled[row];
        model.set_sampled_host_value(sampled[row]);
        model.set_position(model.position() + 1);
        model.metrics().cumulative_decode_ms += gpu_elapsed_ms;
        model.metrics().decoded_tokens += 1;
    }
}

void commit_packed_prefill(std::span<const PackedSessionContext> sessions,
                           std::span<const int32_t> explicit_tokens,
                           std::span<const int32_t> final_rows,
                           std::span<const PackedPrefillRow> rows) {
    if (rows.size() != sessions.size() || final_rows.size() < sessions.size()) {
        throw std::invalid_argument("packed prefill commit rows are inconsistent");
    }
    for (size_t request = 0; request < sessions.size(); ++request) {
        if (final_rows[request] < 0 ||
            static_cast<size_t>(final_rows[request]) >= explicit_tokens.size()) {
            throw std::invalid_argument("packed prefill final row is out of range");
        }
        const PackedSessionContext& model = sessions[request];
        const PackedPrefillRow& descriptor = rows[request];
        model.set_sampled_host_value(explicit_tokens[static_cast<size_t>(final_rows[request])]);
        model.set_position(model.position() + static_cast<int>(descriptor.token_count));
        model.metrics().prefill_tokens += descriptor.token_count;
        model.set_phase(descriptor.finalize != 0 ? SessionPhase::Ready : SessionPhase::Prefilling);
        if (model.phase() == SessionPhase::Ready) {
            model.set_active_segmented_attention(model.use_segmented_attention(model.position()));
        }
    }
}

} // namespace celeg
