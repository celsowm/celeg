#include "celeg/backend/cuda/packed/commit.hpp"
#include "support/assertions.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

bool no_segmented_attention(const void*, int) { return false; }

celeg::PackedSessionContext make_session(
    celeg::SessionPhase& phase,
    int& position,
    bool& local_kv,
    bool& active_segmented,
    celeg::GenerationConfig& generation,
    celeg::RuntimeMetrics& metrics,
    celeg::PinnedBuffer<int32_t>& sampled_host) {
    celeg::PackedSessionContext session;
    session.owner = reinterpret_cast<void*>(0x1);
    session.phase_state = &phase;
    session.position_state = &position;
    session.max_context_value = 32;
    session.local_kv_cache_available_state = &local_kv;
    session.active_segmented_attention_state = &active_segmented;
    session.generation_state = &generation;
    session.metrics_state = &metrics;
    session.sampled_host_state = &sampled_host;
    session.segmented_attention = no_segmented_attention;
    return session;
}

} // namespace

int main() {
    try {
        celeg::SessionPhase phase = celeg::SessionPhase::Ready;
        int position = 4;
        bool local_kv = true;
        bool active_segmented = false;
        celeg::GenerationConfig generation;
        celeg::RuntimeMetrics metrics;
        celeg::PinnedBuffer<int32_t> sampled_host(1);
        sampled_host.data()[0] = 9;
        const auto session = make_session(phase, position, local_kv,
                                          active_segmented, generation, metrics,
                                          sampled_host);
        const std::vector<celeg::PackedSessionContext> sessions{session};
        const std::vector<int32_t> sampled{17};
        std::vector<int32_t> output(1);
        celeg::commit_packed_decode(sessions, sampled, 2.5, output);
        CELEG_TEST_CHECK(output[0] == 17);
        CELEG_TEST_CHECK(sampled_host.data()[0] == 17);
        CELEG_TEST_CHECK(position == 5);
        CELEG_TEST_CHECK(metrics.decoded_tokens == 1);
        CELEG_TEST_CHECK(metrics.cumulative_decode_ms == 2.5);

        position = 5;
        phase = celeg::SessionPhase::Prefilling;
        const std::vector<int32_t> tokens{21, 22};
        const std::vector<int32_t> final_rows{1};
        const std::vector<celeg::PackedPrefillRow> rows{{0, 2, 1}};
        celeg::commit_packed_prefill(sessions, tokens, final_rows, rows);
        CELEG_TEST_CHECK(sampled_host.data()[0] == 22);
        CELEG_TEST_CHECK(position == 7);
        CELEG_TEST_CHECK(phase == celeg::SessionPhase::Ready);
        CELEG_TEST_CHECK(metrics.prefill_tokens == 2);

        position = 7;
        bool rejected = false;
        try {
            celeg::commit_packed_decode(sessions, {}, 1.0, output);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        CELEG_TEST_CHECK(rejected);
        CELEG_TEST_CHECK(position == 7);

        rejected = false;
        try {
            celeg::commit_packed_prefill(
                sessions, tokens, std::vector<int32_t>{99}, rows);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        CELEG_TEST_CHECK(rejected);
        CELEG_TEST_CHECK(position == 7);
        CELEG_TEST_CHECK(phase == celeg::SessionPhase::Ready);
        CELEG_TEST_CHECK(metrics.prefill_tokens == 2);

        std::cout << "packed_commit_test: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "packed_commit_test: " << error.what() << '\n';
        return 1;
    }
}
