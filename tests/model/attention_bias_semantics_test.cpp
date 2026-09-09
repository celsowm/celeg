#include "celeg/attention/bias_semantics.hpp"
#include "celeg/backend/cpu/paged_kv.hpp"
#include "support/assertions.hpp"

#include <array>
#include <cmath>

namespace {

void alibi_matches_canonical_semantics() {
    constexpr std::array<float, 2> slopes{0.5f, 0.125f};
    const celeg::CpuAttentionBias bias = celeg::CpuAttentionBias::lower(
        celeg::AlibiBiasSpec{{slopes.begin(), slopes.end()}});

    for (int head = 0; head < 2; ++head) {
        for (const auto [query, key] :
             std::array<std::pair<int, int>, 6>{{
                 {0, 0}, {8, 7}, {8, 9}, {64, 127}, {127, 64}, {31, 15}}}) {
            const float expected = celeg::attention_semantics::alibi_bias(
                slopes[static_cast<size_t>(head)], query, key);
            CELEG_TEST_CHECK(std::abs(
                bias.score(head, query, key) - expected) < 1.0e-6f);
        }
    }

    CELEG_TEST_CHECK(std::abs(bias.score(0, 8, 7) - bias.score(0, 8, 9)) < 1.0e-6f);
}

void relative_bias_matches_canonical_buckets() {
    constexpr int bucket_count = 32;
    constexpr int max_distance = 128;
    std::array<float, bucket_count> values{};
    for (int bucket = 0; bucket < bucket_count; ++bucket) {
        values[static_cast<size_t>(bucket)] = static_cast<float>(bucket);
    }

    for (bool bidirectional : {false, true}) {
        const celeg::CpuAttentionBias bias = celeg::CpuAttentionBias::lower(
            celeg::RelativePositionBiasSpec{
                bucket_count, max_distance, bidirectional},
            values, 1);
        for (const auto [query, key] :
             std::array<std::pair<int, int>, 8>{{
                 {0, 0}, {7, 0}, {31, 15}, {127, 0},
                 {8, 9}, {8, 7}, {64, 127}, {127, 64}}}) {
            const int expected_bucket =
                celeg::attention_semantics::relative_position_bucket(
                    query, key, bucket_count, max_distance, bidirectional);
            CELEG_TEST_CHECK(bias.score(0, query, key) ==
                             static_cast<float>(expected_bucket));
        }
    }
}

}

int main() {
    alibi_matches_canonical_semantics();
    relative_bias_matches_canonical_buckets();
    return 0;
}
