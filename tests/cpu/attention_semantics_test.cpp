#include "operators/attention.hpp"
#include "celeg/backend/cpu/kernels.hpp"
#include "celeg/backend/cpu/paged_kv.hpp"
#include "support/assertions.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

bool close(float actual, float expected, float tolerance = 1.0e-5f) {
    return std::abs(actual - expected) <= tolerance;
}

void test_current_value_orthogonalization() {
    celeg::AttentionSpec layout;
    layout.query_heads = 4;
    layout.key_value_heads = 2;
    layout.head_dim = 2;
    layout.output_transform = celeg::OrthogonalizeCurrentValueSpec{1.0e-6f};

    float output[] = {
        1.0f, 1.0f,
        2.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 2.0f,
    };
    const float current_value[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };

    celeg::apply_cpu_attention_output_transform(layout, output, current_value);

    CELEG_TEST_CHECK(close(output[0], 0.0f));
    CELEG_TEST_CHECK(close(output[1], 1.0f));
    CELEG_TEST_CHECK(close(output[2], 0.0f));
    CELEG_TEST_CHECK(close(output[3], 0.0f));
    CELEG_TEST_CHECK(close(output[4], 1.0f));
    CELEG_TEST_CHECK(close(output[5], 0.0f));
    CELEG_TEST_CHECK(close(output[6], 0.0f));
    CELEG_TEST_CHECK(close(output[7], 0.0f));

    for (int query_head = 0; query_head < layout.query_heads; ++query_head) {
        const int value_head = query_head / (layout.query_heads / layout.key_value_heads);
        const float* value = current_value + value_head * layout.head_dim;
        const float* transformed = output + query_head * layout.head_dim;
        const float dot = transformed[0] * value[0] + transformed[1] * value[1];
        CELEG_TEST_CHECK(close(dot, 0.0f));
    }
}

void test_adjacent_pair_rope() {
    celeg::RopePositionSpec rope;
    rope.theta = 10000.0;
    rope.rotary_fraction = 1.0;
    rope.pairing = celeg::RopePairingKind::AdjacentPairs;

    float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
    celeg::cpu_rope(values, 1, 4, 1, rope);

    const float c0 = std::cos(1.0f);
    const float s0 = std::sin(1.0f);
    const float second_frequency = std::pow(10000.0f, -0.5f);
    const float c1 = std::cos(second_frequency);
    const float s1 = std::sin(second_frequency);

    CELEG_TEST_CHECK(close(values[0], 1.0f * c0 - 2.0f * s0));
    CELEG_TEST_CHECK(close(values[1], 2.0f * c0 + 1.0f * s0));
    CELEG_TEST_CHECK(close(values[2], 3.0f * c1 - 4.0f * s1));
    CELEG_TEST_CHECK(close(values[3], 4.0f * c1 + 3.0f * s1));
}

void test_bidirectional_pattern_reads_future_keys() {
    celeg::CpuAttentionPattern pattern;
    pattern.storage = celeg::BidirectionalPattern{};

    CELEG_TEST_CHECK(pattern.allows(0, 0));
    CELEG_TEST_CHECK(pattern.allows(0, 1));
    CELEG_TEST_CHECK(pattern.allows(1, 0));
    CELEG_TEST_CHECK(pattern.allows(2, 7));
    CELEG_TEST_CHECK(pattern.first_candidate(3) == 0);
    CELEG_TEST_CHECK(pattern.may_read_future(0, 8));
    CELEG_TEST_CHECK(pattern.may_read_future(7, 8));
}

void test_prefix_lm_pattern_boundaries() {
    celeg::CpuAttentionPattern pattern;
    pattern.storage = celeg::PrefixLmPattern{4};

    CELEG_TEST_CHECK(pattern.allows(0, 0));
    CELEG_TEST_CHECK(pattern.allows(0, 3));
    CELEG_TEST_CHECK(!pattern.allows(0, 4));
    CELEG_TEST_CHECK(pattern.allows(2, 3));
    CELEG_TEST_CHECK(!pattern.allows(2, 5));
    CELEG_TEST_CHECK(pattern.allows(4, 0));
    CELEG_TEST_CHECK(pattern.allows(4, 4));
    CELEG_TEST_CHECK(!pattern.allows(4, 5));
    CELEG_TEST_CHECK(pattern.allows(7, 6));
    CELEG_TEST_CHECK(!pattern.allows(7, 8));
    CELEG_TEST_CHECK(pattern.first_candidate(0) == 0);
    CELEG_TEST_CHECK(pattern.first_candidate(7) == 0);
    CELEG_TEST_CHECK(pattern.may_read_future(0, 8));
    CELEG_TEST_CHECK(pattern.may_read_future(3, 8));
    CELEG_TEST_CHECK(!pattern.may_read_future(4, 8));

    celeg::CpuAttentionPattern full_prefix;
    full_prefix.storage = celeg::PrefixLmPattern{8};
    CELEG_TEST_CHECK(!full_prefix.may_read_future(0, 8));
    CELEG_TEST_CHECK(full_prefix.allows(0, 7));
}

void test_query_key_norm_uses_same_query_scale() {
    constexpr int kHeadDim = 64;
    constexpr int kHeads = 2;
    constexpr int kWidth = kHeads * kHeadDim;

    celeg::RopePositionSpec rope;
    rope.theta = 10000.0;
    rope.rotary_fraction = 1.0;
    rope.pairing = celeg::RopePairingKind::SplitHalf;

    celeg::AttentionSpec plain;
    plain.query_heads = kHeads;
    plain.key_value_heads = kHeads;
    plain.head_dim = kHeadDim;
    plain.query_scale = 0.125f;
    plain.position = rope;

    celeg::AttentionSpec normed = plain;
    celeg::NormSpec norm;
    norm.epsilon = 0.0f;
    norm.granularity = celeg::NormGranularity::PerHead;
    normed.query_norm = norm;

    celeg::CpuCompiledModel::AttentionWeights weights;
    weights.q_norm.assign(kHeadDim, 1.0f);

    std::vector<float> plain_query(kWidth);
    for (int i = 0; i < kWidth; ++i) plain_query[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    std::vector<float> normed_query = plain_query;

    const std::array<int32_t, 3> position{0, 0, 0};
    celeg::apply_cpu_attention_qk(plain, weights, plain_query.data(), nullptr, 0, position);
    celeg::apply_cpu_attention_qk(normed, weights, normed_query.data(), nullptr, 0, position);

    for (int i = 0; i < kWidth; ++i) {
        CELEG_TEST_CHECK(close(plain_query[i], (i % 2 == 0) ? 1.0f : -1.0f));
        CELEG_TEST_CHECK(close(normed_query[i], plain_query[i]));
    }
}

void test_query_only_norm_with_key_projection() {
    celeg::AttentionSpec layout;
    layout.query_heads = 1;
    layout.key_value_heads = 1;
    layout.head_dim = 4;
    layout.query_scale = 0.5f;
    layout.position = celeg::NoPositionEncodingSpec{};
    celeg::NormSpec norm;
    norm.epsilon = 0.0f;
    norm.granularity = celeg::NormGranularity::PerHead;
    layout.query_norm = norm;

    celeg::CpuCompiledModel::AttentionWeights weights;
    weights.q_norm.assign(4, 1.0f);
    weights.k.segments.emplace_back(celeg::CpuInt8Matrix{});

    float query[] = {2.0f, 2.0f, 2.0f, 2.0f};
    float key[] = {3.0f, 3.0f, 3.0f, 3.0f};
    const std::array<int32_t, 3> position{0, 0, 0};
    celeg::apply_cpu_attention_qk(layout, weights, query, key, 0, position);

    for (float value : query) CELEG_TEST_CHECK(close(value, 1.0f));
    for (float value : key) CELEG_TEST_CHECK(close(value, 3.0f));
}

void test_key_only_norm_without_query_norm() {
    celeg::AttentionSpec layout;
    layout.query_heads = 1;
    layout.key_value_heads = 1;
    layout.head_dim = 4;
    layout.query_scale = 0.5f;
    layout.position = celeg::NoPositionEncodingSpec{};
    celeg::NormSpec norm;
    norm.epsilon = 0.0f;
    norm.granularity = celeg::NormGranularity::PerHead;
    layout.key_norm = norm;

    celeg::CpuCompiledModel::AttentionWeights weights;
    weights.k_norm.assign(4, 1.0f);
    weights.k.segments.emplace_back(celeg::CpuInt8Matrix{});

    float query[] = {3.0f, 3.0f, 3.0f, 3.0f};
    float key[] = {2.0f, 2.0f, 2.0f, 2.0f};
    const std::array<int32_t, 3> position{0, 0, 0};
    celeg::apply_cpu_attention_qk(layout, weights, query, key, 0, position);

    for (float value : query) CELEG_TEST_CHECK(close(value, 3.0f));
    for (float value : key) CELEG_TEST_CHECK(close(value, 1.0f));
}

}

int main() {
    test_current_value_orthogonalization();
    test_adjacent_pair_rope();
    test_bidirectional_pattern_reads_future_keys();
    test_prefix_lm_pattern_boundaries();
    test_query_key_norm_uses_same_query_scale();
    test_query_only_norm_with_key_projection();
    test_key_only_norm_without_query_norm();
    return 0;
}
