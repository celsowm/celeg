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

    // Queries inside the prefix can attend anywhere inside the prefix, including
    // future prefix tokens, but cannot see tokens after the prefix.
    CELEG_TEST_CHECK(pattern.allows(0, 0));
    CELEG_TEST_CHECK(pattern.allows(0, 3));
    CELEG_TEST_CHECK(!pattern.allows(0, 4));
    CELEG_TEST_CHECK(pattern.allows(2, 3));
    CELEG_TEST_CHECK(!pattern.allows(2, 5));

    // The first query after the prefix transitions to ordinary causal semantics.
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

    // A prefix covering the complete sequence has no out-of-prefix future region.
    celeg::CpuAttentionPattern full_prefix;
    full_prefix.storage = celeg::PrefixLmPattern{8};
    CELEG_TEST_CHECK(!full_prefix.may_read_future(0, 8));
    CELEG_TEST_CHECK(full_prefix.allows(0, 7));
}

// The attention kernels fold 1/sqrt(head_dim) into the scores themselves, so
// apply_cpu_attention_qk must premultiply the query by query_scale/(1/sqrt(head_dim))
// on every path. The query-key-norm path used to apply query_scale directly,
// leaving every score a further 1/sqrt(head_dim) too small -- 8x for head_dim 64.
// Softmax over a single position normalises that away, so it stayed invisible
// until a sequence had two or more tokens, and only mattered enough to corrupt
// output once one position carried a much larger value vector than the others.
void test_query_key_norm_uses_same_query_scale() {
    constexpr int kHeadDim = 64;
    constexpr int kHeads = 2;
    constexpr int kWidth = kHeads * kHeadDim;

    // A real rotary spec, evaluated at position 0 where RoPE is the identity,
    // so the rotation drops out and only the query scale is under test.
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
    // Unit norm weights over a query whose per-head RMS is exactly 1 make the
    // RMS normalisation an identity, so the only thing separating the two
    // layouts is the query scale.
    weights.q_norm.assign(kHeadDim, 1.0f);

    std::vector<float> plain_query(kWidth);
    for (int i = 0; i < kWidth; ++i) plain_query[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    std::vector<float> normed_query = plain_query;

    const std::array<int32_t, 3> position{0, 0, 0};
    celeg::apply_cpu_attention_qk(plain, weights, plain_query.data(), nullptr, 0, position);
    celeg::apply_cpu_attention_qk(normed, weights, normed_query.data(), nullptr, 0, position);

    // query_scale equals the kernel's own 1/sqrt(64) = 0.125 here, so the
    // correct premultiplier is exactly 1 and the query is unchanged.
    for (int i = 0; i < kWidth; ++i) {
        CELEG_TEST_CHECK(close(plain_query[i], (i % 2 == 0) ? 1.0f : -1.0f));
        CELEG_TEST_CHECK(close(normed_query[i], plain_query[i]));
    }
}

}

int main() {
    test_current_value_orthogonalization();
    test_adjacent_pair_rope();
    test_bidirectional_pattern_reads_future_keys();
    test_prefix_lm_pattern_boundaries();
    test_query_key_norm_uses_same_query_scale();
    return 0;
}
