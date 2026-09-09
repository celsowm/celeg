#include "celeg/attention/online_semantics.hpp"
#include "support/assertions.hpp"

#include <array>
#include <cmath>
#include <limits>

namespace {

void first_score_initializes_state() {
    const auto t = celeg::attention_semantics::online_transition(
        -std::numeric_limits<float>::infinity(), 0.0f, 2.0f);
    CELEG_TEST_CHECK(t.maximum == 2.0f);
    CELEG_TEST_CHECK(t.previous_scale == 0.0f);
    CELEG_TEST_CHECK(std::abs(t.current_scale - 1.0f) < 1.0e-6f);
    CELEG_TEST_CHECK(std::abs(t.denominator - 1.0f) < 1.0e-6f);
}

void increasing_score_rescales_previous_mass() {
    const auto t = celeg::attention_semantics::online_transition(
        1.0f, 2.0f, 3.0f);
    CELEG_TEST_CHECK(t.maximum == 3.0f);
    CELEG_TEST_CHECK(std::abs(t.previous_scale - std::exp(-2.0f)) < 1.0e-6f);
    CELEG_TEST_CHECK(std::abs(t.current_scale - 1.0f) < 1.0e-6f);
    CELEG_TEST_CHECK(std::abs(t.denominator -
        (2.0f * std::exp(-2.0f) + 1.0f)) < 1.0e-6f);
}

void lower_score_preserves_maximum() {
    const auto t = celeg::attention_semantics::online_transition(
        4.0f, 1.5f, 2.0f);
    CELEG_TEST_CHECK(t.maximum == 4.0f);
    CELEG_TEST_CHECK(std::abs(t.previous_scale - 1.0f) < 1.0e-6f);
    CELEG_TEST_CHECK(std::abs(t.current_scale - std::exp(-2.0f)) < 1.0e-6f);
}

void accumulation_uses_transition_scales() {
    const auto t = celeg::attention_semantics::online_transition(1.0f, 1.0f, 2.0f);
    const float actual = celeg::attention_semantics::online_accumulate(3.0f, 5.0f, t);
    const float expected = 3.0f * std::exp(-1.0f) + 5.0f;
    CELEG_TEST_CHECK(std::abs(actual - expected) < 1.0e-6f);
}

}

int main() {
    first_score_initializes_state();
    increasing_score_rescales_previous_mass();
    lower_score_preserves_maximum();
    accumulation_uses_transition_scales();
    return 0;
}
