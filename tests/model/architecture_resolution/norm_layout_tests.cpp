#include "norm_layout_tests.hpp"

#include "support.hpp"
#include "support/assertions.hpp"

namespace celeg::architecture_resolution_test {

void run_norm_layout_tests(const celeg::ArchitectureCatalog& catalog) {
    const auto pre_only = resolve_norm_layout(catalog, NormFixtureLayout::PreOnly);
    CELEG_TEST_CHECK(pre_only.graph.layers[0].mixer_norm.before.has_value());
    CELEG_TEST_CHECK(!pre_only.graph.layers[0].mixer_norm.after.has_value());
    CELEG_TEST_CHECK(pre_only.graph.layers[0].feed_forward_norm.before.has_value());
    CELEG_TEST_CHECK(!pre_only.graph.layers[0].feed_forward_norm.after.has_value());
    CELEG_TEST_CHECK(has_weight_role(pre_only.weight_plan,
                                     celeg::TensorRole::AttentionInputNorm, 0));
    CELEG_TEST_CHECK(!has_weight_role(pre_only.weight_plan,
                                      celeg::TensorRole::AttentionPostNorm, 0));

    const auto post_only = resolve_norm_layout(catalog, NormFixtureLayout::PostOnly);
    CELEG_TEST_CHECK(!post_only.graph.layers[0].mixer_norm.before.has_value());
    CELEG_TEST_CHECK(post_only.graph.layers[0].mixer_norm.after.has_value());
    CELEG_TEST_CHECK(!post_only.graph.layers[0].feed_forward_norm.before.has_value());
    CELEG_TEST_CHECK(post_only.graph.layers[0].feed_forward_norm.after.has_value());
    CELEG_TEST_CHECK(!has_weight_role(post_only.weight_plan,
                                      celeg::TensorRole::AttentionInputNorm, 0));
    CELEG_TEST_CHECK(has_weight_role(post_only.weight_plan,
                                     celeg::TensorRole::AttentionPostNorm, 0));
    CELEG_TEST_CHECK(!has_weight_role(post_only.weight_plan,
                                      celeg::TensorRole::FfnInputNorm, 0));
    CELEG_TEST_CHECK(has_weight_role(post_only.weight_plan,
                                     celeg::TensorRole::FfnOutputNorm, 0));

    const auto sandwich = resolve_norm_layout(catalog, NormFixtureLayout::Sandwich);
    CELEG_TEST_CHECK(sandwich.graph.layers[0].mixer_norm.before.has_value());
    CELEG_TEST_CHECK(sandwich.graph.layers[0].mixer_norm.after.has_value());
    CELEG_TEST_CHECK(sandwich.graph.layers[0].feed_forward_norm.before.has_value());
    CELEG_TEST_CHECK(sandwich.graph.layers[0].feed_forward_norm.after.has_value());

    const auto no_norm = resolve_norm_layout(catalog, NormFixtureLayout::None);
    CELEG_TEST_CHECK(!no_norm.graph.layers[0].mixer_norm.before.has_value());
    CELEG_TEST_CHECK(!no_norm.graph.layers[0].mixer_norm.after.has_value());
    CELEG_TEST_CHECK(!no_norm.graph.layers[0].feed_forward_norm.before.has_value());
    CELEG_TEST_CHECK(!no_norm.graph.layers[0].feed_forward_norm.after.has_value());
}

}
