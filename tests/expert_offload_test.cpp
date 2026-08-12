#include "celeg/backend/cuda/moe/offload.hpp"
#include "support/assertions.hpp"
#include <algorithm>
#include <iostream>

namespace {

// Reference LFM2.5-8B-A1B MoE topology (config.json):
//   hidden 2048, moe_intermediate 1792, 22 MoE layers, 32 experts,
//   4 experts/token, 6 attention layers, 8 KV heads, head_dim 64.
celeg::RuntimeTopology make_8b_a1b_shape() {
    celeg::RuntimeTopology shape;
    shape.exec.hidden = 2048;
    shape.exec.num_hidden_layers = 24;
    shape.exec.num_dense_layers = 2;
    shape.exec.feed_forward_kinds.assign(static_cast<size_t>(shape.exec.num_hidden_layers),
                                    celeg::FeedForwardKind::MixtureOfExperts);
    shape.exec.feed_forward_kinds[0] = celeg::FeedForwardKind::Dense;
    shape.exec.feed_forward_kinds[1] = celeg::FeedForwardKind::Dense;
    shape.exec.moe_intermediate = 1792;
    shape.exec.num_experts = 32;
    shape.exec.experts_per_token = 4;
    shape.exec.mixer_kinds.assign(static_cast<size_t>(shape.exec.num_hidden_layers),
                             celeg::MixerKind::ShortConvolution);
    std::fill_n(shape.exec.mixer_kinds.begin(), 6, celeg::MixerKind::Attention);
    shape.exec.attention_layouts.resize(static_cast<size_t>(shape.exec.num_hidden_layers));
    for (auto& attention : shape.exec.attention_layouts) {
        attention.query_heads = 32;
        attention.key_value_heads = 8;
        attention.head_dim = 64;
        attention.pattern = celeg::FullCausalPattern{};
    }
    for (auto& attention : shape.exec.attention_layouts) {
        attention.position = celeg::RopePositionSpec{1.0e6, 1.0, {}};
    }
    // Derived field used by the KV planner; the 8B-A1B model has 6 attention
    // layers.
    shape.exec.attention_layer_count = 6;
    return shape;
}

void test_byte_helpers() {
    const celeg::RuntimeTopology shape = make_8b_a1b_shape();
    // 21 MiB per expert per the proposal.
    const std::size_t per_expert = celeg::bytes_per_expert_bf16(shape.exec);
    CELEG_TEST_CHECK(per_expert == 3ull * 1792ull * 2048ull * 2ull);
    CELEG_TEST_CHECK(per_expert == 21ull * 1024ull * 1024ull);

    CELEG_TEST_CHECK(celeg::moe_layer_count(shape.exec) == 22);

    // 12 KiB/token: 6 layers * 2 * 8 KV heads * 64 * 2 bytes.
    const std::size_t kv_per_token = celeg::kv_cache_bytes(shape.exec, 1);
    CELEG_TEST_CHECK(kv_per_token == 12ull * 1024ull);
    CELEG_TEST_CHECK(celeg::kv_cache_bytes(shape.exec, 16384) == 12ull * 1024ull * 16384ull);
}

void test_disabled_plan() {
    celeg::ExpertOffloadPlanInputs in(make_8b_a1b_shape().exec);
    in.options.mode = celeg::ExpertOffloadMode::None;
    const celeg::ExpertOffloadPlan plan = celeg::plan_expert_offload(in);
    CELEG_TEST_CHECK(!plan.enabled);
    CELEG_TEST_CHECK(plan.experts_per_layer == in.shape.num_experts);
    CELEG_TEST_CHECK(plan.host_experts_per_layer == 0);
}

void test_auto_plan_rtx3060() {
    celeg::ExpertOffloadPlanInputs in(make_8b_a1b_shape().exec);
    in.options.mode = celeg::ExpertOffloadMode::Auto;
    in.options.gpu_memory_reserve_bytes = 768ull * 1024 * 1024;
    in.gpu_free_bytes = static_cast<std::size_t>(10.42 * 1024 * 1024 * 1024);
    in.non_expert_weight_bytes =
        static_cast<std::size_t>(1.33 * 1024 * 1024 * 1024);
    in.workspace_bytes = static_cast<std::size_t>(0.5 * 1024 * 1024 * 1024);
    in.context_tokens = 16384;

    const celeg::ExpertOffloadPlan plan = celeg::plan_expert_offload(in);
    CELEG_TEST_CHECK(plan.enabled);
    // With ~10.42 GiB free and this topology the planner should land in the
    // 14-16 experts/layer band from the proposal.
    CELEG_TEST_CHECK(plan.experts_per_layer >= 14 && plan.experts_per_layer <= 16);
    CELEG_TEST_CHECK(plan.host_experts_per_layer ==
           in.shape.num_experts - plan.experts_per_layer);
    CELEG_TEST_CHECK(plan.gpu_expert_cache_bytes ==
           static_cast<std::size_t>(plan.experts_per_layer) * 22ull *
               plan.bytes_per_expert);
}

void test_explicit_per_layer() {
    celeg::ExpertOffloadPlanInputs in(make_8b_a1b_shape().exec);
    in.options.mode = celeg::ExpertOffloadMode::Host;
    in.options.experts_per_layer = 14;
    const celeg::ExpertOffloadPlan plan = celeg::plan_expert_offload(in);
    CELEG_TEST_CHECK(plan.enabled);
    CELEG_TEST_CHECK(plan.experts_per_layer == 14);
    CELEG_TEST_CHECK(plan.host_experts_per_layer == 18);
    // 14 * 22 * 21 MiB = 6.32 GiB (per the proposal table).
    CELEG_TEST_CHECK(plan.gpu_expert_cache_bytes ==
           14ull * 22ull * 21ull * 1024ull * 1024ull);
}

void test_infeasible_throws() {
    celeg::ExpertOffloadPlanInputs in(make_8b_a1b_shape().exec);
    in.options.mode = celeg::ExpertOffloadMode::Auto;
    in.gpu_free_bytes = static_cast<std::size_t>(1.5 * 1024 * 1024 * 1024);
    in.non_expert_weight_bytes =
        static_cast<std::size_t>(1.33 * 1024 * 1024 * 1024);
    bool threw = false;
    try {
        (void)celeg::plan_expert_offload(in);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CELEG_TEST_CHECK(threw);
}

void test_report_nonempty() {
    celeg::ExpertOffloadPlanInputs in(make_8b_a1b_shape().exec);
    in.options.mode = celeg::ExpertOffloadMode::Host;
    in.options.experts_per_layer = 14;
    const celeg::ExpertOffloadPlan plan = celeg::plan_expert_offload(in);
    const std::string report = plan.report();
    CELEG_TEST_CHECK(report.find("MoE offload plan:") != std::string::npos);
    CELEG_TEST_CHECK(report.find("experts per layer:") != std::string::npos);
}

} // namespace

int main() {
    test_byte_helpers();
    test_disabled_plan();
    test_auto_plan_rtx3060();
    test_explicit_per_layer();
    test_infeasible_throws();
    test_report_nonempty();
    std::cout << "expert_offload_test: ok\n";
    return 0;
}
