#include "celeg/backend/cuda/moe/offload.hpp"
#include "support/assertions.hpp"
#include <algorithm>
#include <iostream>
#include <utility>

namespace {

// Reference LFM2.5-8B-A1B MoE topology (config.json):
//   hidden 2048, moe_intermediate 1792, 22 MoE layers, 32 experts,
//   4 experts/token, 6 attention layers, 8 KV heads, head_dim 64.
celeg::CompiledModelProgram make_8b_a1b_program() {
    celeg::CompiledModelProgram program;
    program.hidden = 2048;
    program.layers.resize(24);
    for (int index = 0; index < 24; ++index) {
        auto& layer = program.layers[static_cast<size_t>(index)];
        layer.mixer = index < 6 ? celeg::CompiledMixer::Attention
                                : celeg::CompiledMixer::ShortConvolution;
        layer.feed_forward = index < 2 ? celeg::CompiledFeedForward::Dense
                                       : celeg::CompiledFeedForward::MixtureOfExperts;
        if (index < 6) {
            celeg::AttentionSpec attention;
            attention.query_heads = 32;
            attention.key_value_heads = 8;
            attention.head_dim = 64;
            attention.pattern = celeg::FullCausalPattern{};
            attention.position = celeg::RopePositionSpec{1.0e6, 1.0, {}};
            layer.attention = attention;
        }
        layer.feed_forward_intermediate = 1792;
        if (index >= 2) {
            celeg::MoeLayerProgram moe;
            moe.router.expert_count = 32;
            moe.router.experts_per_token = 4;
            moe.routed.mlp.hidden_size = 2048;
            moe.routed.mlp.intermediate_size = 1792;
            layer.moe = std::move(moe);
        }
    }
    return program;
}

void test_byte_helpers() {
    const celeg::CompiledModelProgram program = make_8b_a1b_program();
    // 21 MiB per expert per the proposal.
    const std::size_t per_expert = celeg::bytes_per_expert_bf16(program);
    CELEG_TEST_CHECK(per_expert == 3ull * 1792ull * 2048ull * 2ull);
    CELEG_TEST_CHECK(per_expert == 21ull * 1024ull * 1024ull);

    CELEG_TEST_CHECK(celeg::moe_layer_count(program) == 22);

    // 12 KiB/token: 6 layers * 2 * 8 KV heads * 64 * 2 bytes.
    const std::size_t kv_per_token = celeg::kv_cache_bytes(program, 1);
    CELEG_TEST_CHECK(kv_per_token == 12ull * 1024ull);
    CELEG_TEST_CHECK(celeg::kv_cache_bytes(program, 16384) == 12ull * 1024ull * 16384ull);
}

void test_disabled_plan() {
    celeg::ExpertOffloadPlanInputs in(make_8b_a1b_program());
    in.options.mode = celeg::ExpertOffloadMode::None;
    const celeg::ExpertOffloadPlan plan = celeg::plan_expert_offload(in);
    CELEG_TEST_CHECK(!plan.enabled);
    CELEG_TEST_CHECK(plan.experts_per_layer == 32);
    CELEG_TEST_CHECK(plan.host_experts_per_layer == 0);
}

void test_auto_plan_rtx3060() {
    celeg::ExpertOffloadPlanInputs in(make_8b_a1b_program());
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
           32 - plan.experts_per_layer);
    CELEG_TEST_CHECK(plan.gpu_expert_cache_bytes ==
           static_cast<std::size_t>(plan.experts_per_layer) * 22ull *
               plan.bytes_per_expert);
}

void test_explicit_per_layer() {
    celeg::ExpertOffloadPlanInputs in(make_8b_a1b_program());
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
    celeg::ExpertOffloadPlanInputs in(make_8b_a1b_program());
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
    celeg::ExpertOffloadPlanInputs in(make_8b_a1b_program());
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
