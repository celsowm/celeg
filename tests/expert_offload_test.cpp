#include "lfm/runtime/moe/offload.hpp"

#include <cassert>
#include <iostream>

namespace {

// Reference LFM2.5-8B-A1B MoE topology (config.json):
//   hidden 2048, moe_intermediate 1792, 22 MoE layers, 32 experts,
//   4 experts/token, 6 attention layers, 8 KV heads, head_dim 64.
lfm::ModelShape make_8b_a1b_shape() {
    lfm::ModelShape shape;
    shape.hidden = 2048;
    shape.num_hidden_layers = 24;
    shape.num_dense_layers = 2;
    shape.num_attention_heads = 32;
    shape.num_key_value_heads = 8;
    shape.head_dim = 64;
    shape.architecture = lfm::ArchitectureKind::MoeLfm2;
    shape.moe_intermediate = 1792;
    shape.num_experts = 32;
    shape.experts_per_token = 4;
    // Derived field used by the KV planner; the 8B-A1B model has 6 attention
    // layers.
    shape.attention_layer_count = 6;
    return shape;
}

void test_byte_helpers() {
    const lfm::ModelShape shape = make_8b_a1b_shape();
    // 21 MiB per expert per the proposal.
    const std::size_t per_expert = lfm::bytes_per_expert_bf16(shape);
    assert(per_expert == 3ull * 1792ull * 2048ull * 2ull);
    assert(per_expert == 21ull * 1024ull * 1024ull);

    assert(lfm::moe_layer_count(shape) == 22);

    // 12 KiB/token: 6 layers * 2 * 8 KV heads * 64 * 2 bytes.
    const std::size_t kv_per_token = lfm::kv_cache_bytes(shape, 1);
    assert(kv_per_token == 12ull * 1024ull);
    assert(lfm::kv_cache_bytes(shape, 16384) == 12ull * 1024ull * 16384ull);
}

void test_disabled_plan() {
    lfm::ExpertOffloadPlanInputs in;
    in.shape = make_8b_a1b_shape();
    in.options.mode = lfm::ExpertOffloadMode::None;
    const lfm::ExpertOffloadPlan plan = lfm::plan_expert_offload(in);
    assert(!plan.enabled);
    assert(plan.experts_per_layer == in.shape.num_experts);
    assert(plan.host_experts_per_layer == 0);
}

void test_auto_plan_rtx3060() {
    lfm::ExpertOffloadPlanInputs in;
    in.shape = make_8b_a1b_shape();
    in.options.mode = lfm::ExpertOffloadMode::Auto;
    in.options.gpu_memory_reserve_bytes = 768ull * 1024 * 1024;
    in.gpu_free_bytes = static_cast<std::size_t>(10.42 * 1024 * 1024 * 1024);
    in.non_expert_weight_bytes =
        static_cast<std::size_t>(1.33 * 1024 * 1024 * 1024);
    in.workspace_bytes = static_cast<std::size_t>(0.5 * 1024 * 1024 * 1024);
    in.context_tokens = 16384;

    const lfm::ExpertOffloadPlan plan = lfm::plan_expert_offload(in);
    assert(plan.enabled);
    // With ~10.42 GiB free and this topology the planner should land in the
    // 14-16 experts/layer band from the proposal.
    assert(plan.experts_per_layer >= 14 && plan.experts_per_layer <= 16);
    assert(plan.host_experts_per_layer ==
           in.shape.num_experts - plan.experts_per_layer);
    assert(plan.gpu_expert_cache_bytes ==
           static_cast<std::size_t>(plan.experts_per_layer) * 22ull *
               plan.bytes_per_expert);
}

void test_explicit_per_layer() {
    lfm::ExpertOffloadPlanInputs in;
    in.shape = make_8b_a1b_shape();
    in.options.mode = lfm::ExpertOffloadMode::Host;
    in.options.experts_per_layer = 14;
    const lfm::ExpertOffloadPlan plan = lfm::plan_expert_offload(in);
    assert(plan.enabled);
    assert(plan.experts_per_layer == 14);
    assert(plan.host_experts_per_layer == 18);
    // 14 * 22 * 21 MiB = 6.32 GiB (per the proposal table).
    assert(plan.gpu_expert_cache_bytes ==
           14ull * 22ull * 21ull * 1024ull * 1024ull);
}

void test_infeasible_throws() {
    lfm::ExpertOffloadPlanInputs in;
    in.shape = make_8b_a1b_shape();
    in.options.mode = lfm::ExpertOffloadMode::Auto;
    in.gpu_free_bytes = static_cast<std::size_t>(1.5 * 1024 * 1024 * 1024);
    in.non_expert_weight_bytes =
        static_cast<std::size_t>(1.33 * 1024 * 1024 * 1024);
    bool threw = false;
    try {
        (void)lfm::plan_expert_offload(in);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
}

void test_report_nonempty() {
    lfm::ExpertOffloadPlanInputs in;
    in.shape = make_8b_a1b_shape();
    in.options.mode = lfm::ExpertOffloadMode::Host;
    in.options.experts_per_layer = 14;
    const lfm::ExpertOffloadPlan plan = lfm::plan_expert_offload(in);
    const std::string report = plan.report();
    assert(report.find("MoE offload plan:") != std::string::npos);
    assert(report.find("experts per layer:") != std::string::npos);
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
