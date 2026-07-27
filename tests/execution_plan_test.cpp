#include "lfm/model/execution/plan.hpp"
#include "support/assertions.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    using lfm::AttentionMode;
    using lfm::ExecutionPlan;
    using lfm::LinearKernelKind;
    using lfm::ModelOptions;
    using lfm::WeightMode;

    ModelOptions bf16;
    auto plan = ExecutionPlan::compile(bf16, 4096);
    LFM_TEST_CHECK(plan.linear_kernel() == LinearKernelKind::Bf16Cublas);
    LFM_TEST_CHECK(!plan.segmented_attention(4096));

    ModelOptions int4;
    int4.weight_mode = WeightMode::Int4;
    auto int4_plan = ExecutionPlan::compile(int4, 4096);
    LFM_TEST_CHECK(int4_plan.linear_kernel() == LinearKernelKind::W4A16);

    ModelOptions automatic;
    automatic.fast_attention = true;
    automatic.attention_mode = AttentionMode::Auto;
    automatic.attention_auto_threshold = 1024;
    auto auto_plan = ExecutionPlan::compile(automatic, 4096);
    LFM_TEST_CHECK(!auto_plan.segmented_attention(1023));
    LFM_TEST_CHECK(auto_plan.segmented_attention(1024));

    bool rejected = false;
    try {
        ModelOptions invalid;
        invalid.attention_mode = AttentionMode::Segmented;
        (void)ExecutionPlan::compile(invalid, 4096);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    LFM_TEST_CHECK(rejected);

    std::cout << "execution_plan_test: ok\n";
    return 0;
}
