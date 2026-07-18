#include "lfm/execution_plan.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

int main() {
    using namespace lfm;

    ModelOptions bf16;
    auto plan = ExecutionPlan::compile(bf16, 4096);
    assert(plan.linear_kernel() == LinearKernelKind::Bf16Cublas);
    assert(!plan.segmented_attention(4096));

    ModelOptions int4;
    int4.weight_mode = WeightMode::Int4;
    auto int4_plan = ExecutionPlan::compile(int4, 4096);
    assert(int4_plan.linear_kernel() == LinearKernelKind::W4A16);

    ModelOptions automatic;
    automatic.fast_attention = true;
    automatic.attention_mode = AttentionMode::Auto;
    automatic.attention_auto_threshold = 1024;
    auto auto_plan = ExecutionPlan::compile(automatic, 4096);
    assert(!auto_plan.segmented_attention(1023));
    assert(auto_plan.segmented_attention(1024));

    bool rejected = false;
    try {
        ModelOptions invalid;
        invalid.attention_mode = AttentionMode::Segmented;
        (void)ExecutionPlan::compile(invalid, 4096);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);

    std::cout << "execution_plan_test: ok\n";
    return 0;
}
