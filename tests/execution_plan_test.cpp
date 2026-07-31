#include "lfm/model/execution/plan.hpp"
#include "support/assertions.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

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

    // Phase 1.4: --weight-mode native must not report plain BF16 cuBLASLt. The
    // actual execution mixes BF16 (norms / conv) and GGUF MMQ (linear blocks);
    // the plan label must be honest about that, or the diagnostics contradict
    // the real storage / kernels. See lfm25_multi_stage_refactoring_plan.md
    // section 1.4.
    ModelOptions native;
    native.weight_mode = WeightMode::NativeGguf;
    auto native_plan = ExecutionPlan::compile(native, 4096);
    LFM_TEST_CHECK(native_plan.linear_kernel() ==
                   LinearKernelKind::MixedBf16AndGgufMmq);
    {
        const std::string desc = native_plan.description();
        LFM_TEST_CHECK(desc.find("linear=mixed-bf16-and-gguf-mmq") !=
                       std::string::npos);
        // The diagnostic must not falsely contain the BF16 cuBLASLt label.
        LFM_TEST_CHECK(desc.find("bf16-cublaslt") == std::string::npos);
    }

    std::cout << "execution_plan_test: ok\n";
    return 0;
}
