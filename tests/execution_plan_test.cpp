#include "celeg/model/execution/plan.hpp"
#include "support/assertions.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    using celeg::AttentionMode;
    using celeg::ExecutionPlan;
    using celeg::LinearKernelKind;
    using celeg::ModelOptions;
    using celeg::WeightMode;

    ModelOptions bf16;
    auto plan = ExecutionPlan::compile(bf16, 4096);
    CELEG_TEST_CHECK(plan.linear_kernel() == LinearKernelKind::Bf16Cublas);
    CELEG_TEST_CHECK(!plan.segmented_attention(4096));

    ModelOptions int4;
    int4.weight_mode = WeightMode::Int4;
    auto int4_plan = ExecutionPlan::compile(int4, 4096);
    CELEG_TEST_CHECK(int4_plan.linear_kernel() == LinearKernelKind::W4A16);

    ModelOptions automatic;
    automatic.fast_attention = true;
    automatic.attention_mode = AttentionMode::Auto;
    automatic.attention_auto_threshold = 1024;
    auto auto_plan = ExecutionPlan::compile(automatic, 4096);
    CELEG_TEST_CHECK(!auto_plan.segmented_attention(1023));
    CELEG_TEST_CHECK(auto_plan.segmented_attention(1024));

    bool rejected = false;
    try {
        ModelOptions invalid;
        invalid.attention_mode = AttentionMode::Segmented;
        (void)ExecutionPlan::compile(invalid, 4096);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CELEG_TEST_CHECK(rejected);

    // Phase 1.4: --weight-mode native must not report plain BF16 cuBLASLt. The
    // actual execution mixes BF16 (norms / conv) and GGUF MMQ (linear blocks);
    // the plan label must be honest about that, or the diagnostics contradict
    // the real storage / kernels. See docs/ARCHITECTURE_RULES.md
    // section 1.4.
    ModelOptions native;
    native.weight_mode = WeightMode::NativeGguf;
    auto native_plan = ExecutionPlan::compile(native, 4096);
    CELEG_TEST_CHECK(native_plan.linear_kernel() ==
                   LinearKernelKind::MixedBf16AndGgufMmq);
    {
        const std::string desc = native_plan.description();
        CELEG_TEST_CHECK(desc.find("linear=mixed-bf16-and-gguf-mmq") !=
                       std::string::npos);
        // The diagnostic must not falsely contain the BF16 cuBLASLt label.
        CELEG_TEST_CHECK(desc.find("bf16-cublaslt") == std::string::npos);
    }

    std::cout << "execution_plan_test: ok\n";
    return 0;
}
