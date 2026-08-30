#include "celeg/backend/cuda/execution_plan.hpp"
#include "support/assertions.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

int main() try {
    using celeg::AttentionMode;
    using celeg::CudaExecutionPlan;
    using celeg::LinearKernelKind;
    using celeg::CudaModelOptions;
    using celeg::WeightMode;

    const uint64_t before_compile = CudaExecutionPlan::compile_count();
    CudaModelOptions bf16;
    auto plan = CudaExecutionPlan::compile(bf16, 4096);
    CELEG_TEST_CHECK(CudaExecutionPlan::compile_count() == before_compile + 1);
    CELEG_TEST_CHECK(plan.linear_kernel() == LinearKernelKind::Bf16Cublas);
    CELEG_TEST_CHECK(!plan.segmented_attention(4096));
    CELEG_TEST_CHECK(plan.fingerprint() != 0);
    celeg::CudaDeviceCapabilities device;
    device.device_ordinal = 2;
    device.compute_major = 8;
    device.compute_minor = 6;
    device.mmq_tensor_core_supported = true;
    device.mmq_tensor_core_enabled = true;
    auto device_plan = CudaExecutionPlan::compile(bf16, 4096, device);
    CELEG_TEST_CHECK(device_plan.device().device_ordinal == 2);
    CELEG_TEST_CHECK(device_plan.mmq_tensor_cores_enabled());
    CELEG_TEST_CHECK(device_plan.fingerprint() != plan.fingerprint());
    device.mmq_tensor_core_supported = false;
    device.mmq_tensor_core_enabled = true;
    const auto rejected_mmq_plan = CudaExecutionPlan::compile(bf16, 4096, device);
    CELEG_TEST_CHECK(!rejected_mmq_plan.mmq_tensor_cores_enabled());

    CudaModelOptions int4;
    int4.weight_mode = WeightMode::Int4;
    auto int4_plan = CudaExecutionPlan::compile(int4, 4096);
    CELEG_TEST_CHECK(int4_plan.linear_kernel() == LinearKernelKind::W4A16);

    CudaModelOptions automatic;
    automatic.fast_attention = true;
    automatic.attention_mode = AttentionMode::Auto;
    automatic.attention_auto_threshold = 1024;
    auto auto_plan = CudaExecutionPlan::compile(automatic, 4096);
    CELEG_TEST_CHECK(!auto_plan.segmented_attention(1023));
    CELEG_TEST_CHECK(auto_plan.segmented_attention(1024));

    bool rejected = false;
    try {
        CudaModelOptions invalid;
        invalid.attention_mode = AttentionMode::Segmented;
        (void)CudaExecutionPlan::compile(invalid, 4096);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CELEG_TEST_CHECK(rejected);

    {
        CudaModelOptions mtp;
        mtp.enable_mtp = true;
        bool rejected_graph = false;
        try {
            (void)CudaExecutionPlan::compile(mtp, 4096);
        } catch (const std::invalid_argument&) {
            rejected_graph = true;
        }
        CELEG_TEST_CHECK(rejected_graph);
        mtp.cuda_graph = false;
        mtp.mtp_speculative_tokens = 1;
        auto mtp_plan = CudaExecutionPlan::compile(mtp, 4096);
        CELEG_TEST_CHECK(mtp_plan.options().enable_mtp);
        mtp.mtp_speculative_tokens = 2;
        bool rejected_depth = false;
        try {
            (void)CudaExecutionPlan::compile(mtp, 4096);
        } catch (const std::invalid_argument&) {
            rejected_depth = true;
        }
        CELEG_TEST_CHECK(rejected_depth);
    }

    CudaModelOptions native;
    native.weight_mode = WeightMode::NativeGguf;
    auto native_plan = CudaExecutionPlan::compile(native, 4096);
    CELEG_TEST_CHECK(native_plan.linear_kernel() ==
                   LinearKernelKind::MixedBf16AndGgufMmq);
    {
        const std::string desc = native_plan.description();
        CELEG_TEST_CHECK(desc.find("linear=mixed-bf16-and-gguf-mmq") !=
                       std::string::npos);
        CELEG_TEST_CHECK(desc.find("bf16-cublaslt") == std::string::npos);
    }

    std::cout << "execution_plan_test: ok\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "execution_plan_test: " << error.what() << '\n';
    return 1;
}
