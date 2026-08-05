#include "celeg/backend/cpu/compiler.hpp"
#include "celeg/backend/cuda/compiler.hpp"
#include "support/assertions.hpp"

#include <iostream>
#include <stdexcept>

int main() {
    celeg::ResolvedModel model;
    model.provenance.identity = "compiler-fixture";
    model.capabilities.supports_cpu = true;
    model.capabilities.supports_cuda = true;
    celeg::LayerSpec attention_layer;
    attention_layer.mixer = celeg::AttentionSpec{};
    attention_layer.feed_forward = celeg::DenseFeedForwardSpec{};
    model.graph.layers.push_back(attention_layer);
    celeg::LayerSpec convolution_layer;
    convolution_layer.mixer = celeg::ShortConvolutionSpec{};
    convolution_layer.feed_forward = celeg::MixtureOfExpertsSpec{
        16, 4, 2, true, true, 1.5f};
    model.graph.layers.push_back(convolution_layer);
    model.topology.hidden = 8;
    model.weight_plan.requests.push_back({celeg::TensorRole::AttentionInputNorm, 0, -1, {}});
    model.weight_plan.requests.push_back({celeg::TensorRole::AttentionOutput, 0, -1, {}});
    model.weight_plan.requests.push_back({celeg::TensorRole::ShortConvInput, 1, -1, {}});

    const auto cpu = celeg::CpuModelCompiler{}.compile(model);
    const auto cuda = celeg::CudaModelCompiler{}.compile(model);
    CELEG_TEST_CHECK(cpu.layers.size() == 2);
    CELEG_TEST_CHECK(cpu.layers[0].mixer == celeg::CompiledMixer::Attention);
    CELEG_TEST_CHECK(cpu.layers[1].mixer == celeg::CompiledMixer::ShortConvolution);
    CELEG_TEST_CHECK(cpu.layers[1].feed_forward == celeg::CompiledFeedForward::MixtureOfExperts);
    CELEG_TEST_CHECK(cpu.layers[1].moe.has_value());
    CELEG_TEST_CHECK(cpu.layers[1].moe->router.expert_count == 4);
    CELEG_TEST_CHECK(cpu.layers[1].moe->router.normalization ==
                     celeg::MoeNormalizationKind::SumSelected);
    CELEG_TEST_CHECK(cpu.layers[0].weight_request_indices.size() == 2);
    CELEG_TEST_CHECK(cuda.layers.size() == cpu.layers.size());

    celeg::MoeLayerProgram grouped;
    grouped.router = {celeg::MoeRouterScoreKind::SoftmaxLogits,
                      celeg::MoeSelectionKind::GroupedTopK,
                      celeg::MoeNormalizationKind::None, 8, 2, 2, 4, false, 1.0f};
    grouped.routed.mlp = {celeg::MoeActivation::SwiGLU, 8, 16};
    grouped.routed.payload.regions = {{celeg::TensorRole::MoeExpertGate, 0, 32,
                                       celeg::MoePayloadDType::BF16},
                                      {celeg::TensorRole::MoeExpertUp, 32, 32,
                                       celeg::MoePayloadDType::BF16}};
    grouped.routed.payload.total_bytes = 64;
    grouped.residency.expert_count = 8;
    grouped.validate();
    const std::string grouped_fingerprint = grouped.fingerprint();
    grouped.router.experts_per_token = 3;
    CELEG_TEST_CHECK(grouped.fingerprint() != grouped_fingerprint);
    bool overlap_rejected = false;
    grouped.router.experts_per_token = 2;
    grouped.routed.payload.regions[1].offset = 16;
    try { grouped.validate(); }
    catch (const std::invalid_argument&) { overlap_rejected = true; }
    CELEG_TEST_CHECK(overlap_rejected);

    celeg::ResolvedModel unsupported = model;
    auto& unsupported_moe = std::get<celeg::MixtureOfExpertsSpec>(
        unsupported.graph.layers[1].feed_forward);
    unsupported_moe.routing_group_count = 2;
    unsupported_moe.routing_experts_per_group = 2;
    bool backend_rejected = false;
    try { (void)celeg::CpuModelCompiler{}.compile(unsupported); }
    catch (const std::invalid_argument&) { backend_rejected = true; }
    CELEG_TEST_CHECK(backend_rejected);

    celeg::CompiledModelProgram fused = cpu;
    fused.layers[1].moe->routed.payload.layout = celeg::MoePayloadLayout::Fused;
    bool fused_rejected = false;
    try {
        celeg::validate_moe_backend_capabilities(fused, "cpu", {});
    } catch (const std::invalid_argument&) {
        fused_rejected = true;
    }
    CELEG_TEST_CHECK(fused_rejected);

    model.capabilities.supports_cpu = false;
    bool rejected = false;
    try { (void)celeg::CpuModelCompiler{}.compile(model); }
    catch (const std::invalid_argument&) { rejected = true; }
    CELEG_TEST_CHECK(rejected);
    std::cout << "model_compiler_test: ok\n";
    return 0;
}
