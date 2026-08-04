#include "celeg/backend/cpu/compiler.hpp"
#include "celeg/backend/cuda/compiler.hpp"
#include "support/assertions.hpp"

#include <iostream>
#include <stdexcept>

int main() {
    celeg::ResolvedModel model;
    model.identity = "compiler-fixture";
    model.capabilities.supports_cpu = true;
    model.capabilities.supports_cuda = true;
    celeg::LayerSpec attention_layer;
    attention_layer.mixer = celeg::AttentionSpec{};
    attention_layer.feed_forward = celeg::DenseFeedForwardSpec{};
    model.graph.layers.push_back(attention_layer);
    celeg::LayerSpec convolution_layer;
    convolution_layer.mixer = celeg::ShortConvolutionSpec{};
    convolution_layer.feed_forward = celeg::MixtureOfExpertsSpec{};
    model.graph.layers.push_back(convolution_layer);
    model.weight_plan.requests.push_back({celeg::TensorRole::AttentionInputNorm, 0, -1, {}});
    model.weight_plan.requests.push_back({celeg::TensorRole::AttentionOutput, 0, -1, {}});
    model.weight_plan.requests.push_back({celeg::TensorRole::ShortConvInput, 1, -1, {}});

    const auto cpu = celeg::CpuModelCompiler{}.compile(model);
    const auto cuda = celeg::CudaModelCompiler{}.compile(model);
    CELEG_TEST_CHECK(cpu.layers.size() == 2);
    CELEG_TEST_CHECK(cpu.layers[0].mixer == celeg::CompiledMixer::Attention);
    CELEG_TEST_CHECK(cpu.layers[1].mixer == celeg::CompiledMixer::ShortConvolution);
    CELEG_TEST_CHECK(cpu.layers[1].feed_forward == celeg::CompiledFeedForward::MixtureOfExperts);
    CELEG_TEST_CHECK(cpu.layers[0].weight_request_indices.size() == 2);
    CELEG_TEST_CHECK(cuda.layers.size() == cpu.layers.size());

    model.capabilities.supports_cpu = false;
    bool rejected = false;
    try { (void)celeg::CpuModelCompiler{}.compile(model); }
    catch (const std::invalid_argument&) { rejected = true; }
    CELEG_TEST_CHECK(rejected);
    std::cout << "model_compiler_test: ok\n";
    return 0;
}
