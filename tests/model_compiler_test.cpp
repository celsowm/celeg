#include "celeg/backend/cpu/compiler.hpp"
#include "celeg/backend/cuda/compiler.hpp"
#include "celeg/backend/moe_capabilities.hpp"
#include "celeg/model/position.hpp"
#include "support/assertions.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

const celeg::AttentionSpec& compiled_attention(
    const celeg::CompiledLayerProgram& layer) {
    return std::get<celeg::CompiledAttentionProgram>(layer.mixer).semantics;
}

const celeg::CompiledAttentionStateLayout& compiled_attention_state(
    const celeg::CompiledLayerProgram& layer) {
    return std::get<celeg::CompiledAttentionProgram>(layer.mixer).state_layout;
}

} // namespace

int main() {
    celeg::ResolvedModel model;
    model.provenance.identity = "compiler-fixture";
    model.capabilities.supports_cpu = true;
    model.capabilities.supports_cuda = true;
    celeg::LayerSpec attention_layer;
    celeg::AttentionSpec attention;
    attention.query_heads = 1;
    attention.key_value_heads = 1;
    attention.head_dim = 8;
    attention.position = celeg::RopePositionSpec{10000.0, 1.0, {}};
    attention_layer.mixer = attention;
    attention_layer.feed_forward = celeg::DenseFeedForwardSpec{
        16, celeg::ActivationKind::SwiGLU};
    model.graph.layers.push_back(attention_layer);

    celeg::LayerSpec convolution_layer;
    convolution_layer.mixer = celeg::ShortConvolutionSpec{3, 8, false};
    convolution_layer.feed_forward = celeg::MixtureOfExpertsSpec{
        16, 4, 2, true, true, 1.5f};
    model.graph.layers.push_back(convolution_layer);
    model.graph.hidden = 8;
    model.weight_plan.requests.push_back(
        {celeg::TensorRole::AttentionInputNorm, 0, -1, {}});
    model.weight_plan.requests.push_back(
        {celeg::TensorRole::AttentionOutput, 0, -1, {}});
    model.weight_plan.requests.push_back(
        {celeg::TensorRole::ShortConvInput, 1, -1, {}});

    const auto cpu = celeg::CpuModelCompiler{}.compile(model);
    const auto cuda = celeg::CudaModelCompiler{}.compile(model);
    CELEG_TEST_CHECK(cpu.layers.size() == 2);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::CompiledAttentionProgram>(
        cpu.layers[0].mixer));
    CELEG_TEST_CHECK(std::holds_alternative<celeg::ShortConvolutionSpec>(
        cpu.layers[1].mixer));
    CELEG_TEST_CHECK(std::holds_alternative<celeg::MoeLayerProgram>(
        cpu.layers[1].feed_forward));
    const auto& compiled_moe = std::get<celeg::MoeLayerProgram>(
        cpu.layers[1].feed_forward);
    CELEG_TEST_CHECK(compiled_moe.router.expert_count == 4);
    CELEG_TEST_CHECK(compiled_moe.router.normalization ==
                     celeg::MoeNormalizationKind::SumSelected);
    CELEG_TEST_CHECK(cpu.layers[0].weight_request_indices.size() == 2);
    const auto& ordinary_state = std::get<celeg::CompiledOrdinaryKvStateLayout>(
        compiled_attention_state(cpu.layers[0]));
    CELEG_TEST_CHECK(ordinary_state.key_width == 8 &&
                     ordinary_state.value_width == 8 &&
                     ordinary_state.persistent_elements() == 16);
    CELEG_TEST_CHECK(cuda.layers.size() == cpu.layers.size());

    celeg::ModelGraph mlp_only_graph;
    mlp_only_graph.hidden = 8;
    celeg::LayerSpec mlp_only_layer;
    mlp_only_layer.mixer = celeg::MlpBlockSpec{64, celeg::ActivationKind::Relu2};
    mlp_only_layer.feed_forward = std::monostate{};
    mlp_only_graph.layers.push_back(std::move(mlp_only_layer));
    const celeg::ExecutionTopology mlp_only_topology =
        celeg::ExecutionTopology::derive(mlp_only_graph);
    CELEG_TEST_CHECK(mlp_only_topology.mlp_only_layer_count == 1);
    CELEG_TEST_CHECK(mlp_only_topology.max_feed_forward_intermediate == 64);

    celeg::ResolvedModel unsupported_pattern = model;
    std::get<celeg::AttentionSpec>(unsupported_pattern.graph.layers[0].mixer).pattern =
        celeg::BidirectionalPattern{};
    const auto bidirectional_cpu = celeg::CpuModelCompiler{}.compile(unsupported_pattern);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::BidirectionalPattern>(
        compiled_attention(bidirectional_cpu.layers[0]).pattern));
    bool cuda_pattern_rejected = false;
    try { (void)celeg::CudaModelCompiler{}.compile(unsupported_pattern); }
    catch (const std::invalid_argument&) { cuda_pattern_rejected = true; }
    CELEG_TEST_CHECK(cuda_pattern_rejected);

    celeg::ResolvedModel sparse_pattern = model;
    std::get<celeg::AttentionSpec>(sparse_pattern.graph.layers[0].mixer).pattern =
        celeg::BlockSparsePattern{16, 2, 1};
    const auto sparse_cpu = celeg::CpuModelCompiler{}.compile(sparse_pattern);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::BlockSparsePattern>(
        compiled_attention(sparse_cpu.layers[0]).pattern));
    bool cuda_sparse_rejected = false;
    try { (void)celeg::CudaModelCompiler{}.compile(sparse_pattern); }
    catch (const std::invalid_argument&) { cuda_sparse_rejected = true; }
    CELEG_TEST_CHECK(cuda_sparse_rejected);

    celeg::ResolvedModel unsupported_position = model;
    auto& alibi_attention =
        std::get<celeg::AttentionSpec>(unsupported_position.graph.layers[0].mixer);
    alibi_attention.position = celeg::NoPositionEncodingSpec{};
    alibi_attention.bias = celeg::AlibiBiasSpec{{1.0f}};
    const auto alibi_cpu = celeg::CpuModelCompiler{}.compile(unsupported_position);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::AlibiBiasSpec>(
        compiled_attention(alibi_cpu.layers[0]).bias));
    const auto cuda_alibi = celeg::CudaModelCompiler{}.compile(unsupported_position);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::AlibiBiasSpec>(
        compiled_attention(cuda_alibi.layers[0]).bias));

    celeg::ResolvedModel unsupported_relative = model;
    auto& relative_attention =
        std::get<celeg::AttentionSpec>(unsupported_relative.graph.layers[0].mixer);
    relative_attention.position = celeg::NoPositionEncodingSpec{};
    relative_attention.bias = celeg::RelativePositionBiasSpec{32, 128, true};
    const auto cpu_relative = celeg::CpuModelCompiler{}.compile(unsupported_relative);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::RelativePositionBiasSpec>(
        compiled_attention(cpu_relative.layers[0]).bias));
    bool cuda_relative_rejected = false;
    try { (void)celeg::CudaModelCompiler{}.compile(unsupported_relative); }
    catch (const std::invalid_argument&) { cuda_relative_rejected = true; }
    CELEG_TEST_CHECK(cuda_relative_rejected);

    celeg::ResolvedModel int8_state = model;
    std::get<celeg::AttentionSpec>(int8_state.graph.layers[0].mixer)
        .state_storage.key = celeg::StateScalarType::INT8;
    const auto int8_program = celeg::CpuModelCompiler{}.compile(int8_state);
    CELEG_TEST_CHECK(int8_program.semantic_fingerprint != cpu.semantic_fingerprint);

    celeg::ResolvedModel latent_state = model;
    std::get<celeg::AttentionSpec>(latent_state.graph.layers[0].mixer).state =
        celeg::LatentAttentionStateSpec{32, 16, 16, true};
    const auto latent_program = celeg::CpuModelCompiler{}.compile(latent_state);
    const auto& latent_layout = std::get<celeg::CompiledLatentStateLayout>(
        compiled_attention_state(latent_program.layers[0]));
    CELEG_TEST_CHECK(latent_layout.latent_width == 64);
    const auto cuda_latent = celeg::CudaModelCompiler{}.compile(latent_state);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::CompiledLatentStateLayout>(
        compiled_attention_state(cuda_latent.layers[0])));

    celeg::ResolvedModel latent_gate = latent_state;
    std::get<celeg::AttentionSpec>(latent_gate.graph.layers[0].mixer).output_gate.kind =
        celeg::AttentionGateKind::Sigmoid;
    bool cuda_latent_gate_rejected = false;
    try { (void)celeg::CudaModelCompiler{}.compile(latent_gate); }
    catch (const std::invalid_argument&) { cuda_latent_gate_rejected = true; }
    CELEG_TEST_CHECK(cuda_latent_gate_rejected);

    celeg::ResolvedModel oversized_latent = latent_state;
    std::get<celeg::AttentionSpec>(oversized_latent.graph.layers[0].mixer).state =
        celeg::LatentAttentionStateSpec{513, 16, 16, true};
    bool cuda_oversized_latent_rejected = false;
    try { (void)celeg::CudaModelCompiler{}.compile(oversized_latent); }
    catch (const std::invalid_argument&) { cuda_oversized_latent_rejected = true; }
    CELEG_TEST_CHECK(cuda_oversized_latent_rejected);

    celeg::ResolvedModel unsupported_source = model;
    std::get<celeg::AttentionSpec>(unsupported_source.graph.layers[0].mixer).sources =
        celeg::AttentionSourceSpec{celeg::AttentionSourceKind::CurrentSequence,
                                   celeg::AttentionSourceKind::ExternalMemory, 0};
    const auto external_cpu = celeg::CpuModelCompiler{}.compile(unsupported_source);
    CELEG_TEST_CHECK(compiled_attention(external_cpu.layers[0]).uses_external_memory());

    celeg::RopePositionSpec linear{10000.0, 1.0,
        {celeg::RopeScalingKind::Linear, 2.0, 32}};
    const double base_frequency = celeg::rope_frequency(
        celeg::RopePositionSpec{10000.0, 1.0, {}}, 2, 8, 64);
    CELEG_TEST_CHECK(std::abs(celeg::rope_frequency(linear, 2, 8, 64) -
                             base_frequency / 2.0) < 1.0e-12);
    celeg::RopePositionSpec dynamic{10000.0, 1.0,
        {celeg::RopeScalingKind::DynamicNtk, 2.0, 32}};
    CELEG_TEST_CHECK(celeg::rope_frequency(dynamic, 1, 8, 16) !=
                     celeg::rope_frequency(dynamic, 1, 8, 64));
    celeg::RopePositionSpec long_rope{10000.0, 1.0,
        {celeg::RopeScalingKind::Long, 2.0, 32, 1.0, 0.0, 0.0, 1.0, 1.0,
         {1.0f, 2.0f, 3.0f, 4.0f}, {2.0f, 3.0f, 4.0f, 5.0f}}};
    CELEG_TEST_CHECK(celeg::rope_frequency(long_rope, 2, 8, 16) !=
                     celeg::rope_frequency(long_rope, 2, 8, 64));
    celeg::RopePositionSpec yarn_rope{10000.0, 1.0,
        {celeg::RopeScalingKind::Yarn, 2.0, 32, 1.25, 1.0, 4.0}};
    CELEG_TEST_CHECK(celeg::rope_frequency(yarn_rope, 0, 8, 64) !=
                     celeg::rope_frequency(yarn_rope, 3, 8, 64));
    celeg::RopePositionSpec llama3_rope{10000.0, 1.0,
        {celeg::RopeScalingKind::Llama3Frequency, 2.0, 32, 1.0, 1.0, 1.0, 8.0}};
    CELEG_TEST_CHECK(celeg::rope_frequency(llama3_rope, 0, 8, 64) !=
                     celeg::rope_frequency(llama3_rope, 3, 8, 64));
    celeg::ResolvedModel cuda_yarn = model;
    std::get<celeg::AttentionSpec>(cuda_yarn.graph.layers[0].mixer).position = yarn_rope;
    (void)celeg::CudaModelCompiler{}.compile(cuda_yarn);
    celeg::ResolvedModel cuda_llama3 = model;
    std::get<celeg::AttentionSpec>(cuda_llama3.graph.layers[0].mixer).position = llama3_rope;
    (void)celeg::CudaModelCompiler{}.compile(cuda_llama3);
    celeg::ResolvedModel cuda_long = model;
    std::get<celeg::AttentionSpec>(cuda_long.graph.layers[0].mixer).position = long_rope;
    (void)celeg::CudaModelCompiler{}.compile(cuda_long);

    celeg::MoeLayerProgram grouped;
    grouped.router.score = celeg::MoeRouterScoreKind::SoftmaxLogits;
    grouped.router.selection = celeg::MoeGroupedTopKSelectionSpec{2, 4, 1, 2};
    grouped.router.normalization = celeg::MoeNormalizationKind::None;
    grouped.router.expert_count = 8;
    grouped.router.experts_per_token = 2;
    grouped.router.routed_scaling = 1.0f;
    grouped.routed.mlp = {celeg::MoeActivation::SwiGLU, 8, 16};
    grouped.routed.payload.regions = {{celeg::TensorRole::MoeExpertGate, 32},
                                      {celeg::TensorRole::MoeExpertUp, 32}};
    grouped.validate();
    const std::string grouped_fingerprint = grouped.fingerprint();
    grouped.router.experts_per_token = 3;
    CELEG_TEST_CHECK(grouped.fingerprint() != grouped_fingerprint);
    bool empty_region_rejected = false;
    grouped.router.experts_per_token = 2;
    grouped.routed.payload.regions[1].elements = 0;
    try { grouped.validate(); }
    catch (const std::invalid_argument&) { empty_region_rejected = true; }
    CELEG_TEST_CHECK(empty_region_rejected);

    celeg::ResolvedModel unsupported = model;
    auto& unsupported_moe = std::get<celeg::MixtureOfExpertsSpec>(
        unsupported.graph.layers[1].feed_forward);
    unsupported_moe.selection = celeg::MoeGroupedTopKSelectionSpec{2, 2, 1, 1};
    bool backend_rejected = false;
    try { (void)celeg::CpuModelCompiler{}.compile(unsupported); }
    catch (const std::invalid_argument&) { backend_rejected = true; }
    CELEG_TEST_CHECK(backend_rejected);

    celeg::CompiledModelProgram fused = cpu;
    std::get<celeg::MoeLayerProgram>(fused.layers[1].feed_forward)
        .routed.payload.layout = celeg::MoePayloadLayout::Fused;
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
