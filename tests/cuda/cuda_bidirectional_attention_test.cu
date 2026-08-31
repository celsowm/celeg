#include "backend/cuda/compiler.hpp"
#include "kernels/kernels.cuh"
#include "support/assertions.hpp"
#include "utils.cuh"

#include "celeg/model/position.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <variant>
#include <vector>

namespace {

celeg::ResolvedModel bidirectional_fixture() {
    celeg::ResolvedModel model;
    model.provenance.identity = "cuda-bidirectional-fixture";
    model.graph.hidden = 2;

    celeg::LayerSpec layer;
    layer.mixer_norm.before = celeg::NormSpec{
        1.0e-5f, celeg::NormWeightKind::Scale};
    layer.feed_forward_norm.before = celeg::NormSpec{
        1.0e-5f, celeg::NormWeightKind::Scale};

    celeg::AttentionSpec attention;
    attention.query_heads = 1;
    attention.key_value_heads = 1;
    attention.head_dim = 2;
    attention.pattern = celeg::BidirectionalPattern{};
    attention.position = celeg::NoPositionEncodingSpec{};
    layer.mixer = attention;
    layer.feed_forward = celeg::DenseFeedForwardSpec{
        4, celeg::ActivationKind::SwiGLU};
    model.graph.layers.push_back(layer);
    return model;
}

void check_compiler_contract() {
    const celeg::ResolvedModel model = bidirectional_fixture();
    const celeg::CompiledModelProgram program =
        celeg::CudaModelCompiler{}.compile(model);
    CELEG_TEST_CHECK(program.layers.size() == 1);
    const auto* compiled = std::get_if<celeg::CompiledAttentionProgram>(
        &program.layers[0].mixer);
    CELEG_TEST_CHECK(compiled != nullptr);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::BidirectionalPattern>(
        compiled->semantics.pattern));

    celeg::ResolvedModel prefix = model;
    std::get<celeg::AttentionSpec>(prefix.graph.layers[0].mixer).pattern =
        celeg::PrefixLmPattern{1};
    bool prefix_rejected = false;
    try {
        (void)celeg::CudaModelCompiler{}.compile(prefix);
    } catch (const std::invalid_argument&) {
        prefix_rejected = true;
    }
    CELEG_TEST_CHECK(prefix_rejected);

    celeg::ResolvedModel biased = model;
    std::get<celeg::AttentionSpec>(biased.graph.layers[0].mixer).bias =
        celeg::AlibiBiasSpec{{0.5f}};
    bool biased_rejected = false;
    try {
        (void)celeg::CudaModelCompiler{}.compile(biased);
    } catch (const std::invalid_argument&) {
        biased_rejected = true;
    }
    CELEG_TEST_CHECK(biased_rejected);
}

void check_bidirectional_prefill(celeg::CudaStream& stream) {
    const std::vector<__nv_bfloat16> query = {
        celeg::cuda_test::to_bf16(1.0f), celeg::cuda_test::to_bf16(0.0f),
        celeg::cuda_test::to_bf16(1.0f), celeg::cuda_test::to_bf16(0.0f)};
    const std::vector<__nv_bfloat16> keys = {
        celeg::cuda_test::to_bf16(1.0f), celeg::cuda_test::to_bf16(0.0f),
        celeg::cuda_test::to_bf16(0.0f), celeg::cuda_test::to_bf16(1.0f)};
    const std::vector<__nv_bfloat16> values = {
        celeg::cuda_test::to_bf16(2.0f), celeg::cuda_test::to_bf16(4.0f),
        celeg::cuda_test::to_bf16(6.0f), celeg::cuda_test::to_bf16(8.0f)};

    celeg::DeviceBuffer<__nv_bfloat16> dquery(query.size());
    celeg::DeviceBuffer<__nv_bfloat16> dkeys(keys.size());
    celeg::DeviceBuffer<__nv_bfloat16> dvalues(values.size());
    celeg::DeviceBuffer<__nv_bfloat16> causal_out(query.size());
    celeg::DeviceBuffer<__nv_bfloat16> bidirectional_out(query.size());
    CELEG_CUDA(cudaMemcpy(dquery.data(), query.data(), dquery.bytes(),
                          cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dkeys.data(), keys.data(), dkeys.bytes(),
                          cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dvalues.data(), values.data(), dvalues.bytes(),
                          cudaMemcpyHostToDevice));

    const celeg::GqaGeometry geometry{
        .q_heads = 1, .kv_heads = 1, .head_dim = 2};
    celeg::launch_gqa_prefill_strict({
        .query = dquery.data(),
        .kv = {.keys = dkeys.data(), .values = dvalues.data()},
        .out = causal_out.data(),
        .geometry = geometry,
        .extent = {.rows = 2},
        .stream = stream.get()});
    celeg::launch_gqa_prefill_strict({
        .query = dquery.data(),
        .kv = {.keys = dkeys.data(), .values = dvalues.data()},
        .out = bidirectional_out.data(),
        .geometry = geometry,
        .extent = {.rows = 2, .seq_len = 2},
        .stream = stream.get()});

    std::array<__nv_bfloat16, 4> causal{};
    std::array<__nv_bfloat16, 4> bidirectional{};
    CELEG_CUDA(cudaMemcpyAsync(causal.data(), causal_out.data(), causal_out.bytes(),
                               cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(bidirectional.data(), bidirectional_out.data(),
                               bidirectional_out.bytes(), cudaMemcpyDeviceToHost,
                               stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));

    CELEG_TEST_CHECK(std::abs(celeg::cuda_test::to_float(causal[0]) - 2.0f) < 0.02f);
    CELEG_TEST_CHECK(std::abs(celeg::cuda_test::to_float(causal[1]) - 4.0f) < 0.02f);

    const float first_weight =
        std::exp(1.0f / std::sqrt(2.0f)) /
        (std::exp(1.0f / std::sqrt(2.0f)) + 1.0f);
    const float expected0 = first_weight * 2.0f + (1.0f - first_weight) * 6.0f;
    const float expected1 = first_weight * 4.0f + (1.0f - first_weight) * 8.0f;
    for (int row = 0; row < 2; ++row) {
        CELEG_TEST_CHECK(std::abs(
            celeg::cuda_test::to_float(bidirectional[row * 2]) - expected0) < 0.03f);
        CELEG_TEST_CHECK(std::abs(
            celeg::cuda_test::to_float(bidirectional[row * 2 + 1]) - expected1) < 0.03f);
    }
    CELEG_TEST_CHECK(std::abs(
        celeg::cuda_test::to_float(bidirectional[0]) -
        celeg::cuda_test::to_float(causal[0])) > 1.0f);
}

}

int main() {
    check_compiler_contract();
    celeg::CudaStream stream;
    check_bidirectional_prefill(stream);
    return 0;
}
