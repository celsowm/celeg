#include "backend/cuda/compiler.hpp"
#include "kernels/kernels.cuh"
#include "support/assertions.hpp"
#include "utils.cuh"

#include <array>
#include <cmath>
#include <stdexcept>
#include <variant>
#include <vector>

namespace {

celeg::ResolvedModel block_sparse_fixture() {
    celeg::ResolvedModel model;
    model.provenance.identity = "cuda-block-sparse-fixture";
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
    attention.pattern = celeg::BlockSparsePattern{2, 1, 1};
    attention.position = celeg::NoPositionEncodingSpec{};
    layer.mixer = attention;
    layer.feed_forward = celeg::DenseFeedForwardSpec{
        4, celeg::ActivationKind::SwiGLU};
    model.graph.layers.push_back(layer);
    return model;
}

void check_compiler_contract() {
    const celeg::ResolvedModel model = block_sparse_fixture();
    const celeg::CompiledModelProgram program =
        celeg::CudaModelCompiler{}.compile(model);
    const auto* compiled = std::get_if<celeg::CompiledAttentionProgram>(
        &program.layers[0].mixer);
    CELEG_TEST_CHECK(compiled != nullptr);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::BlockSparsePattern>(
        compiled->semantics.pattern));

    celeg::ResolvedModel invalid = model;
    std::get<celeg::AttentionSpec>(invalid.graph.layers[0].mixer).pattern =
        celeg::BlockSparsePattern{0, 1, 0};
    bool invalid_rejected = false;
    try {
        (void)celeg::CudaModelCompiler{}.compile(invalid);
    } catch (const std::invalid_argument&) {
        invalid_rejected = true;
    }
    CELEG_TEST_CHECK(invalid_rejected);

    celeg::ResolvedModel dynamic = model;
    std::get<celeg::AttentionSpec>(dynamic.graph.layers[0].mixer).pattern =
        celeg::DynamicSparsePattern{2, 1};
    bool dynamic_rejected = false;
    try {
        (void)celeg::CudaModelCompiler{}.compile(dynamic);
    } catch (const std::invalid_argument&) {
        dynamic_rejected = true;
    }
    CELEG_TEST_CHECK(dynamic_rejected);
}

void check_block_sparse_prefill(celeg::CudaStream& stream) {
    constexpr int rows = 6;
    std::vector<__nv_bfloat16> query(rows * 2, celeg::cuda_test::to_bf16(0.0f));
    std::vector<__nv_bfloat16> keys(rows * 2, celeg::cuda_test::to_bf16(0.0f));
    const std::array<float, rows> scalar_values{1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f};
    std::vector<__nv_bfloat16> values(rows * 2, celeg::cuda_test::to_bf16(0.0f));
    for (int row = 0; row < rows; ++row) {
        values[row * 2] = celeg::cuda_test::to_bf16(scalar_values[row]);
    }

    celeg::DeviceBuffer<__nv_bfloat16> dquery(query.size());
    celeg::DeviceBuffer<__nv_bfloat16> dkeys(keys.size());
    celeg::DeviceBuffer<__nv_bfloat16> dvalues(values.size());
    celeg::DeviceBuffer<__nv_bfloat16> output(values.size());
    CELEG_CUDA(cudaMemcpy(dquery.data(), query.data(), dquery.bytes(),
                          cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dkeys.data(), keys.data(), dkeys.bytes(),
                          cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dvalues.data(), values.data(), dvalues.bytes(),
                          cudaMemcpyHostToDevice));

    celeg::launch_gqa_prefill_block_sparse({
        .query = dquery.data(),
        .kv = {.keys = dkeys.data(), .values = dvalues.data()},
        .out = output.data(),
        .geometry = {.q_heads = 1, .kv_heads = 1, .head_dim = 2},
        .extent = {.rows = rows},
        .stream = stream.get()},
        {.block_size = 2, .local_blocks = 1, .global_blocks = 1});

    std::array<__nv_bfloat16, rows * 2> host{};
    CELEG_CUDA(cudaMemcpyAsync(host.data(), output.data(), output.bytes(),
                               cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));

    CELEG_TEST_CHECK(std::abs(celeg::cuda_test::to_float(host[4 * 2]) -
                              (1.0f + 2.0f + 16.0f) / 3.0f) < 0.05f);
    CELEG_TEST_CHECK(std::abs(celeg::cuda_test::to_float(host[5 * 2]) -
                              (1.0f + 2.0f + 16.0f + 32.0f) / 4.0f) < 0.05f);

    const float dense_row5 = (1.0f + 2.0f + 4.0f + 8.0f + 16.0f + 32.0f) / 6.0f;
    CELEG_TEST_CHECK(std::abs(celeg::cuda_test::to_float(host[5 * 2]) -
                              dense_row5) > 1.0f);
}

}

int main() {
    check_compiler_contract();
    celeg::CudaStream stream;
    check_block_sparse_prefill(stream);
    return 0;
}
