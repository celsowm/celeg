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

celeg::ResolvedModel dynamic_sparse_fixture() {
    celeg::ResolvedModel model;
    model.provenance.identity = "cuda-dynamic-sparse-fixture";
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
    attention.pattern = celeg::DynamicSparsePattern{2, 1};
    attention.position = celeg::NoPositionEncodingSpec{};
    layer.mixer = attention;
    layer.feed_forward = celeg::DenseFeedForwardSpec{
        4, celeg::ActivationKind::SwiGLU};
    model.graph.layers.push_back(layer);
    return model;
}

void check_compiler_contract() {
    const celeg::ResolvedModel model = dynamic_sparse_fixture();
    const celeg::CompiledModelProgram program =
        celeg::CudaModelCompiler{}.compile(model);
    const auto* compiled = std::get_if<celeg::CompiledAttentionProgram>(
        &program.layers[0].mixer);
    CELEG_TEST_CHECK(compiled != nullptr);
    CELEG_TEST_CHECK(std::holds_alternative<celeg::DynamicSparsePattern>(
        compiled->semantics.pattern));

    celeg::ResolvedModel invalid = model;
    std::get<celeg::AttentionSpec>(invalid.graph.layers[0].mixer).pattern =
        celeg::DynamicSparsePattern{2, 33};
    bool rejected = false;
    try {
        (void)celeg::CudaModelCompiler{}.compile(invalid);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CELEG_TEST_CHECK(rejected);
}

void check_content_selected_top_block(celeg::CudaStream& stream) {
    constexpr int rows = 4;
    std::vector<__nv_bfloat16> query(rows * 2, celeg::cuda_test::to_bf16(0.0f));
    std::vector<__nv_bfloat16> keys(rows * 2, celeg::cuda_test::to_bf16(0.0f));
    std::vector<__nv_bfloat16> values(rows * 2, celeg::cuda_test::to_bf16(0.0f));

    query[3 * 2] = celeg::cuda_test::to_bf16(1.0f);
    keys[0 * 2] = celeg::cuda_test::to_bf16(10.0f);
    keys[1 * 2] = celeg::cuda_test::to_bf16(10.0f);
    keys[2 * 2] = celeg::cuda_test::to_bf16(0.0f);
    keys[3 * 2] = celeg::cuda_test::to_bf16(0.0f);
    values[0 * 2] = celeg::cuda_test::to_bf16(2.0f);
    values[1 * 2] = celeg::cuda_test::to_bf16(4.0f);
    values[2 * 2] = celeg::cuda_test::to_bf16(100.0f);
    values[3 * 2] = celeg::cuda_test::to_bf16(200.0f);

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

    celeg::launch_gqa_prefill_dynamic_sparse({
        .query = dquery.data(),
        .kv = {.keys = dkeys.data(), .values = dvalues.data()},
        .out = output.data(),
        .geometry = {.q_heads = 1, .kv_heads = 1, .head_dim = 2},
        .extent = {.rows = rows},
        .stream = stream.get()},
        {.block_size = 2, .max_selected_blocks = 1});

    std::array<__nv_bfloat16, rows * 2> host{};
    CELEG_CUDA(cudaMemcpyAsync(host.data(), output.data(), output.bytes(),
                               cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));

    CELEG_TEST_CHECK(std::abs(celeg::cuda_test::to_float(host[3 * 2]) - 3.0f) < 0.05f);
    CELEG_TEST_CHECK(std::abs(celeg::cuda_test::to_float(host[3 * 2]) - 150.0f) > 100.0f);
}

}

int main() {
    check_compiler_contract();
    celeg::CudaStream stream;
    check_content_selected_top_block(stream);
    return 0;
}
