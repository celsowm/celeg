#include "backend/cuda/compiler.hpp"
#include "celeg/attention/dynamic_sparse_semantics.hpp"
#include "kernels/kernels.cuh"
#include "support/assertions.hpp"
#include "support/cuda_kernel_assertions.cuh"
#include "utils.cuh"

#include <algorithm>
#include <array>
#include <cfloat>
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

    /// `build_model_program` requires every layer to own at least one weight
    /// request; the fixture has no checkpoint behind it, so declare the roles
    /// the layer's specs imply directly.
    for (celeg::TensorRole role : {
             celeg::TensorRole::AttentionInputNorm,
             celeg::TensorRole::AttentionQuery,
             celeg::TensorRole::AttentionKey,
             celeg::TensorRole::AttentionValue,
             celeg::TensorRole::AttentionOutput,
             celeg::TensorRole::FfnInputNorm,
             celeg::TensorRole::FfnGate,
             celeg::TensorRole::FfnUp,
             celeg::TensorRole::FfnDown}) {
        model.weight_plan.requests.push_back({role, 0, -1, {}});
    }
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

float bf16_round(float value) {
    return celeg::cuda_test::to_float(celeg::cuda_test::to_bf16(value));
}

float host_dot(const std::vector<__nv_bfloat16>& query,
               const std::vector<__nv_bfloat16>& keys,
               int query_row, int token, int head_dim) {
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        dot += celeg::cuda_test::to_float(query[query_row * head_dim + d]) *
               celeg::cuda_test::to_float(keys[token * head_dim + d]);
    }
    return dot;
}

std::vector<float> dynamic_sparse_host_oracle(
    const std::vector<__nv_bfloat16>& query,
    const std::vector<__nv_bfloat16>& keys,
    const std::vector<__nv_bfloat16>& values,
    int rows, int head_dim, int block_size, int max_selected_blocks) {
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> output(static_cast<size_t>(rows) * head_dim, 0.0f);

    for (int query_row = 0; query_row < rows; ++query_row) {
        const int candidate_count =
            celeg::attention_semantics::dynamic_sparse_candidate_count(
                query_row, block_size);
        std::vector<float> candidate_scores(candidate_count, -FLT_MAX);
        for (int block = 0; block < candidate_count; ++block) {
            const int begin = celeg::attention_semantics::dynamic_sparse_block_begin(
                block, block_size);
            const int end =
                celeg::attention_semantics::dynamic_sparse_block_end_exclusive(
                    query_row, block, block_size);
            for (int token = begin; token < end; ++token) {
                candidate_scores[block] = std::max(
                    candidate_scores[block],
                    host_dot(query, keys, query_row, token, head_dim) * scale);
            }
        }

        std::vector<int> selected(max_selected_blocks, -1);
        std::vector<float> selected_scores(
            max_selected_blocks,
            celeg::attention_semantics::dynamic_sparse_lowest_score());
        celeg::attention_semantics::dynamic_sparse_select_top_k(
            candidate_scores.data(), candidate_count, max_selected_blocks,
            selected.data(), selected_scores.data());

        float maximum = -FLT_MAX;
        for (int token = 0; token <= query_row; ++token) {
            if (!celeg::attention_semantics::dynamic_sparse_selected_block(
                    token / block_size, selected.data(), max_selected_blocks)) {
                continue;
            }
            const float dot = host_dot(query, keys, query_row, token, head_dim);
            const float score = bf16_round(bf16_round(dot) * scale);
            maximum = std::max(maximum, score);
        }

        float denominator = 0.0f;
        for (int token = 0; token <= query_row; ++token) {
            if (!celeg::attention_semantics::dynamic_sparse_selected_block(
                    token / block_size, selected.data(), max_selected_blocks)) {
                continue;
            }
            const float dot = host_dot(query, keys, query_row, token, head_dim);
            const float score = bf16_round(bf16_round(dot) * scale);
            denominator += std::exp(score - maximum);
        }

        for (int token = 0; token <= query_row; ++token) {
            if (!celeg::attention_semantics::dynamic_sparse_selected_block(
                    token / block_size, selected.data(), max_selected_blocks)) {
                continue;
            }
            const float dot = host_dot(query, keys, query_row, token, head_dim);
            const float score = bf16_round(bf16_round(dot) * scale);
            const float probability = bf16_round(
                std::exp(score - maximum) / denominator);
            for (int d = 0; d < head_dim; ++d) {
                output[query_row * head_dim + d] +=
                    probability * celeg::cuda_test::to_float(
                        values[token * head_dim + d]);
            }
        }
        for (int d = 0; d < head_dim; ++d) {
            output[query_row * head_dim + d] = bf16_round(
                output[query_row * head_dim + d]);
        }
    }
    return output;
}

void check_output_matches_host_oracle(celeg::CudaStream& stream,
                                      int max_selected_blocks) {
    constexpr int rows = 6;
    constexpr int head_dim = 2;
    constexpr int block_size = 2;

    const float query_values[rows][head_dim] = {
        {1.0f, 0.5f}, {0.5f, 1.0f}, {1.0f, -0.25f},
        {-0.5f, 1.5f}, {0.75f, 0.25f}, {1.25f, -0.5f}};
    const float key_values[rows][head_dim] = {
        {2.0f, 0.0f}, {1.0f, 1.0f}, {-1.0f, 3.0f},
        {0.5f, 2.0f}, {4.0f, -1.0f}, {1.0f, -2.0f}};
    const float value_values[rows][head_dim] = {
        {2.0f, -1.0f}, {4.0f, 3.0f}, {8.0f, 1.0f},
        {16.0f, -2.0f}, {32.0f, 4.0f}, {64.0f, -8.0f}};

    std::vector<__nv_bfloat16> query(rows * head_dim);
    std::vector<__nv_bfloat16> keys(rows * head_dim);
    std::vector<__nv_bfloat16> values(rows * head_dim);
    for (int row = 0; row < rows; ++row) {
        for (int d = 0; d < head_dim; ++d) {
            query[row * head_dim + d] =
                celeg::cuda_test::to_bf16(query_values[row][d]);
            keys[row * head_dim + d] =
                celeg::cuda_test::to_bf16(key_values[row][d]);
            values[row * head_dim + d] =
                celeg::cuda_test::to_bf16(value_values[row][d]);
        }
    }

    const std::vector<float> expected = dynamic_sparse_host_oracle(
        query, keys, values, rows, head_dim, block_size, max_selected_blocks);

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
        .geometry = {.q_heads = 1, .kv_heads = 1, .head_dim = head_dim},
        .extent = {.rows = rows},
        .stream = stream.get()},
        {.block_size = block_size,
         .max_selected_blocks = max_selected_blocks});

    std::array<__nv_bfloat16, rows * head_dim> actual{};
    CELEG_CUDA(cudaMemcpyAsync(actual.data(), output.data(), output.bytes(),
                               cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));

    for (int index = 0; index < rows * head_dim; ++index) {
        CELEG_TEST_CHECK(std::abs(
            celeg::cuda_test::to_float(actual[index]) - expected[index]) < 0.08f);
    }
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
    check_output_matches_host_oracle(stream, 1);
    check_output_matches_host_oracle(stream, 2);
    return 0;
}
