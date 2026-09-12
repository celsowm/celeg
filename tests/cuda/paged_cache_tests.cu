#include "paged_cache_tests.hpp"

#include "utils.cuh"
#include "../support/assertions.hpp"
#include "../support/cuda_kernel_assertions.cuh"
#include "backend/cuda/paged_kv.hpp"
#include "celeg/model/resolved.hpp"
#include "celeg/model/runtime_types.hpp"

#include <iostream>
#include <vector>

namespace celeg::cuda_test {
namespace {

/// Minimal two-shape fixture backing the clone/prefix-clone checks.
struct TestShape {
    celeg::RuntimeTopology topology;
    celeg::CompiledModelProgram program;
};

std::vector<TestShape> registered_model_shapes() {
    std::vector<TestShape> shapes;
    for (int model = 0; model < 2; ++model) {
        TestShape result;
        auto& shape = result.topology;
        const int query_heads = model == 0 ? 16 : 32;
        const std::vector<bool> attention_layers = model == 0
            ? std::vector<bool>{
                false, false, true, false, true, false, true,
                false, true, false, true, false, true, false}
            : std::vector<bool>{
                false, false, true, false, false, true, false, false,
                true, false, true, false, true, false, true, false};
        celeg::ModelGraph graph;
        graph.hidden = model == 0 ? 1024 : 2048;
        graph.layers.resize(attention_layers.size());
        for (size_t index = 0; index < attention_layers.size(); ++index) {
            auto& layer = graph.layers[index];
            if (attention_layers[index]) {
                celeg::AttentionSpec attention;
                attention.query_heads = query_heads;
                attention.key_value_heads = 8;
                attention.head_dim = 64;
                attention.pattern = celeg::FullCausalPattern{};
                attention.position = celeg::RopePositionSpec{1.0e6, 1.0, {}};
                layer.mixer = attention;
            } else {
                layer.mixer = celeg::ShortConvolutionSpec{3, graph.hidden};
            }
            layer.feed_forward = celeg::DenseFeedForwardSpec{
                model == 0 ? 2560 : 12288, celeg::ActivationKind::SwiGLU};
        }
        shape = celeg::compose_runtime_topology(std::move(shape.dims), graph);
        result.program.hidden = graph.hidden;
        result.program.layers.resize(graph.layers.size());
        for (size_t index = 0; index < graph.layers.size(); ++index) {
            auto& compiled = result.program.layers[index];
            compiled.feed_forward = celeg::CompiledDenseFeedForwardProgram{
                model == 0 ? 2560 : 12288, celeg::ActivationKind::SwiGLU};
            if (const auto* attention = std::get_if<celeg::AttentionSpec>(
                    &graph.layers[index].mixer)) {
                celeg::CompiledOrdinaryKvStateLayout state_layout;
                state_layout.key_width = attention->key_value_width();
                state_layout.value_width = attention->key_value_width();
                state_layout.storage =
                    std::get<celeg::OrdinaryKvStateSpec>(attention->state).storage;
                compiled.mixer = celeg::CompiledAttentionProgram{
                    *attention, state_layout};
            } else {
                compiled.mixer = std::get<celeg::ShortConvolutionSpec>(
                    graph.layers[index].mixer);
            }
        }
        shapes.push_back(std::move(result));
    }
    if (shapes.empty()) {
        std::cerr << "registered_model_shapes: no model shapes registered\n";
        std::abort();
    }
    return shapes;
}

}

void run_paged_cache_tests(celeg::CudaStream& stream) {
(void)stream;
for (const TestShape& shape : registered_model_shapes()) {
    celeg::PhysicalPagedKvCache cache(3, 1, 4, celeg::KvCacheMode::Bf16,
                                      shape.topology.exec, shape.program);
    auto source = cache.allocate_tokens(1);
    CELEG_TEST_CHECK(source && source->size() == 1);
    const uint32_t source_page = source->front();
    const size_t page_elements = cache.page_vector_elements();
    std::vector<__nv_bfloat16> contents(page_elements, to_bf16(0.0f));
    contents[0] = to_bf16(3.5f);
    CELEG_CUDA(cudaMemcpy(cache.key_bf16() +
                        static_cast<size_t>(source_page) * page_elements,
                        contents.data(), contents.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
    auto cloned = cache.clone_page(source_page);
    CELEG_TEST_CHECK(cloned && *cloned != source_page);
    CELEG_TEST_CHECK(cache.ref_count(source_page) == 1);
    CELEG_TEST_CHECK(cache.ref_count(*cloned) == 1);
    __nv_bfloat16 copied{};
    CELEG_CUDA(cudaMemcpy(&copied, cache.key_bf16() +
                        static_cast<size_t>(*cloned) * page_elements,
                        sizeof(copied), cudaMemcpyDeviceToHost));
    expect_near(to_float(copied), 3.5f, 0.01f);
    cache.release(*source);
    cache.release(std::vector<uint32_t>{*cloned});
    CELEG_TEST_CHECK(cache.free_pages() == cache.total_pages());
}

for (const TestShape& shape : registered_model_shapes()) {
    constexpr int page_tokens = 4;
    celeg::PhysicalPagedKvCache cache(3, page_tokens, 8,
                                    celeg::KvCacheMode::Bf16,
                                    shape.topology.exec, shape.program);
    auto source = cache.allocate_tokens(page_tokens);
    CELEG_TEST_CHECK(source && source->size() == 1);
    const uint32_t source_page = source->front();
    const size_t page_elements = cache.page_vector_elements();
    std::vector<__nv_bfloat16> contents(page_elements, to_bf16(9.0f));
    CELEG_CUDA(cudaMemcpy(cache.key_bf16() +
                        static_cast<size_t>(source_page) * page_elements,
                        contents.data(), contents.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
    auto cloned = cache.clone_page_prefix(source_page, 1);
    CELEG_TEST_CHECK(cloned);
    std::vector<__nv_bfloat16> copied(page_elements);
    CELEG_CUDA(cudaMemcpy(copied.data(), cache.key_bf16() +
                        static_cast<size_t>(*cloned) * page_elements,
                        copied.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
    for (int layer = 0; layer < cache.attention_layers(); ++layer) {
        const size_t layer_base = cache.layer_vector_offset(layer);
        expect_near(to_float(copied[layer_base]), 9.0f, 0.01f);
    }
    cache.release(*source);
    cache.release(std::vector<uint32_t>{*cloned});
}

for (const TestShape& shape : registered_model_shapes()) {
    celeg::PhysicalPagedKvCache cache(3, 1, 4, celeg::KvCacheMode::Int8,
                                      shape.topology.exec, shape.program);
    auto source = cache.allocate_tokens(1);
    CELEG_TEST_CHECK(source && source->size() == 1);
    const uint32_t source_page = source->front();
    const size_t page_elements = cache.page_vector_elements();
    const size_t scale_elements = cache.page_scale_elements();
    const int8_t quantized = -37;
    const float scale = 0.03125f;
    CELEG_CUDA(cudaMemcpy(cache.key_int8() +
                        static_cast<size_t>(source_page) * page_elements,
                        &quantized, sizeof(quantized), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(cache.key_scales() +
                        static_cast<size_t>(source_page) * scale_elements,
                        &scale, sizeof(scale), cudaMemcpyHostToDevice));
    auto cloned = cache.clone_page(source_page);
    CELEG_TEST_CHECK(cloned && *cloned != source_page);
    int8_t copied_quantized = 0;
    float copied_scale = 0.0f;
    CELEG_CUDA(cudaMemcpy(&copied_quantized, cache.key_int8() +
                        static_cast<size_t>(*cloned) * page_elements,
                        sizeof(copied_quantized), cudaMemcpyDeviceToHost));
    CELEG_CUDA(cudaMemcpy(&copied_scale, cache.key_scales() +
                        static_cast<size_t>(*cloned) * scale_elements,
                        sizeof(copied_scale), cudaMemcpyDeviceToHost));
    CELEG_TEST_CHECK(copied_quantized == quantized);
    expect_near(copied_scale, scale, 1e-7f);
    cache.release(*source);
    cache.release(std::vector<uint32_t>{*cloned});
}
}

}
