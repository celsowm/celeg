#include "paged_decode_tests.hpp"

#include "utils.cuh"
#include "../support/assertions.hpp"
#include "../support/cuda_kernel_assertions.cuh"
#include "kernels/kernels.cuh"
#include "backend/cuda/paged_kv.hpp"
#include "celeg/model/reference.hpp"

#include <cstdint>
#include <vector>

namespace celeg::cuda_test {

void run_paged_decode_tests(celeg::CudaStream& stream) {
{
    constexpr int seq_len = 5;
    constexpr int q_heads = 2;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 2;
    std::vector<__nv_bfloat16> q = {
        to_bf16(1.0f), to_bf16(0.5f),
        to_bf16(-0.5f), to_bf16(1.0f)};
    std::vector<__nv_bfloat16> k(seq_len * kv_heads * head_dim);
    std::vector<__nv_bfloat16> v(seq_len * kv_heads * head_dim);
    for (int token_index = 0; token_index < seq_len; ++token_index) {
        k[token_index * 2] = to_bf16(0.25f * (token_index + 1));
        k[token_index * 2 + 1] = to_bf16(1.0f - 0.1f * token_index);
        v[token_index * 2] = to_bf16(static_cast<float>(token_index));
        v[token_index * 2 + 1] = to_bf16(static_cast<float>(token_index + 2));
    }
    int32_t position = seq_len - 1;
    celeg::DeviceBuffer<__nv_bfloat16> dq(q.size()), dk(k.size()), dv(v.size());
    celeg::DeviceBuffer<__nv_bfloat16> reference(q.size()), segmented(q.size());
    celeg::DeviceBuffer<int32_t> dposition(1);
    constexpr int max_segments = 8;
    celeg::DeviceBuffer<float> partial_max(q_heads * max_segments);
    celeg::DeviceBuffer<float> partial_denom(q_heads * max_segments);
    celeg::DeviceBuffer<float> partial_accum(q_heads * max_segments * head_dim);
    CELEG_CUDA(cudaMemcpy(dq.data(), q.data(), dq.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dk.data(), k.data(), dk.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dv.data(), v.data(), dv.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dposition.data(), &position, sizeof(position), cudaMemcpyHostToDevice));
    celeg::launch_gqa_decode_online_device({
        .query = dq.data(),
        .kv = {.keys = dk.data(), .values = dv.data()},
        .out = reference.data(),
        .geometry = {.q_heads = q_heads, .kv_heads = kv_heads, .head_dim = head_dim},
        .extent = {.position = dposition.data()},
        .stream = stream.get()});
    std::vector<__nv_bfloat16> a(q.size());
    CELEG_CUDA(cudaMemcpyAsync(a.data(), reference.data(), reference.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    /// The segment count is a device property, so the result has to be
    /// independent of it: fewer segments than tokens, exactly as many, and
    /// more (which leaves a tail of segments the partial kernel skips).
    /// min_segments == segments forces the widest split the allocation
    /// allows; min_segments == 1 lets kDecodeTokensPerSegment decide, which
    /// for this tiny sequence collapses to a single segment.
    for (int segments : {1, 2, 3, 5, max_segments})
    for (int min_segments : {1, segments}) {
        CELEG_CUDA(cudaMemsetAsync(partial_max.data(), 0, partial_max.bytes(),
                                   stream.get()));
        CELEG_CUDA(cudaMemsetAsync(partial_denom.data(), 0, partial_denom.bytes(),
                                   stream.get()));
        CELEG_CUDA(cudaMemsetAsync(partial_accum.data(), 0, partial_accum.bytes(),
                                   stream.get()));
        celeg::launch_gqa_decode_segmented_device({
            .query = dq.data(),
            .kv = {.keys = dk.data(), .values = dv.data()},
            .out = segmented.data(),
            .geometry = {.q_heads = q_heads, .kv_heads = kv_heads, .head_dim = head_dim},
            .extent = {.position = dposition.data()},
            .segmentation = {.segments = segments,
                             .min_segments = min_segments,
                             .partial_max = partial_max.data(),
                             .partial_denom = partial_denom.data(),
                             .partial_accum = partial_accum.data()},
            .stream = stream.get()});
        std::vector<__nv_bfloat16> b(q.size());
        CELEG_CUDA(cudaMemcpyAsync(b.data(), segmented.data(), segmented.bytes(),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        for (size_t i = 0; i < a.size(); ++i) {
            expect_near(to_float(a[i]), to_float(b[i]), 0.02f);
        }
    }
}

{
    constexpr int page_tokens = 2;
    constexpr int attention_layers = 1;
    constexpr int q_heads = 1;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 2;
    constexpr int page_count = 2;
    std::vector<__nv_bfloat16> keys(page_count * attention_layers *
                                    page_tokens * kv_heads * head_dim,
                                    to_bf16(0.0f));
    std::vector<__nv_bfloat16> values(keys.size(), to_bf16(0.0f));
    const size_t base = static_cast<size_t>(page_tokens) * head_dim;
    keys[base + 0] = to_bf16(1.0f);
    keys[base + 1] = to_bf16(0.0f);
    keys[base + 2] = to_bf16(0.0f);
    keys[base + 3] = to_bf16(1.0f);
    values[base + 0] = to_bf16(2.0f);
    values[base + 1] = to_bf16(4.0f);
    values[base + 2] = to_bf16(6.0f);
    values[base + 3] = to_bf16(8.0f);
    std::vector<__nv_bfloat16> query = {to_bf16(1.0f), to_bf16(0.0f)};
    std::vector<uint32_t> table = {1};
    std::vector<int32_t> positions = {1};
    celeg::DeviceBuffer<__nv_bfloat16> dq(query.size()), dk(keys.size()),
        dv(values.size()), dout(head_dim);
    celeg::DeviceBuffer<uint32_t> dtable(table.size());
    celeg::DeviceBuffer<int32_t> dpositions(positions.size());
    CELEG_CUDA(cudaMemcpy(dq.data(), query.data(), dq.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dk.data(), keys.data(), dk.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dv.data(), values.data(), dv.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dtable.data(), table.data(), dtable.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dpositions.data(), positions.data(), dpositions.bytes(), cudaMemcpyHostToDevice));
    celeg::launch_gqa_decode_paged_batch({
        .query = dq.data(),
        .kv = {.keys = dk.data(), .values = dv.data()},
        .index = {.page_tables = dtable.data(),
                  .page_table_stride = 1,
                  .attention_slot = 0,
                  .page_tokens = page_tokens,
                  .page_vector_elements = page_tokens * kv_heads * head_dim,
                  .layer_vector_offset = 0},
        .out = dout.data(),
        .positions = dpositions.data(),
        .rows = 1,
        .geometry = {.q_heads = q_heads, .kv_heads = kv_heads, .head_dim = head_dim},
        .fast = false,
        .stream = stream.get()});
    std::vector<__nv_bfloat16> output(head_dim);
    CELEG_CUDA(cudaMemcpyAsync(output.data(), dout.data(), dout.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    const auto expected = celeg::reference::gqa_decode_strict_bf16(
        {1.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 1.0f},
        {2.0f, 4.0f, 6.0f, 8.0f}, 2, q_heads, kv_heads, head_dim);
    for (int i = 0; i < head_dim; ++i) {
        expect_near(to_float(output[static_cast<size_t>(i)]),
                    expected[static_cast<size_t>(i)], 0.02f);
    }
}

{
    constexpr int page_tokens = 2;
    constexpr int attention_layers = 1;
    constexpr int q_heads = 1;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 2;
    constexpr int page_count = 2;
    std::vector<__nv_bfloat16> keys(page_count * attention_layers *
                                    page_tokens * kv_heads * head_dim,
                                    to_bf16(0.0f));
    std::vector<__nv_bfloat16> values(keys.size(), to_bf16(0.0f));
    const size_t p1 = static_cast<size_t>(page_tokens) * head_dim;
    keys[p1 + 0] = to_bf16(1.0f); keys[p1 + 1] = to_bf16(0.0f);
    keys[p1 + 2] = to_bf16(0.0f); keys[p1 + 3] = to_bf16(1.0f);
    values[p1 + 0] = to_bf16(2.0f); values[p1 + 1] = to_bf16(4.0f);
    values[p1 + 2] = to_bf16(6.0f); values[p1 + 3] = to_bf16(8.0f);
    keys[0] = to_bf16(0.5f); keys[1] = to_bf16(0.5f);
    values[0] = to_bf16(10.0f); values[1] = to_bf16(12.0f);
    std::vector<__nv_bfloat16> query = {to_bf16(1.0f), to_bf16(0.0f)};
    std::vector<uint32_t> table = {1, 0};
    std::vector<int32_t> positions = {2};
    celeg::DeviceBuffer<__nv_bfloat16> dq(query.size()), dk(keys.size()),
        dv(values.size()), normal(head_dim), segmented(head_dim);
    celeg::DeviceBuffer<uint32_t> dtable(table.size());
    celeg::DeviceBuffer<int32_t> dpositions(positions.size());
    celeg::DeviceBuffer<float> pmax(3), pdenom(3), paccum(3 * head_dim);
    CELEG_CUDA(cudaMemcpy(dq.data(), query.data(), dq.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dk.data(), keys.data(), dk.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dv.data(), values.data(), dv.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dtable.data(), table.data(), dtable.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dpositions.data(), positions.data(), dpositions.bytes(), cudaMemcpyHostToDevice));
    const celeg::PagedKvIndex paged_index{
        .page_tables = dtable.data(),
        .page_table_stride = 2,
        .attention_slot = 0,
        .page_tokens = page_tokens,
        .page_vector_elements = page_tokens * kv_heads * head_dim,
        .layer_vector_offset = 0};
    const celeg::GqaGeometry paged_geometry{
        .q_heads = q_heads, .kv_heads = kv_heads, .head_dim = head_dim};
    celeg::launch_gqa_decode_paged_batch({
        .query = dq.data(),
        .kv = {.keys = dk.data(), .values = dv.data()},
        .index = paged_index,
        .out = normal.data(),
        .positions = dpositions.data(),
        .rows = 1,
        .geometry = paged_geometry,
        .fast = true,
        .stream = stream.get()});
    celeg::launch_gqa_decode_paged_segmented_batch({
        .query = dq.data(),
        .kv = {.keys = dk.data(), .values = dv.data()},
        .index = paged_index,
        .out = segmented.data(),
        .positions = dpositions.data(),
        .rows = 1,
        .geometry = paged_geometry,
        .segmentation = {.chunk_tokens = 1,
                         .chunks = 3,
                         .partial_max = pmax.data(),
                         .partial_denom = pdenom.data(),
                         .partial_accum = paccum.data()},
        .stream = stream.get()});
    std::vector<__nv_bfloat16> a(head_dim), b(head_dim);
    CELEG_CUDA(cudaMemcpyAsync(a.data(), normal.data(), normal.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(b.data(), segmented.data(), segmented.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    for (int i = 0; i < head_dim; ++i) {
        expect_near(to_float(a[static_cast<size_t>(i)]),
                    to_float(b[static_cast<size_t>(i)]), 0.03f);
    }
}

{
    constexpr int page_tokens = 2;
    constexpr int attention_layers = 1;
    constexpr int q_heads = 1;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 2;
    constexpr int page_count = 2;
    const size_t vector_count = page_count * attention_layers *
                                page_tokens * kv_heads * head_dim;
    const size_t scale_count = page_count * attention_layers *
                               page_tokens * kv_heads;
    std::vector<int8_t> keys(vector_count, 0), values(vector_count, 0);
    std::vector<float> key_scales(scale_count, 0.01f);
    std::vector<float> value_scales(scale_count, 0.02f);
    const size_t p1 = static_cast<size_t>(page_tokens) * head_dim;
    keys[p1 + 0] = 100; keys[p1 + 1] = 0;
    keys[p1 + 2] = 0;   keys[p1 + 3] = 100;
    values[p1 + 0] = 100; values[p1 + 1] = 50;
    values[p1 + 2] = 25;  values[p1 + 3] = 75;
    keys[0] = 50; keys[1] = 50;
    values[0] = 80; values[1] = 40;
    std::vector<__nv_bfloat16> query = {to_bf16(1.0f), to_bf16(0.0f)};
    std::vector<uint32_t> table = {1, 0};
    std::vector<int32_t> positions = {2};
    celeg::DeviceBuffer<__nv_bfloat16> dq(query.size()), normal(head_dim),
        segmented(head_dim);
    celeg::DeviceBuffer<int8_t> dk(keys.size()), dv(values.size());
    celeg::DeviceBuffer<float> dks(key_scales.size()), dvs(value_scales.size());
    celeg::DeviceBuffer<uint32_t> dtable(table.size());
    celeg::DeviceBuffer<int32_t> dpositions(positions.size());
    celeg::DeviceBuffer<float> pmax(3), pdenom(3), paccum(3 * head_dim);
    CELEG_CUDA(cudaMemcpy(dq.data(), query.data(), dq.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dk.data(), keys.data(), dk.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dv.data(), values.data(), dv.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dks.data(), key_scales.data(), dks.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dvs.data(), value_scales.data(), dvs.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dtable.data(), table.data(), dtable.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dpositions.data(), positions.data(), dpositions.bytes(), cudaMemcpyHostToDevice));
    const celeg::Int8KvPoolView int8_pool{
        .keys = dk.data(),
        .values = dv.data(),
        .key_scales = dks.data(),
        .value_scales = dvs.data()};
    const celeg::PagedKvIndex int8_index{
        .page_tables = dtable.data(),
        .page_table_stride = 2,
        .attention_slot = 0,
        .page_tokens = page_tokens,
        .page_vector_elements = page_tokens * kv_heads * head_dim,
        .layer_vector_offset = 0};
    const celeg::PagedKvScaleIndex int8_scale_index{
        .page_scale_elements = page_tokens * kv_heads,
        .layer_scale_offset = 0};
    const celeg::GqaGeometry int8_geometry{
        .q_heads = q_heads, .kv_heads = kv_heads, .head_dim = head_dim};
    celeg::launch_gqa_decode_int8_paged_batch({
        .query = dq.data(),
        .kv = int8_pool,
        .index = int8_index,
        .scale_index = int8_scale_index,
        .out = normal.data(),
        .positions = dpositions.data(),
        .rows = 1,
        .geometry = int8_geometry,
        .fast = true,
        .stream = stream.get()});
    celeg::launch_gqa_decode_int8_paged_segmented_batch({
        .query = dq.data(),
        .kv = int8_pool,
        .index = int8_index,
        .scale_index = int8_scale_index,
        .out = segmented.data(),
        .positions = dpositions.data(),
        .rows = 1,
        .geometry = int8_geometry,
        .segmentation = {.chunk_tokens = 1,
                         .chunks = 3,
                         .partial_max = pmax.data(),
                         .partial_denom = pdenom.data(),
                         .partial_accum = paccum.data()},
        .stream = stream.get()});
    std::vector<__nv_bfloat16> a(head_dim), b(head_dim);
    CELEG_CUDA(cudaMemcpyAsync(a.data(), normal.data(), normal.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(b.data(), segmented.data(), segmented.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    for (int i = 0; i < head_dim; ++i) {
        expect_near(to_float(a[static_cast<size_t>(i)]),
                    to_float(b[static_cast<size_t>(i)]), 0.03f);
    }
}

{
    constexpr int page_tokens = 2;
    constexpr int attention_layers = 1;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 2;
    constexpr int page_count = 2;
    std::vector<__nv_bfloat16> k = {to_bf16(1.0f), to_bf16(-0.5f)};
    std::vector<__nv_bfloat16> v = {to_bf16(2.0f), to_bf16(-1.0f)};
    std::vector<uint32_t> table = {1};
    std::vector<int32_t> positions = {0};
    const size_t vector_count = page_count * attention_layers *
                                page_tokens * kv_heads * head_dim;
    const size_t scale_count = page_count * attention_layers *
                               page_tokens * kv_heads;
    celeg::DeviceBuffer<__nv_bfloat16> dk(k.size()), dv(v.size());
    celeg::DeviceBuffer<int8_t> key_pool(vector_count), value_pool(vector_count);
    celeg::DeviceBuffer<float> key_scales(scale_count), value_scales(scale_count);
    celeg::DeviceBuffer<uint32_t> dtable(table.size());
    celeg::DeviceBuffer<int32_t> dpositions(positions.size());
    key_pool.zero_async(stream.get());
    value_pool.zero_async(stream.get());
    key_scales.zero_async(stream.get());
    value_scales.zero_async(stream.get());
    CELEG_CUDA(cudaMemcpy(dk.data(), k.data(), dk.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dv.data(), v.data(), dv.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dtable.data(), table.data(), dtable.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dpositions.data(), positions.data(), dpositions.bytes(), cudaMemcpyHostToDevice));
    celeg::launch_store_kv_int8_paged_batch(
        dk.data(), dv.data(), key_pool.data(), value_pool.data(),
        key_scales.data(), value_scales.data(), dtable.data(), 1,
        dpositions.data(), 1, 0, page_tokens,
        page_tokens * kv_heads * head_dim, 0,
        page_tokens * kv_heads, 0, kv_heads, head_dim, stream.get());
    std::vector<int8_t> host_keys(vector_count), host_values(vector_count);
    std::vector<float> host_key_scales(scale_count), host_value_scales(scale_count);
    CELEG_CUDA(cudaMemcpyAsync(host_keys.data(), key_pool.data(), key_pool.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(host_values.data(), value_pool.data(), value_pool.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(host_key_scales.data(), key_scales.data(), key_scales.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(host_value_scales.data(), value_scales.data(), value_scales.bytes(),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    const size_t vector_base = static_cast<size_t>(page_tokens) * head_dim;
    const size_t scale_base = static_cast<size_t>(page_tokens);
    expect_near(host_key_scales[scale_base], 1.0f / 127.0f, 1e-4f);
    expect_near(host_value_scales[scale_base], 2.0f / 127.0f, 1e-4f);
    CELEG_TEST_CHECK(host_keys[vector_base] == 127);
    CELEG_TEST_CHECK(host_values[vector_base] == 127);
}
}

}
