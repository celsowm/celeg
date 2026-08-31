#include "attention_contiguous_dispatch.hpp"

#include "attention_decode_dispatch.hpp"
#include "attention_layer_support.hpp"
#include "detail/compiled_model.hpp"

namespace celeg {

void dispatch_cuda_standard_attention_contiguous(
    CudaCompiledModel& model,
    AttentionLayer& attention,
    AttentionLayer& owner,
    const AttentionCapability& plan,
    __nv_bfloat16* query) {
    const AttentionSpec& layout = attention.layout;
    const AttentionSpec& owner_layout = owner.layout;
    const auto* block_sparse = std::get_if<BlockSparsePattern>(&layout.pattern);

    dispatch_cuda_contiguous_decode_attention({
        .plan = plan,
        .position_mode = CudaDecodePositionMode::HostScalar,
        .block_sparse = block_sparse,
        .query = query,
        .bf16_kv = cuda_bf16_kv_view(owner),
        .int8_kv = cuda_int8_kv_view(owner),
        .out = model.workspace_.op_output_.data(),
        .geometry = make_cuda_gqa_geometry(layout, owner_layout),
        .extent = block_sparse
            ? AttentionExtent{.position = model.position_device_.data()}
            : AttentionExtent{.seq_len = model.session_.position_ + 1},
        .stream = model.stream_.get()});
}

}
