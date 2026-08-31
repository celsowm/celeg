#include "attention_token_qk.hpp"

#include "attention_qk_prepare.hpp"
#include "detail/compiled_model.hpp"

namespace celeg {

void prepare_cuda_token_attention_qk(
    CudaCompiledModel& model,
    AttentionLayer& attention,
    __nv_bfloat16* query,
    __nv_bfloat16* key,
    bool paged,
    const std::array<int32_t, 3>* rope_position) {
    const AttentionSpec& layout = attention.layout;
    const float qk_epsilon = layout.query_norm
        ? layout.query_norm->epsilon
        : (layout.key_norm
            ? layout.key_norm->epsilon
            : model.resources_.program_.final_norm.epsilon);

    const auto* multi = paged ? nullptr : layout.multi_axis_position();
    if (layout.rope_position() && multi) {
        const auto& position = rope_position
            ? *rope_position
            : model.session_.next_rope_position_;
        CELEG_CUDA(cudaMemcpyAsync(
            model.mrope_position_device_.data(),
            position.data(),
            sizeof(position),
            cudaMemcpyHostToDevice,
            model.stream_.get()));
    }

    CudaAttentionQkPreparation preparation{
        .layout = &layout,
        .query = query,
        .key = attention.key ? key : nullptr,
        .query_norm = attention.q_norm,
        .key_norm = attention.k_norm,
        .norm_epsilon = qk_epsilon,
        .position_mode = multi
            ? CudaQkPositionMode::MultiAxisDevice
            : CudaQkPositionMode::HostScalar,
        .host_position = model.session_.position_,
        .device_position = multi ? model.mrope_position_device_.data() : nullptr,
        .stream = model.stream_.get()};
    if (multi) {
        preparation.mrope_section0 = multi->sections[0];
        preparation.mrope_section1 = multi->sections[1];
        preparation.mrope_section2 = multi->sections[2];
        preparation.mrope_interleaved = multi->interleaved;
    }
    prepare_cuda_attention_qk(preparation);
}

}
