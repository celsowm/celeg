#include "attention_output_gate.hpp"

#include "detail/compiled_model.hpp"
#include "kernels/kernels.cuh"

#include <stdexcept>

namespace celeg {

__nv_bfloat16* prepare_cuda_token_attention_gate(
    CudaCompiledModel& model, AttentionLayer& attention, __nv_bfloat16* query) {
    const AttentionSpec& layout = attention.layout;
    if (!layout.output_gate.has_value()) return query;

    const bool packed = layout.output_gate->packed_with_query;
    if (!packed) {
        if (!attention.gate) {
            throw std::logic_error("CUDA token attention is missing its output gate binding");
        }
        return query;
    }

    launch_extract_attention_output_gate(
        query,
        model.workspace_.q_.data(),
        model.workspace_.attention_gate_.data(),
        1,
        layout.query_width(),
        layout.head_dim,
        model.stream_.get());
    return model.workspace_.q_.data();
}

void apply_cuda_token_attention_gate(
    CudaCompiledModel& model, AttentionLayer& attention) {
    const AttentionSpec& layout = attention.layout;
    if (!layout.output_gate.has_value()) return;

    if (!layout.output_gate->packed_with_query) {
        if (!attention.gate) {
            throw std::logic_error("CUDA token attention is missing its output gate binding");
        }
        model.linear(
            model.workspace_.normed_.data(),
            *attention.gate,
            model.workspace_.attention_gate_.data(),
            1,
            layout.query_width(),
            model.resources_.program_.hidden);
    }

    launch_sigmoid_multiply(
        model.workspace_.op_output_.data(),
        model.workspace_.attention_gate_.data(),
        layout.query_width(),
        model.stream_.get());
}

}
