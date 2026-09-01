#include "detail/compiled_model.hpp"
#include "attention_layer_support.hpp"
#include "attention_projection.hpp"
#include "residual_fusion.hpp"
#include "kernels/kernels.cuh"

#include <stdexcept>

namespace celeg {
namespace {

struct CudaStandardProjectionBuffers {
    const __nv_bfloat16* input = nullptr;
    __nv_bfloat16* query = nullptr;
    __nv_bfloat16* key = nullptr;
    __nv_bfloat16* value = nullptr;
    int rows = 1;
};

void project_cuda_standard_attention_qkv_impl(
    CudaCompiledModel& model, AttentionLayer& attention,
    const CudaStandardProjectionBuffers& buffers) {
    if (!attention.query || !attention.out) {
        throw std::logic_error(
            "CUDA standard attention has incomplete query/output bindings");
    }
    if ((attention.key == nullptr) != (attention.value == nullptr)) {
        throw std::logic_error(
            "CUDA standard attention must bind key/value together");
    }
    if (!buffers.input || !buffers.query) {
        throw std::logic_error("CUDA standard attention projection is unavailable");
    }
    const AttentionSpec& layout = attention.layout;
    const int hidden = model.resources_.program_.hidden;
    auto native_fanout = model.native_fanout_scope(
        buffers.input, buffers.rows, hidden);
    model.linear(
        buffers.input, *attention.query, buffers.query,
        buffers.rows, attention.query->rows, hidden);
    if (attention.key) {
        if (!buffers.key || !buffers.value) {
            throw std::logic_error("CUDA standard attention KV projection storage is unavailable");
        }
        model.linear(
            buffers.input, *attention.key, buffers.key,
            buffers.rows, layout.key_value_width(), hidden);
        model.linear(
            buffers.input, *attention.value, buffers.value,
            buffers.rows, layout.key_value_width(), hidden);
    }
}

struct CudaLatentProjectionBuffers {
    const __nv_bfloat16* input = nullptr;
    __nv_bfloat16* query_content = nullptr;
    __nv_bfloat16* query_rope = nullptr;
    __nv_bfloat16* key = nullptr;
    __nv_bfloat16* value = nullptr;
    __nv_bfloat16* key_rope = nullptr;
    int rows = 1;
};

void project_cuda_latent_attention_qkv_impl(
    CudaCompiledModel& model, AttentionLayer& attention,
    const CudaLatentProjectionBuffers& buffers) {
    require_cuda_projected_latent_bindings(attention);
    const AttentionSpec& layout = attention.layout;
    const auto& latent = *layout.latent_state();
    const int hidden = model.resources_.program_.hidden;
    auto native_fanout = model.native_fanout_scope(
        buffers.input, buffers.rows, hidden);
    model.linear(
        buffers.input, *attention.latent_query, buffers.query_content,
        buffers.rows, layout.latent_query_content_width(), hidden);
    if (layout.latent_query_rope_width() != 0) {
        model.linear(
            buffers.input, *attention.latent_query_rope, buffers.query_rope,
            buffers.rows, layout.latent_query_rope_width(), hidden);
    }
    if (attention.latent_key && attention.latent_value) {
        model.linear(
            buffers.input, *attention.latent_key, buffers.key,
            buffers.rows, latent.latent_rank, hidden);
        model.linear(
            buffers.input, *attention.latent_value, buffers.value,
            buffers.rows, latent.latent_rank, hidden);
        if (attention.latent_key_rope && latent.decoupled_rope &&
            latent.rope_head_dim != 0) {
            model.linear(
                buffers.input, *attention.latent_key_rope, buffers.key_rope,
                buffers.rows, latent.rope_head_dim, hidden);
        }
    }
}

struct CudaFactorizedLatentProjectionBuffers {
    const __nv_bfloat16* input = nullptr;
    __nv_bfloat16* low_rank_query = nullptr;
    __nv_bfloat16* query_projection = nullptr;
    __nv_bfloat16* query_content = nullptr;
    __nv_bfloat16* query_rope = nullptr;
    __nv_bfloat16* key = nullptr;
    __nv_bfloat16* value = nullptr;
    __nv_bfloat16* key_rope = nullptr;
    int rows = 1;
};

void project_cuda_factorized_latent_attention_qkv_impl(
    CudaCompiledModel& model, AttentionLayer& attention,
    const CudaFactorizedLatentProjectionBuffers& buffers) {
    const AttentionSpec& layout = attention.layout;
    const auto& latent = *layout.latent_state();
    const auto& factorized = *latent.factorized_projection();
    const auto& latent_expansion_storage =
        std::get<Bf16LinearStorage>(attention.latent_expansion->storage);
    const int hidden = model.resources_.program_.hidden;

    auto native_fanout = model.native_fanout_scope(
        buffers.input, buffers.rows, hidden);
    model.linear(
        buffers.input, *attention.latent_query_projection,
        buffers.low_rank_query, buffers.rows, factorized.query_rank, hidden);
    launch_rmsnorm(
        buffers.low_rank_query, attention.latent_query_norm,
        buffers.low_rank_query, buffers.rows, factorized.query_rank,
        factorized.query_latent_norm.epsilon, model.stream_.get());
    model.linear(
        buffers.low_rank_query, *attention.latent_query_expansion,
        buffers.query_projection, buffers.rows,
        layout.query_heads * (latent.nope_head_dim + latent.rope_head_dim),
        factorized.query_rank);
    launch_factorized_latent_query({
        .query_projection = buffers.query_projection,
        .expansion = latent_expansion_storage.data,
        .query_content = buffers.query_content,
        .rows = buffers.rows,
        .query_heads = layout.query_heads,
        .query_nope = latent.nope_head_dim,
        .query_rope_dim = latent.rope_head_dim,
        .latent_rank = latent.latent_rank,
        .stream = model.stream_.get()});
    launch_factorized_latent_rope({
        .query_projection = buffers.query_projection,
        .query_rope = buffers.query_rope,
        .rows = buffers.rows,
        .query_heads = layout.query_heads,
        .query_nope = latent.nope_head_dim,
        .query_rope_dim = latent.rope_head_dim,
        .stream = model.stream_.get()});
    model.linear(
        buffers.input, *attention.latent_key_projection,
        buffers.query_projection, buffers.rows,
        latent.latent_rank + latent.rope_head_dim, hidden);
    launch_rmsnorm(
        buffers.query_projection, attention.latent_key_norm,
        buffers.key, buffers.rows, latent.latent_rank,
        factorized.key_latent_norm.epsilon, model.stream_.get());
    CELEG_CUDA(cudaMemcpyAsync(
        buffers.value, buffers.key,
        static_cast<size_t>(buffers.rows) * latent.latent_rank *
            sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToDevice, model.stream_.get()));
    CELEG_CUDA(cudaMemcpy2DAsync(
        buffers.key_rope,
        static_cast<size_t>(latent.rope_head_dim) * sizeof(__nv_bfloat16),
        buffers.query_projection + latent.latent_rank,
        static_cast<size_t>(latent.latent_rank + latent.rope_head_dim) *
            sizeof(__nv_bfloat16),
        static_cast<size_t>(latent.rope_head_dim) * sizeof(__nv_bfloat16),
        static_cast<size_t>(buffers.rows), cudaMemcpyDeviceToDevice,
        model.stream_.get()));
}

struct CudaFactorizedLatentOutputBuffers {
    const __nv_bfloat16* normalized_input = nullptr;
    const __nv_bfloat16* latent_output = nullptr;
    __nv_bfloat16* decompressed = nullptr;
    __nv_bfloat16* gate = nullptr;
    __nv_bfloat16* hidden = nullptr;
    int rows = 1;
};

void project_cuda_factorized_latent_attention_output_impl(
    CudaCompiledModel& model, AttentionLayer& attention,
    const CompiledLayerProgram& semantics,
    const CudaFactorizedLatentOutputBuffers& buffers) {
    const AttentionSpec& layout = attention.layout;
    const auto& latent = *layout.latent_state();
    const auto& factorized = *latent.factorized_projection();
    const auto& latent_expansion_storage =
        std::get<Bf16LinearStorage>(attention.latent_expansion->storage);
    const int hidden = model.resources_.program_.hidden;

    launch_factorized_latent_value({
        .latent_output = buffers.latent_output,
        .expansion = latent_expansion_storage.data,
        .value_output = buffers.decompressed,
        .rows = buffers.rows,
        .query_heads = layout.query_heads,
        .query_nope = latent.nope_head_dim,
        .value_dim = factorized.value_head_dim,
        .latent_rank = latent.latent_rank,
        .stream = model.stream_.get()});
    if (layout.output_gate.has_value()) {
        model.linear(
            buffers.normalized_input, *attention.gate, buffers.gate,
            buffers.rows, layout.output_gate_width(), hidden);
        if (layout.output_gate->granularity == AttentionGateGranularity::HeadWise) {
            launch_sigmoid_multiply_headwise(
                buffers.decompressed, buffers.gate, buffers.rows,
                layout.query_heads, factorized.value_head_dim, model.stream_.get());
        } else {
            launch_sigmoid_multiply(
                buffers.decompressed, buffers.gate,
                buffers.rows * layout.latent_output_width(), model.stream_.get());
        }
    }

    const bool fuse_residual = cuda_can_fuse_mixer_residual(
        model.resources_.options().fused_residuals, semantics);
    model.linear(
        buffers.decompressed, *attention.out, buffers.hidden,
        buffers.rows, hidden, layout.latent_output_width(),
        fuse_residual ? 1.0f : 0.0f);
    launch_scale(
        buffers.hidden, buffers.rows * hidden,
        semantics.residual.multiplier, model.stream_.get());
}

void project_cuda_prefill_attention_output_impl(
    CudaCompiledModel& model, AttentionLayer& attention,
    const CompiledLayerProgram& semantics, int rows, int input_width) {
    const int hidden = model.resources_.program_.hidden;
    const bool fuse_residual = cuda_can_fuse_mixer_residual(
        model.resources_.options().fused_residuals, semantics);
    model.linear(
        model.workspace_.prefill_op_output_.data(), *attention.out,
        model.workspace_.prefill_hidden_.data(), rows, hidden, input_width,
        fuse_residual ? 1.0f : 0.0f);
    launch_scale(
        model.workspace_.prefill_hidden_.data(), rows * hidden,
        semantics.residual.multiplier, model.stream_.get());
}

}

CudaQkvProjectionView CudaCompiledModel::project_standard_attention_qkv(
    AttentionLayer& attention) {
    CudaQkvProjectionView qkv = make_cuda_qkv_projection_view(
        attention, workspace_.qkv_output_.data());
    project_cuda_standard_attention_qkv_impl(*this, attention, {
        .input = workspace_.normed_.data(),
        .query = qkv.query,
        .key = qkv.key,
        .value = qkv.value,
        .rows = 1});
    return qkv;
}

void project_cuda_prefill_standard_attention_qkv(
    CudaCompiledModel& model, AttentionLayer& attention, int rows) {
    const AttentionSpec& layout = attention.layout;
    const bool gate_packed = layout.output_gate.has_value() &&
        layout.output_gate->packed_with_query;
    project_cuda_standard_attention_qkv_impl(model, attention, {
        .input = model.workspace_.prefill_normed_.data(),
        .query = gate_packed
            ? model.workspace_.prefill_qkv_.data()
            : model.workspace_.prefill_q_.data(),
        .key = model.workspace_.prefill_k_.data(),
        .value = model.workspace_.prefill_v_.data(),
        .rows = rows});
}

void require_cuda_projected_latent_bindings(const AttentionLayer& attention) {
    const AttentionSpec& layout = attention.layout;
    if (!attention.latent_query || !attention.out) {
        throw std::logic_error(
            "CUDA projected latent attention has incomplete query/output bindings");
    }
    if (layout.latent_query_rope_width() != 0 && !attention.latent_query_rope) {
        throw std::logic_error(
            "CUDA projected latent attention is missing the query RoPE projection");
    }
    if ((attention.latent_key == nullptr) != (attention.latent_value == nullptr)) {
        throw std::logic_error(
            "CUDA projected latent attention must bind latent key/value together");
    }
}

void project_cuda_latent_attention_qkv(
    CudaCompiledModel& model, AttentionLayer& attention) {
    project_cuda_latent_attention_qkv_impl(model, attention, {
        .input = model.workspace_.normed_.data(),
        .query_content = model.workspace_.latent_query_content_.data(),
        .query_rope = model.workspace_.latent_query_rope_.data(),
        .key = model.workspace_.latent_key_.data(),
        .value = model.workspace_.latent_value_.data(),
        .key_rope = model.workspace_.latent_key_rope_.data(),
        .rows = 1});
}

void project_cuda_prefill_latent_attention_qkv(
    CudaCompiledModel& model, AttentionLayer& attention, int rows) {
    project_cuda_latent_attention_qkv_impl(model, attention, {
        .input = model.workspace_.prefill_normed_.data(),
        .query_content = model.workspace_.prefill_latent_query_content_.data(),
        .query_rope = model.workspace_.prefill_latent_query_rope_.data(),
        .key = model.workspace_.prefill_latent_key_.data(),
        .value = model.workspace_.prefill_latent_value_.data(),
        .key_rope = model.workspace_.prefill_latent_key_rope_.data(),
        .rows = rows});
}

void project_cuda_factorized_latent_attention_qkv(
    CudaCompiledModel& model, AttentionLayer& attention) {
    project_cuda_factorized_latent_attention_qkv_impl(model, attention, {
        .input = model.workspace_.normed_.data(),
        .low_rank_query = model.workspace_.latent_projection_.data(),
        .query_projection = model.workspace_.qkv_output_.data(),
        .query_content = model.workspace_.latent_query_content_.data(),
        .query_rope = model.workspace_.latent_query_rope_.data(),
        .key = model.workspace_.latent_key_.data(),
        .value = model.workspace_.latent_value_.data(),
        .key_rope = model.workspace_.latent_key_rope_.data(),
        .rows = 1});
}

void project_cuda_prefill_factorized_latent_attention_qkv(
    CudaCompiledModel& model, AttentionLayer& attention, int rows) {
    project_cuda_factorized_latent_attention_qkv_impl(model, attention, {
        .input = model.workspace_.prefill_normed_.data(),
        .low_rank_query = model.workspace_.prefill_latent_projection_.data(),
        .query_projection = model.workspace_.prefill_qkv_.data(),
        .query_content = model.workspace_.prefill_latent_query_content_.data(),
        .query_rope = model.workspace_.prefill_latent_query_rope_.data(),
        .key = model.workspace_.prefill_latent_key_.data(),
        .value = model.workspace_.prefill_latent_value_.data(),
        .key_rope = model.workspace_.prefill_latent_key_rope_.data(),
        .rows = rows});
}

void project_cuda_prefill_standard_attention_output(
    CudaCompiledModel& model, AttentionLayer& attention,
    const CompiledLayerProgram& semantics, int rows) {
    project_cuda_prefill_attention_output_impl(
        model, attention, semantics, rows, attention.layout.query_width());
}

void project_cuda_prefill_latent_attention_output(
    CudaCompiledModel& model, AttentionLayer& attention,
    const CompiledLayerProgram& semantics, int rows) {
    project_cuda_prefill_attention_output_impl(
        model, attention, semantics, rows,
        attention.layout.latent_query_content_width());
}

void project_cuda_factorized_latent_attention_output(
    CudaCompiledModel& model, AttentionLayer& attention,
    const CompiledLayerProgram& semantics) {
    project_cuda_factorized_latent_attention_output_impl(model, attention, semantics, {
        .normalized_input = model.workspace_.normed_.data(),
        .latent_output = model.workspace_.op_output_.data(),
        .decompressed = model.workspace_.latent_decompressed_.data(),
        .gate = model.workspace_.attention_gate_.data(),
        .hidden = model.workspace_.hidden_.data(),
        .rows = 1});
}

void project_cuda_prefill_factorized_latent_attention_output(
    CudaCompiledModel& model, AttentionLayer& attention,
    const CompiledLayerProgram& semantics, int rows) {
    project_cuda_factorized_latent_attention_output_impl(model, attention, semantics, {
        .normalized_input = model.workspace_.prefill_normed_.data(),
        .latent_output = model.workspace_.prefill_op_output_.data(),
        .decompressed = model.workspace_.prefill_latent_decompressed_.data(),
        .gate = model.workspace_.prefill_attention_gate_.data(),
        .hidden = model.workspace_.prefill_hidden_.data(),
        .rows = rows});
}

void CudaCompiledModel::project_standard_attention_output(
    AttentionLayer& attention, const CompiledLayerProgram& semantics) {
    const bool fuse_residual = cuda_can_fuse_mixer_residual(
        resources_.options().fused_residuals, semantics);
    linear(workspace_.op_output_.data(), *attention.out, workspace_.hidden_.data(),
           1, resources_.program_.hidden, attention.layout.query_width(),
           fuse_residual ? 1.0f : 0.0f);
    launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                 semantics.residual.multiplier, stream_.get());
}

void CudaCompiledModel::project_latent_attention_output(
    AttentionLayer& attention, const CompiledLayerProgram& semantics) {
    const bool fuse_residual = cuda_can_fuse_mixer_residual(
        resources_.options().fused_residuals, semantics);
    linear(workspace_.op_output_.data(), *attention.out, workspace_.hidden_.data(),
           1, resources_.program_.hidden,
           attention.layout.latent_query_content_width(),
           fuse_residual ? 1.0f : 0.0f);
    launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                 semantics.residual.multiplier, stream_.get());
}

}
