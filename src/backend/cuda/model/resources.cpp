#include "celeg/detail/model/compiled_model.hpp"

#include <cstddef>
namespace celeg {

void CudaCompiledModel::allocate_celeg_resources() {
    // Size all per-session device buffers from the runtime shape.
    sampling_.reset_vocabulary(static_cast<size_t>(resources_.shape_.vocab_size));
    workspace_.hidden_.reset(static_cast<size_t>(resources_.shape_.hidden));
    workspace_.residual_.reset(static_cast<size_t>(resources_.shape_.hidden));
    workspace_.normed_.reset(static_cast<size_t>(resources_.shape_.hidden));
    workspace_.op_output_.reset(static_cast<size_t>(resources_.shape_.hidden));
    workspace_.qkv_output_.reset(static_cast<size_t>(
        resources_.shape_.maximum_attention_projection_width()));
    workspace_.conv_projected_.reset(static_cast<size_t>(3 * resources_.shape_.hidden));
    workspace_.gate_up_.reset(static_cast<size_t>(2 * resources_.shape_.max_feed_forward_intermediate));
    workspace_.activated_.reset(static_cast<size_t>(resources_.shape_.max_feed_forward_intermediate));
    workspace_.mlp_output_.reset(static_cast<size_t>(resources_.shape_.hidden));
    workspace_.logits_.reset(static_cast<size_t>(resources_.shape_.vocab_size));
    if (resources_.shape_.has_per_layer_input) {
        const size_t packed = static_cast<size_t>(resources_.shape_.num_hidden_layers) *
            static_cast<size_t>(resources_.shape_.per_layer_input_size);
        workspace_.per_layer_token_.reset(packed);
        workspace_.per_layer_context_.reset(packed);
        workspace_.per_layer_gate_.reset(static_cast<size_t>(resources_.shape_.per_layer_input_size));
    }

    // MoE scratch (decode path: one token). Sized from the MoE topology when
    // present; harmless (tiny) for dense-only models.
    {
        const int E = resources_.shape_.num_experts > 0 ? resources_.shape_.num_experts : 1;
        const int K = resources_.shape_.experts_per_token > 0 ? resources_.shape_.experts_per_token : 1;
        const int inter = resources_.shape_.moe_intermediate > 0 ? resources_.shape_.moe_intermediate : 1;
        workspace_.moe_hidden_float_.reset(static_cast<size_t>(resources_.shape_.hidden));
        workspace_.moe_sel_.reset(static_cast<size_t>(K));
        workspace_.moe_routing_w_.reset(static_cast<size_t>(K));
        workspace_.moe_router_scratch_.reset(static_cast<size_t>(E));
        workspace_.moe_output_accum_.reset(static_cast<size_t>(resources_.shape_.hidden));
        workspace_.moe_output_.reset(static_cast<size_t>(resources_.shape_.hidden));
        workspace_.moe_gu_scratch_.reset(static_cast<size_t>(K) * 2 * inter);
        workspace_.moe_act_scratch_.reset(static_cast<size_t>(K) * inter);
        workspace_.moe_router_float_.resize(static_cast<size_t>(resources_.shape_.num_hidden_layers));
    }

    workspace_.attention_chunks_ = resources_.plan_.attention_chunks();
    if (workspace_.attention_chunks_ > 0) {
        const size_t partials =
            static_cast<size_t>(resources_.shape_.maximum_attention_query_heads()) * workspace_.attention_chunks_;
        workspace_.attention_partial_max_.reset(partials);
        workspace_.attention_partial_denom_.reset(partials);
        workspace_.attention_partial_accum_.reset(partials * resources_.shape_.maximum_attention_head_dim());
    }
    if (resources_.options_.gemm_backend == GemmBackend::CublasLt &&
        resources_.options_.lt_workspace_bytes > 0) {
        // GemmDispatcher owns the workspace internally.
    }
    gemm_ = std::make_unique<GemmDispatcher>(stream_.get(), resources_.options_);
}

} // namespace celeg

