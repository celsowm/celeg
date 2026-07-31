#include "celeg/detail/model/impl.hpp"

#include <cstddef>
namespace celeg {

void Model::Impl::allocate_celeg_resources() {
    // Size all per-session device buffers from the runtime shape.
    seen_tokens_.reset(static_cast<size_t>(shape_.vocab_size));
    sampling_scores_.reset(static_cast<size_t>(shape_.vocab_size));
    hidden_.reset(static_cast<size_t>(shape_.hidden));
    residual_.reset(static_cast<size_t>(shape_.hidden));
    normed_.reset(static_cast<size_t>(shape_.hidden));
    op_output_.reset(static_cast<size_t>(shape_.hidden));
    qkv_output_.reset(static_cast<size_t>(shape_.qkv_width));
    conv_projected_.reset(static_cast<size_t>(3 * shape_.hidden));
    gate_up_.reset(static_cast<size_t>(2 * shape_.intermediate));
    activated_.reset(static_cast<size_t>(shape_.intermediate));
    mlp_output_.reset(static_cast<size_t>(shape_.hidden));
    logits_.reset(static_cast<size_t>(shape_.vocab_size));

    // MoE scratch (decode path: one token). Sized from the MoE topology when
    // present; harmless (tiny) for dense-only models.
    {
        const int E = shape_.num_experts > 0 ? shape_.num_experts : 1;
        const int K = shape_.experts_per_token > 0 ? shape_.experts_per_token : 1;
        const int inter = shape_.moe_intermediate > 0 ? shape_.moe_intermediate : 1;
        moe_hidden_float_.reset(static_cast<size_t>(shape_.hidden));
        moe_sel_.reset(static_cast<size_t>(K));
        moe_routing_w_.reset(static_cast<size_t>(K));
        moe_router_scratch_.reset(static_cast<size_t>(E));
        moe_output_accum_.reset(static_cast<size_t>(shape_.hidden));
        moe_output_.reset(static_cast<size_t>(shape_.hidden));
        moe_gu_scratch_.reset(static_cast<size_t>(K) * 2 * inter);
        moe_act_scratch_.reset(static_cast<size_t>(K) * inter);
        moe_router_float_.resize(static_cast<size_t>(shape_.num_hidden_layers));
    }

    attention_chunks_ = plan_.attention_chunks();
    if (attention_chunks_ > 0) {
        const size_t partials =
            static_cast<size_t>(shape_.num_attention_heads) * attention_chunks_;
        attention_partial_max_.reset(partials);
        attention_partial_denom_.reset(partials);
        attention_partial_accum_.reset(partials * shape_.head_dim);
    }
    if (options_.gemm_backend == GemmBackend::CublasLt &&
        options_.lt_workspace_bytes > 0) {
        // GemmDispatcher owns the workspace internally.
    }
    gemm_ = std::make_unique<GemmDispatcher>(stream_.get(), options_);
    initialize_rope_tables();
}

} // namespace celeg

