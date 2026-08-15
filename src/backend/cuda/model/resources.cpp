#include "celeg/detail/model/compiled_model.hpp"

#include <algorithm>
#include <cstddef>
namespace celeg {

void CudaCompiledModel::allocate_celeg_resources() {
    sampling_.reset_vocabulary(static_cast<size_t>(resources_.dims_.vocab_size));
    workspace_.hidden_.reset(static_cast<size_t>(resources_.program_.hidden));
    workspace_.residual_.reset(static_cast<size_t>(resources_.program_.hidden));
    workspace_.normed_.reset(static_cast<size_t>(resources_.program_.hidden));
    workspace_.op_output_.reset(static_cast<size_t>(std::max(
        resources_.program_.hidden,
        resources_.shape_.maximum_attention_output_width())));
    workspace_.qkv_output_.reset(static_cast<size_t>(
        resources_.shape_.maximum_attention_projection_width()));
    workspace_.attention_gate_.reset(static_cast<size_t>(
        resources_.shape_.maximum_attention_output_width()));
    workspace_.latent_query_content_.reset(static_cast<size_t>(
        resources_.shape_.maximum_attention_output_width()));
    workspace_.latent_query_rope_.reset(static_cast<size_t>(
        resources_.shape_.maximum_attention_latent_query_rope_width()));
    workspace_.latent_key_.reset(static_cast<size_t>(
        resources_.shape_.maximum_attention_latent_rank()));
    workspace_.latent_value_.reset(static_cast<size_t>(
        resources_.shape_.maximum_attention_latent_rank()));
    workspace_.latent_key_rope_.reset(static_cast<size_t>(
        resources_.shape_.maximum_attention_latent_rope_width()));
    workspace_.latent_projection_.reset(static_cast<size_t>(
        resources_.shape_.maximum_attention_projection_width()));
    workspace_.latent_decompressed_.reset(static_cast<size_t>(
        resources_.shape_.maximum_attention_output_width()));
    workspace_.conv_projected_.reset(static_cast<size_t>(3 * resources_.program_.hidden));
    workspace_.mamba_projected_.reset(static_cast<size_t>(
        resources_.shape_.maximum_mamba_projection_width()));
    workspace_.mamba_inner_.reset(static_cast<size_t>(resources_.shape_.mamba2_intermediate));
    workspace_.gated_delta_qkv_.reset(static_cast<size_t>(resources_.shape_.max_gated_delta_net_qkv_width()));
    workspace_.gated_delta_z_.reset(static_cast<size_t>(resources_.shape_.max_gated_delta_net_output_width()));
    workspace_.gated_delta_b_.reset(static_cast<size_t>(resources_.shape_.max_gated_delta_net_gate_width()));
    workspace_.gated_delta_a_.reset(static_cast<size_t>(resources_.shape_.max_gated_delta_net_gate_width()));
    workspace_.gated_delta_output_.reset(static_cast<size_t>(resources_.shape_.max_gated_delta_net_output_width()));
    workspace_.gate_up_.reset(static_cast<size_t>(2 * resources_.shape_.max_feed_forward_intermediate));
    workspace_.activated_.reset(static_cast<size_t>(resources_.shape_.max_feed_forward_intermediate));
    workspace_.mlp_output_.reset(static_cast<size_t>(resources_.program_.hidden));
    workspace_.logits_.reset(static_cast<size_t>(resources_.dims_.vocab_size));
    if (resources_.options_.enable_mtp && resources_.dims_.mtp_num_hidden_layers > 0) {
        workspace_.mtp_embedding_.reset(static_cast<size_t>(resources_.program_.hidden));
        workspace_.mtp_hidden_norm_.reset(static_cast<size_t>(resources_.program_.hidden));
        workspace_.mtp_fused_.reset(static_cast<size_t>(2 * resources_.program_.hidden));
        workspace_.mtp_base_hidden_.reset(static_cast<size_t>(resources_.program_.hidden));
        workspace_.mtp_logits_.reset(static_cast<size_t>(resources_.dims_.vocab_size));
        workspace_.mtp_token_.reset(1);
        workspace_.mtp_candidate_.reset(1);
        workspace_.mtp_target_candidate_.reset(1);
    }
    workspace_.paged_page_table_.reset(static_cast<size_t>(max_context_));
    workspace_.paged_prefill_tokens_.reset(static_cast<size_t>(max_context_));
    if (resources_.program_.per_layer_input.enabled) {
        const size_t packed = resources_.program_.per_layer_input.packed_width;
        workspace_.per_layer_token_.reset(packed);
        workspace_.per_layer_context_.reset(packed);
        workspace_.per_layer_gate_.reset(
            static_cast<size_t>(resources_.program_.per_layer_input.input_size));
    }

    {
        int E = 1;
        int K = 1;
        int inter = 1;
        for (const CompiledLayerProgram& layer : resources_.program_.layers) {
            const auto* moe = std::get_if<MoeLayerProgram>(&layer.feed_forward);
            if (!moe) continue;
            E = std::max(E, moe->router.expert_count);
            K = std::max(K, moe->router.experts_per_token);
            inter = std::max(inter, moe->routed.mlp.intermediate_size);
        }
        workspace_.moe_hidden_float_.reset(static_cast<size_t>(resources_.program_.hidden));
        workspace_.moe_sel_.reset(static_cast<size_t>(K));
        workspace_.moe_routing_w_.reset(static_cast<size_t>(K));
        workspace_.moe_router_scratch_.reset(static_cast<size_t>(E));
        workspace_.moe_output_accum_.reset(static_cast<size_t>(resources_.program_.hidden));
        workspace_.moe_output_.reset(static_cast<size_t>(resources_.program_.hidden));
        workspace_.moe_shared_gate_.reset(1);
        workspace_.moe_gu_scratch_.reset(static_cast<size_t>(K) * 2 * inter);
        workspace_.moe_act_scratch_.reset(static_cast<size_t>(K) * inter);
        const int mtp_layers = resources_.options_.enable_mtp
            ? resources_.dims_.mtp_num_hidden_layers : 0;
        workspace_.moe_router_float_.resize(static_cast<size_t>(
            resources_.shape_.num_hidden_layers + mtp_layers));
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
    }
    gemm_ = std::make_unique<GemmDispatcher>(stream_.get(), resources_.options_);
}

}

