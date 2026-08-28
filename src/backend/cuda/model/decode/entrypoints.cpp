#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/attention_norm.hpp"
#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/backend/cuda/moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace celeg {

void CudaCompiledModel::forward_token_host(int32_t token, bool compute_logits,
                                           const float* raw_embedding,
                                           const std::array<int32_t, 3>* rope_position) {
    if (session_.position_ >= max_context_) {
        throw std::runtime_error("context limit reached");
    }
    if (raw_embedding) {
        std::vector<__nv_bfloat16> converted(static_cast<size_t>(resources_.program_.hidden));
        for (int index = 0; index < resources_.program_.hidden; ++index) {
            converted[static_cast<size_t>(index)] = __float2bfloat16(raw_embedding[index]);
        }
        CELEG_CUDA(cudaMemcpyAsync(workspace_.hidden_.data(), converted.data(),
                                   converted.size() * sizeof(__nv_bfloat16),
                                   cudaMemcpyHostToDevice, stream_.get()));
        initialize_per_layer_input_host(resources_.dims_.token_policy.pad_token_id);
    } else {
        resources_.weight_layout_->embed_token(
            token, workspace_.hidden_.data(), resources_.program_.hidden, stream_.get());
        launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                     resources_.program_.embedding_transform.multiplier, stream_.get());
        if (resources_.program_.embedding_transform.post_norm) {
            launch_rmsnorm(workspace_.hidden_.data(), resources_.embedding_norm_,
                           workspace_.hidden_.data(), 1, resources_.program_.hidden,
                           resources_.program_.embedding_transform.post_norm->epsilon,
                           stream_.get());
        }
        initialize_per_layer_input_host(token);
    }

    TokenKvPolicy kv;
    kv.kv_layout = AttentionKvLayout::Contiguous;
    kv.position_source = AttentionPositionSource::HostScalar;
    kv.rope_position = rope_position;
    run_token_layers(kv);

    if (resources_.mtp_.available()) {
        run_mtp_forward(token, rope_position);
    }
    if (compute_logits) {
        run_token_logits();
        finalize_mtp_verification();
    }
    ++session_.position_;
    if (!rope_position) {
        for (int32_t& value : session_.next_rope_position_) ++value;
    }
}

void CudaCompiledModel::forward_token_paged_host(
    int32_t token, bool compute_logits, PhysicalPagedKvCache& paged_kv,
    const uint32_t* device_page_table, int page_table_stride) {
    if (session_.position_ >= max_context_) {
        throw std::runtime_error("context limit reached");
    }
    if (paged_kv.mode() != resources_.options_.kv_cache_mode) {
        throw std::invalid_argument("model and physical paged KV modes differ");
    }
    if (resources_.mtp_.available()) {
        throw std::runtime_error("MTP is incompatible with paged KV execution");
    }
    resources_.weight_layout_->embed_token(
        token, workspace_.hidden_.data(), resources_.program_.hidden, stream_.get());
    launch_scale(workspace_.hidden_.data(), resources_.program_.hidden,
                 resources_.program_.embedding_transform.multiplier, stream_.get());
    initialize_per_layer_input_host(token);

    TokenKvPolicy kv;
    kv.kv_layout = AttentionKvLayout::Paged;
    kv.position_source = AttentionPositionSource::DeviceCounter;
    kv.paged_kv = &paged_kv;
    kv.device_page_table = device_page_table;
    kv.page_table_stride = page_table_stride;
    run_token_layers(kv);

    if (compute_logits) run_token_logits();
    ++session_.position_;
    CELEG_CUDA(cudaMemcpyAsync(position_device_.data(), &session_.position_,
                               sizeof(session_.position_),
                               cudaMemcpyHostToDevice, stream_.get()));
}

}
