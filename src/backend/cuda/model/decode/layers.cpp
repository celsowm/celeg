#include "detail/compiled_model.hpp"
#include "backend/cuda/attention_norm.hpp"
#include "kernels/kernels.cuh"
#include "backend/cuda/paged_kv.hpp"
#include "backend/cuda/weight_layout.hpp"
#include "backend/cuda/moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace celeg {

void CudaCompiledModel::run_token_layers(const TokenKvPolicy& kv) {
    int layer_index = 0;
    for (Layer& layer : resources_.layers_) {
        run_token_layer(layer, layer_index, kv);
        ++layer_index;
    }
}

void CudaCompiledModel::run_token_layer(Layer& layer, int layer_index,
                                        const TokenKvPolicy& kv) {
    LayerCommon& common_layer = common(layer);
    const CompiledLayerProgram& semantics =
        resources_.program_.layers.at(static_cast<size_t>(layer_index));
    const bool mixer_after = semantics.mixer_norm.after.has_value();
    const bool mixer_only =
        std::holds_alternative<std::monostate>(semantics.feed_forward);
    const bool fuse_mixer_residual = resources_.options().fused_residuals &&
        std::holds_alternative<CompiledAttentionProgram>(semantics.mixer) &&
        !mixer_after && !mixer_only;

    if (!fuse_mixer_residual) {
        CELEG_CUDA(cudaMemcpyAsync(
            workspace_.residual_.data(), workspace_.hidden_.data(), workspace_.hidden_.bytes(),
            cudaMemcpyDeviceToDevice, stream_.get()));
    }
    if (semantics.mixer_norm.before) {
        launch_rmsnorm(workspace_.hidden_.data(), common_layer.mixer_norm_before,
                       workspace_.normed_.data(), 1, resources_.program_.hidden,
                       semantics.mixer_norm.before->epsilon, stream_.get());
    } else {
        CELEG_CUDA(cudaMemcpyAsync(workspace_.normed_.data(), workspace_.hidden_.data(),
                                  workspace_.hidden_.bytes(), cudaMemcpyDeviceToDevice,
                                  stream_.get()));
    }

    run_token_mixer(layer, semantics, layer_index, kv);

    if (mixer_after) {
        launch_rmsnorm(workspace_.hidden_.data(), common_layer.mixer_norm_after,
                       workspace_.hidden_.data(), 1, resources_.program_.hidden,
                       semantics.mixer_norm.after->epsilon, stream_.get());
    }
    if (!fuse_mixer_residual) {
        launch_residual_add(workspace_.hidden_.data(), workspace_.residual_.data(),
                            resources_.program_.hidden, stream_.get());
    }
    if (!mixer_only) {
        run_mlp_decode(common_layer, layer_index);
    }
    if (std::binary_search(resources_.program_.norm_after_layers.begin(),
                           resources_.program_.norm_after_layers.end(), layer_index)) {
        launch_rmsnorm(workspace_.hidden_.data(), resources_.final_norm_,
                       workspace_.hidden_.data(), 1, resources_.program_.hidden,
                       resources_.program_.final_norm.epsilon, stream_.get());
    }
}

void CudaCompiledModel::run_token_mixer(Layer& layer,
                                        const CompiledLayerProgram& semantics,
                                        int layer_index, const TokenKvPolicy& kv) {
    visit_layer(layer,
      [&](AttentionLayer* attention) {
        const auto* compiled_attention =
            std::get_if<CompiledAttentionProgram>(&semantics.mixer);
        if (!compiled_attention) {
            throw std::logic_error("CUDA token attention has no compiled attention program");
        }
        switch (compiled_attention->execution.kind) {
        case AttentionExecutionKind::Standard:
            run_token_attention(*attention, semantics, layer_index, kv);
            return;
        case AttentionExecutionKind::Latent:
            if (!kv.paged()) {
                throw std::invalid_argument(
                    "CUDA latent attention is not implemented for contiguous host token execution");
            }
            run_token_latent_attention_paged(*attention, semantics, layer_index, kv);
            return;
        case AttentionExecutionKind::FactorizedLatent:
            throw std::invalid_argument(
                "CUDA factorized latent attention is not implemented for host token execution");
        }
        throw std::logic_error("unknown compiled CUDA token attention execution kind");
      },
      [&](GatedDeltaNetLayer* gated_delta) {
        run_token_gated_delta(*gated_delta, semantics);
      },
      [&](Mamba2Layer* mamba) {
        run_token_mamba2(*mamba, semantics, kv);
      },
      [&](MlpOnlyLayer* mlp) {
        run_token_mlp_only(*mlp);
      },
      [&](ConvolutionLayer* convolution) {
        run_token_convolution(*convolution);
      });
}

}
