#include "detail/compiled_model.hpp"
#include "backend/cuda/attention_norm.hpp"
#include "kernels/kernels.cuh"
#include "backend/cuda/paged_kv.hpp"
#include "backend/cuda/weight_layout.hpp"
#include "backend/cuda/moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace celeg {

/// Mirrors prefill_detail::debug_layer_stats for the single-token decode
/// path so a CPU-vs-CUDA divergence can be bisected to the decode layer
/// where it first appears. Enabled only by CELEG_DEBUG_LAYER_STATS (plus
/// CELEG_DEBUG_HIDDEN_DIR for the per-layer dumps); never in normal runs.
void debug_token_layer_stats(CudaCompiledModel& model, int layer_index,
                             const char* stage) {
    static const bool enabled = getenv("CELEG_DEBUG_LAYER_STATS") != nullptr;
    if (!enabled) return;
    const int hidden = model.resources_.program_.hidden;
    std::vector<__nv_bfloat16> host(static_cast<size_t>(hidden));
    CELEG_CUDA(cudaStreamSynchronize(model.stream_.get()));
    CELEG_CUDA(cudaMemcpy(host.data(), model.workspace_.hidden_.data(),
                          host.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));
    double sq = 0.0;
    float mx = 0.0f;
    bool bad = false;
    for (const __nv_bfloat16 value : host) {
        const float element = __bfloat162float(value);
        sq += static_cast<double>(element) * element;
        mx = std::max(mx, std::fabs(element));
        if (!std::isfinite(element)) bad = true;
    }
    fprintf(stderr, "[cuda tok layer %d %s] norm=%.4f max=%.4f bad=%d\n",
            layer_index, stage, std::sqrt(sq), mx, bad ? 1 : 0);
    const char* dump_dir = getenv("CELEG_DEBUG_HIDDEN_DIR");
    if (!dump_dir) return;
    std::vector<float> row(static_cast<size_t>(hidden));
    for (int i = 0; i < hidden; ++i) {
        row[static_cast<size_t>(i)] = __bfloat162float(host[static_cast<size_t>(i)]);
    }
    char path[512];
    snprintf(path, sizeof(path), "%s/cuda_tok_layer%d_%s.f32", dump_dir,
             layer_index, stage);
    if (FILE* out = fopen(path, "wb")) {
        fwrite(row.data(), sizeof(float), row.size(), out);
        fclose(out);
    }
}

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
    debug_token_layer_stats(*this, layer_index, "mixer-out");

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
    debug_token_layer_stats(*this, layer_index, "post-mlp");
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
