#include "detail/compiled_model.hpp"
#include "kernels/kernels.cuh"

#include <stdexcept>

namespace celeg {
namespace {

void copy_device(cudaMemcpyKind kind, void* destination, const void* source,
                 size_t bytes, cudaStream_t stream) {
    if (bytes == 0) return;
    CELEG_CUDA(cudaMemcpyAsync(destination, source, bytes, kind, stream));
}

}

void CudaCompiledModel::snapshot_speculative_state() {
    if (session_.phase_ != SessionPhase::Ready) {
        throw std::runtime_error(
            "speculative state requires a completed ready session");
    }
    if (speculative_snapshot_.layers.size() != resources_.layers_.size()) {
        speculative_snapshot_.layers.resize(resources_.layers_.size());
    }

    speculative_snapshot_.position = session_.position_;
    speculative_snapshot_.next_rope_position = session_.next_rope_position_;
    speculative_snapshot_.phase = session_.phase_;
    speculative_snapshot_.active_segmented_attention =
        session_.active_segmented_attention_;
    speculative_snapshot_.mtp_candidate_ready = mtp_candidate_ready_;
    speculative_snapshot_.metrics = session_.metrics_;

    speculative_snapshot_.seen_tokens.reset(sampling_.seen_tokens.size());
    speculative_snapshot_.logits.reset(workspace_.logits_.size());
    speculative_snapshot_.mtp_logits.reset(
        resources_.mtp_.available() ? workspace_.mtp_logits_.size() : 0);
    speculative_snapshot_.mtp_candidate.reset(
        resources_.mtp_.available() ? workspace_.mtp_candidate_.size() : 0);
    speculative_snapshot_.mtp_target_candidate.reset(
        resources_.mtp_.available() ? workspace_.mtp_target_candidate_.size() : 0);
    speculative_snapshot_.rng_state.reset(sampling_.rng_state.size());
    speculative_snapshot_.position_device.reset(position_device_.size());
    speculative_snapshot_.mrope_position_device.reset(mrope_position_device_.size());

    const cudaStream_t stream = stream_.get();
    copy_device(cudaMemcpyDeviceToDevice, speculative_snapshot_.seen_tokens.data(),
                sampling_.seen_tokens.data(), sampling_.seen_tokens.bytes(), stream);
    copy_device(cudaMemcpyDeviceToDevice, speculative_snapshot_.logits.data(),
                workspace_.logits_.data(), workspace_.logits_.bytes(), stream);
    if (resources_.mtp_.available()) {
        copy_device(cudaMemcpyDeviceToDevice,
                    speculative_snapshot_.mtp_logits.data(),
                    workspace_.mtp_logits_.data(),
                    workspace_.mtp_logits_.bytes(), stream);
        copy_device(cudaMemcpyDeviceToDevice,
                    speculative_snapshot_.mtp_candidate.data(),
                    workspace_.mtp_candidate_.data(),
                    workspace_.mtp_candidate_.bytes(), stream);
        copy_device(cudaMemcpyDeviceToDevice,
                    speculative_snapshot_.mtp_target_candidate.data(),
                    workspace_.mtp_target_candidate_.data(),
                    workspace_.mtp_target_candidate_.bytes(), stream);
    }
    copy_device(cudaMemcpyDeviceToDevice, speculative_snapshot_.rng_state.data(),
                sampling_.rng_state.data(), sampling_.rng_state.bytes(), stream);
    copy_device(cudaMemcpyDeviceToDevice, speculative_snapshot_.position_device.data(),
                position_device_.data(), position_device_.bytes(), stream);
    copy_device(cudaMemcpyDeviceToDevice,
                speculative_snapshot_.mrope_position_device.data(),
                mrope_position_device_.data(), mrope_position_device_.bytes(), stream);

    for (size_t index = 0; index < resources_.layers_.size(); ++index) {
        SpeculativeLayerSnapshot& saved = speculative_snapshot_.layers[index];
        Layer& layer = resources_.layers_[index];
        const auto clear_saved = [&saved]() {
            saved.conv_state.reset(0);
            saved.ssm_state.reset(0);
            saved.recurrent_state.reset(0);
        };
        visit_layer(layer,
          [&](AttentionLayer*) { clear_saved(); },
          [&](ConvolutionLayer* convolution) {
            saved.conv_state.reset(convolution->conv_state.size());
            saved.ssm_state.reset(0);
            saved.recurrent_state.reset(0);
            copy_device(cudaMemcpyDeviceToDevice, saved.conv_state.data(),
                        convolution->conv_state.data(), convolution->conv_state.bytes(), stream);
          },
          [&](Mamba2Layer* mamba) {
            saved.conv_state.reset(mamba->conv_state.size());
            saved.ssm_state.reset(mamba->ssm_state.size());
            saved.recurrent_state.reset(0);
            copy_device(cudaMemcpyDeviceToDevice, saved.conv_state.data(),
                        mamba->conv_state.data(), mamba->conv_state.bytes(), stream);
            copy_device(cudaMemcpyDeviceToDevice, saved.ssm_state.data(),
                        mamba->ssm_state.data(), mamba->ssm_state.bytes(), stream);
          },
          [&](GatedDeltaNetLayer* gated_delta) {
            saved.conv_state.reset(gated_delta->conv_state.size());
            saved.ssm_state.reset(0);
            saved.recurrent_state.reset(gated_delta->recurrent_state.size());
            copy_device(cudaMemcpyDeviceToDevice, saved.conv_state.data(),
                        gated_delta->conv_state.data(), gated_delta->conv_state.bytes(), stream);
            copy_device(cudaMemcpyDeviceToDevice, saved.recurrent_state.data(),
                        gated_delta->recurrent_state.data(),
                        gated_delta->recurrent_state.bytes(), stream);
          },
          [&](MlpOnlyLayer*) { clear_saved(); });
    }
    CELEG_CUDA(cudaStreamSynchronize(stream));
    speculative_snapshot_.valid = true;
}

void CudaCompiledModel::restore_speculative_state() {
    if (!speculative_snapshot_.valid) {
        throw std::runtime_error("no speculative state snapshot is active");
    }
    if (speculative_snapshot_.layers.size() != resources_.layers_.size()) {
        throw std::runtime_error("speculative state topology changed");
    }

    const cudaStream_t stream = stream_.get();
    copy_device(cudaMemcpyDeviceToDevice, sampling_.seen_tokens.data(),
                speculative_snapshot_.seen_tokens.data(), sampling_.seen_tokens.bytes(), stream);
    copy_device(cudaMemcpyDeviceToDevice, workspace_.logits_.data(),
                speculative_snapshot_.logits.data(), workspace_.logits_.bytes(), stream);
    if (resources_.mtp_.available()) {
        copy_device(cudaMemcpyDeviceToDevice, workspace_.mtp_logits_.data(),
                    speculative_snapshot_.mtp_logits.data(),
                    workspace_.mtp_logits_.bytes(), stream);
        copy_device(cudaMemcpyDeviceToDevice, workspace_.mtp_candidate_.data(),
                    speculative_snapshot_.mtp_candidate.data(),
                    workspace_.mtp_candidate_.bytes(), stream);
        copy_device(cudaMemcpyDeviceToDevice,
                    workspace_.mtp_target_candidate_.data(),
                    speculative_snapshot_.mtp_target_candidate.data(),
                    workspace_.mtp_target_candidate_.bytes(), stream);
    }
    copy_device(cudaMemcpyDeviceToDevice, sampling_.rng_state.data(),
                speculative_snapshot_.rng_state.data(), sampling_.rng_state.bytes(), stream);
    copy_device(cudaMemcpyDeviceToDevice, position_device_.data(),
                speculative_snapshot_.position_device.data(), position_device_.bytes(), stream);
    copy_device(cudaMemcpyDeviceToDevice, mrope_position_device_.data(),
                speculative_snapshot_.mrope_position_device.data(),
                mrope_position_device_.bytes(), stream);

    for (size_t index = 0; index < resources_.layers_.size(); ++index) {
        const SpeculativeLayerSnapshot& saved = speculative_snapshot_.layers[index];
        Layer& layer = resources_.layers_[index];
        visit_layer(layer,
          [](AttentionLayer*) {},
          [&](ConvolutionLayer* convolution) {
            copy_device(cudaMemcpyDeviceToDevice, convolution->conv_state.data(),
                        saved.conv_state.data(), convolution->conv_state.bytes(), stream);
          },
          [&](Mamba2Layer* mamba) {
            copy_device(cudaMemcpyDeviceToDevice, mamba->conv_state.data(),
                        saved.conv_state.data(), mamba->conv_state.bytes(), stream);
            copy_device(cudaMemcpyDeviceToDevice, mamba->ssm_state.data(),
                        saved.ssm_state.data(), mamba->ssm_state.bytes(), stream);
          },
          [&](GatedDeltaNetLayer* gated_delta) {
            copy_device(cudaMemcpyDeviceToDevice, gated_delta->conv_state.data(),
                        saved.conv_state.data(), gated_delta->conv_state.bytes(), stream);
            copy_device(cudaMemcpyDeviceToDevice, gated_delta->recurrent_state.data(),
                        saved.recurrent_state.data(),
                        gated_delta->recurrent_state.bytes(), stream);
          },
          [](MlpOnlyLayer*) {});
    }
    session_.position_ = speculative_snapshot_.position;
    session_.next_rope_position_ = speculative_snapshot_.next_rope_position;
    session_.phase_ = speculative_snapshot_.phase;
    session_.active_segmented_attention_ =
        speculative_snapshot_.active_segmented_attention;
    mtp_candidate_ready_ = speculative_snapshot_.mtp_candidate_ready;
    mtp_candidate_used_ = false;
    session_.metrics_ = speculative_snapshot_.metrics;
    CELEG_CUDA(cudaStreamSynchronize(stream));
}

}
