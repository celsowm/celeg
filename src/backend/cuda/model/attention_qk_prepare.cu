#include "attention_qk_prepare.hpp"

#include "backend/cuda/attention_norm.hpp"
#include "backend/cuda/weight_layout.hpp"
#include "kernels/kernels.cuh"
#include "kernels/rope_pairing.hpp"

#include <stdexcept>

namespace celeg {

void prepare_cuda_attention_qk(const CudaAttentionQkPreparation& preparation) {
    if (!preparation.layout || !preparation.query) {
        throw std::logic_error("CUDA attention QK preparation is incomplete");
    }

    const AttentionSpec& layout = *preparation.layout;
    launch_attention_qk_norm(
        layout,
        preparation.query,
        preparation.key,
        preparation.query_norm,
        preparation.key_norm,
        1,
        preparation.stream);

    if (const auto* rope = layout.rope_position()) {
        switch (preparation.position_mode) {
        case CudaQkPositionMode::HostScalar:
            launch_dynamic_qk_norm_rope(
                preparation.query, preparation.key, nullptr, nullptr,
                layout.query_heads, layout.key_value_heads, layout.head_dim,
                preparation.host_position,
                static_cast<float>(rope->theta),
                static_cast<float>(rope->rotary_fraction),
                preparation.norm_epsilon, false,
                lower_cuda_rope_scaling(*rope), rope->pairing,
                preparation.stream);
            break;

        case CudaQkPositionMode::DeviceScalar:
            if (!preparation.device_position) {
                throw std::logic_error("CUDA device-position attention has no position");
            }
            if (rope->pairing == RopePairingKind::AdjacentPairs) {
                launch_adjacent_qk_norm_rope_positions(
                    preparation.query, preparation.key,
                    nullptr, nullptr, 1,
                    layout.query_heads, layout.key_value_heads, layout.head_dim,
                    preparation.device_position,
                    static_cast<float>(rope->theta),
                    static_cast<float>(rope->rotary_fraction),
                    preparation.norm_epsilon, false,
                    lower_cuda_rope_scaling(*rope), preparation.stream);
            } else {
                launch_dynamic_qk_norm_rope_device(
                    preparation.query, preparation.key, nullptr, nullptr,
                    layout.query_heads, layout.key_value_heads, layout.head_dim,
                    preparation.device_position,
                    static_cast<float>(rope->theta),
                    static_cast<float>(rope->rotary_fraction),
                    preparation.norm_epsilon, false,
                    lower_cuda_rope_scaling(*rope), rope->pairing,
                    preparation.stream);
            }
            break;

        case CudaQkPositionMode::MultiAxisDevice:
            if (!preparation.device_position) {
                throw std::logic_error("CUDA M-RoPE attention has no position");
            }
            launch_dynamic_mrope_qk_norm_rope(
                preparation.query, preparation.key, nullptr, nullptr,
                layout.query_heads, layout.key_value_heads, layout.head_dim,
                preparation.device_position,
                preparation.mrope_section0,
                preparation.mrope_section1,
                preparation.mrope_section2,
                preparation.mrope_interleaved,
                static_cast<float>(rope->theta),
                static_cast<float>(rope->rotary_fraction),
                preparation.norm_epsilon, false,
                lower_cuda_rope_scaling(*rope), preparation.stream);
            break;
        }
    }

    launch_scale(
        preparation.query,
        layout.query_width(),
        cuda_query_prescale(layout),
        preparation.stream);
}

void prepare_cuda_latent_attention_qk(
    const CudaLatentQkPreparation& preparation) {
    if (!preparation.layout || !preparation.query_rope) {
        throw std::logic_error("CUDA latent QK preparation is incomplete");
    }

    const AttentionSpec& layout = *preparation.layout;
    const auto& latent = *layout.latent_state();
    const auto* rope = layout.rope_position();
    if (!rope || !preparation.key_rope || !latent.decoupled_rope ||
        latent.rope_head_dim == 0) {
        return;
    }

    const float norm_epsilon = layout.query_norm
        ? layout.query_norm->epsilon
        : preparation.fallback_norm_epsilon;

    switch (preparation.position_mode) {
    case CudaQkPositionMode::HostScalar:
        launch_dynamic_qk_norm_rope(
            preparation.query_rope, preparation.key_rope,
            nullptr, nullptr, layout.query_heads, 1, latent.rope_head_dim,
            preparation.host_position, static_cast<float>(rope->theta), 1.0f,
            norm_epsilon, false, lower_cuda_rope_scaling(*rope), rope->pairing,
            preparation.stream);
        return;

    case CudaQkPositionMode::DeviceScalar:
        if (!preparation.device_position) {
            throw std::logic_error("CUDA latent device-position attention has no position");
        }
        launch_dynamic_qk_norm_rope_device(
            preparation.query_rope, preparation.key_rope,
            nullptr, nullptr, layout.query_heads, 1, latent.rope_head_dim,
            preparation.device_position, static_cast<float>(rope->theta), 1.0f,
            norm_epsilon, false, lower_cuda_rope_scaling(*rope), rope->pairing,
            preparation.stream);
        return;

    case CudaQkPositionMode::MultiAxisDevice:
        throw std::invalid_argument("CUDA latent attention does not support M-RoPE yet");
    }

    throw std::logic_error("unknown CUDA latent QK position mode");
}

}
