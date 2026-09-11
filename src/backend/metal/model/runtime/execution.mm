#include "detail.hpp"
#include "mixer_registry.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace celeg {

/// @brief Temporary debug hook: cap Metal decode layers via
/// `CELEG_METAL_DEBUG_MAX_LAYERS` to bisect non-finite logits layer by layer.
/// Unset (or negative) means no cap. This is a local diagnostic, not a feature.
size_t debug_max_layers() {
    const char* raw = std::getenv("CELEG_METAL_DEBUG_MAX_LAYERS");
    if (!raw || !*raw) return static_cast<size_t>(-1);
    return static_cast<size_t>(std::atoi(raw));
}

void MetalModel::Impl::encode_token(id<MTLCommandBuffer>& command_buffer,
                                    id<MTLComputeCommandEncoder>& encoder, int32_t token,
                                    const std::array<int32_t, 3>* rope_position) {
        if (position >= max_context) throw std::runtime_error("Metal context limit reached");
        const uint32_t hidden_width = static_cast<uint32_t>(model.graph.hidden);
        const uint32_t token_value = static_cast<uint32_t>(token);
        encode_embedding(encoder, hidden_width, token_value);
        if (model.graph.embedding_transform.multiplier != 1.0f) {
            const uint32_t count = hidden_width;
            const float multiplier = model.graph.embedding_transform.multiplier;
            set_buffer(encoder, hidden, 0);
            set_bytes(encoder, &count, sizeof(count), 1);
            set_bytes(encoder, &multiplier, sizeof(multiplier), 2);
            dispatch(encoder, "celeg_scale", count);
        }
        if (model.graph.embedding_transform.post_norm) {
            encode_rmsnorm(encoder, hidden, final_norm, operation, hidden_width,
                           model.graph.embedding_transform.post_norm->epsilon);
            std::swap(hidden, operation);
        }

        bool normed_ready = false;
        const size_t layer_cap = debug_max_layers();
        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            if (layer_index >= layer_cap) break;
            Layer& layer = layers[layer_index];
            const CompiledLayerProgram& program_layer = program.layers[layer_index];
            if (!normed_ready) {
                encode_rmsnorm_save(encoder, hidden, residual, layer.operator_norm, normed,
                               hidden_width, model.graph.final_norm.epsilon);
            }
            normed_ready = false;

            mixer_encode_token(layer, program_layer, encoder, *this, rope_position);

            const uint32_t count = hidden_width;
            const float mixer_multiplier = 1.0f;
            const bool last_layer = layer_index + 1 == layers.size();
            if (std::holds_alternative<std::monostate>(program_layer.feed_forward)) {
                if (last_layer) {
                    encode_residual_rmsnorm(encoder, hidden, residual, final_norm, hidden,
                                            normed, hidden_width, mixer_multiplier,
                                            program.final_norm.epsilon);
                } else {
                    Layer& next_layer = layers[layer_index + 1];
                    encode_residual_rmsnorm_save(
                        encoder, hidden, residual, next_layer.operator_norm, hidden,
                        residual, normed, hidden_width, mixer_multiplier,
                        model.graph.final_norm.epsilon);
                }
                normed_ready = true;
                continue;
            }

            constexpr bool gate_up_ready = false;
            encode_residual_rmsnorm(encoder, hidden, residual, layer.ffn_norm, hidden,
                                    normed, hidden_width, mixer_multiplier,
                                    program.final_norm.epsilon);
            if (layer.moe) {
                encode_moe(command_buffer, encoder, layer);
            } else {
                encode_dense_feed_forward(encoder, layer, gate_up_ready);
            }
            id<MTLBuffer> feed_forward_output = layer.moe ? moe_output : operation;
            if (last_layer) {
                encode_residual_rmsnorm(encoder, feed_forward_output, hidden, final_norm,
                                        hidden, normed, hidden_width, mixer_multiplier,
                                        program.final_norm.epsilon);
            } else {
                Layer& next_layer = layers[layer_index + 1];
                encode_residual_rmsnorm_save(
                    encoder, feed_forward_output, hidden, next_layer.operator_norm, hidden,
                    residual, normed, hidden_width, mixer_multiplier,
                    model.graph.final_norm.epsilon);
            }
            normed_ready = true;
        }

        if (!normed_ready) {
            encode_rmsnorm(encoder, hidden, final_norm, normed, hidden_width,
                           program.final_norm.epsilon);
        }
        encode_matvec(encoder, embedding, normed, logits);
        ++position;
}

void MetalModel::Impl::run_token(int32_t token,
                                 const std::array<int32_t, 3>* rope_position) {
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    begin_commands(command_buffer, encoder);
    encode_token(command_buffer, encoder, token, rope_position);
    finish_commands(command_buffer, encoder);
    apply_logits_transforms();
    if (!rope_position) {
        for (int32_t& value : next_rope_position) ++value;
    }
}

bool MetalModel::Impl::supports_prefill_batch() const {
    if (program.per_layer_input.enabled) return false;
    for (const CompiledLayerProgram& layer : program.layers) {
        if (!mixer_supports_batch(layer)) return false;
    }
    return true;
}

void MetalModel::Impl::encode_prefill_batch(
    id<MTLComputeCommandEncoder>& encoder,
    const std::vector<int32_t>& tokens,
    std::span<const std::array<int32_t, 3>> rope_positions) {
    const uint32_t rows = static_cast<uint32_t>(tokens.size());
    if (!rope_positions.empty() && rope_positions.size() != tokens.size()) {
        throw std::invalid_argument("Metal batched RoPE positions must cover every token");
    }
    if (!batch_rope_positions) {
        batch_rope_positions = zero_buffer(
            static_cast<size_t>(max_context) * 3 * sizeof(int32_t));
    }
    auto* staged_positions = static_cast<int32_t*>(batch_rope_positions.contents);
    for (uint32_t row = 0; row < rows; ++row) {
        const std::array<int32_t, 3> resolved = rope_positions.empty()
            ? std::array<int32_t, 3>{position + static_cast<int32_t>(row),
                                     position + static_cast<int32_t>(row),
                                     position + static_cast<int32_t>(row)}
            : rope_positions[row];
        std::memcpy(staged_positions + static_cast<size_t>(row) * 3,
                    resolved.data(), 3 * sizeof(int32_t));
    }

    const uint32_t hidden_width = static_cast<uint32_t>(model.graph.hidden);
    const uint32_t count = rows * hidden_width;
    const uint32_t base_position = static_cast<uint32_t>(position);
    const auto encode_residual_norm = [&](id<MTLBuffer> input,
                                          id<MTLBuffer> residual,
                                          id<MTLBuffer> weight,
                                          id<MTLBuffer> output,
                                          id<MTLBuffer> normed_output,
                                          float multiplier,
                                          float epsilon) {
        set_buffer(encoder, input, 0);
        set_buffer(encoder, residual, 1);
        set_buffer(encoder, weight, 2);
        set_buffer(encoder, output, 3);
        set_buffer(encoder, normed_output, 4);
        set_bytes(encoder, &rows, sizeof(rows), 5);
        set_bytes(encoder, &hidden_width, sizeof(hidden_width), 6);
        set_bytes(encoder, &multiplier, sizeof(multiplier), 7);
        set_bytes(encoder, &epsilon, sizeof(epsilon), 8);

        const NSUInteger scratch_bytes =
            static_cast<NSUInteger>(hidden_width) * sizeof(float);
        id<MTLComputePipelineState> cached_state =
            pipeline("celeg_residual_rmsnorm_batch_cached");
        const bool use_cached = cached_state.staticThreadgroupMemoryLength + scratch_bytes <=
            device.maxThreadgroupMemoryLength;
        const std::string_view kernel = use_cached
            ? "celeg_residual_rmsnorm_batch_cached"
            : "celeg_residual_rmsnorm_batch";
        id<MTLComputePipelineState> state = use_cached ? cached_state : pipeline(kernel);
        encoder = compute_encoder(encoder);
        order_before_dispatch(encoder);
        [encoder setComputePipelineState:state];
        if (use_cached) {
            [encoder setThreadgroupMemoryLength:scratch_bytes atIndex:0];
        }
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        record_dispatch(kernel);
    };

    encode_embedding_batch(encoder, rows, tokens);
    if (model.graph.embedding_transform.multiplier != 1.0f) {
        const float multiplier = model.graph.embedding_transform.multiplier;
        set_buffer(encoder, batch_hidden, 0);
        set_bytes(encoder, &count, sizeof(count), 1);
        set_bytes(encoder, &multiplier, sizeof(multiplier), 2);
        dispatch(encoder, "celeg_scale", count);
    }
    if (model.graph.embedding_transform.post_norm) {
        encode_rmsnorm_batch(encoder, batch_hidden, final_norm, batch_normed,
                             rows, hidden_width,
                             model.graph.embedding_transform.post_norm->epsilon);
        std::swap(batch_hidden, batch_normed);
    }

    bool normed_ready = false;
    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        Layer& layer = layers[layer_index];
        const CompiledLayerProgram& program_layer = program.layers[layer_index];
        if (!normed_ready) {
            encode_rmsnorm_batch(encoder, batch_hidden, layer.operator_norm, batch_normed,
                                 rows, hidden_width, model.graph.final_norm.epsilon);
        }
        normed_ready = false;

        // Preserve the current hidden state as the residual without copying it.
        // The mixer writes a complete replacement into the other scratch buffer.
        std::swap(batch_hidden, batch_residual);
        if (layer.mixer_kind == Layer::MixerKind::ShortConvolution) {
            encode_matmul(encoder, layer.mixer_in, batch_normed, batch_projected, rows);
        }
        mixer_encode_batch(layer, program_layer, encoder, rows, base_position, *this);

        constexpr float mixer_multiplier = 1.0f;
        const bool last_layer = layer_index + 1 == layers.size();
        if (std::holds_alternative<std::monostate>(program_layer.feed_forward)) {
            if (last_layer) {
                encode_residual_norm(batch_hidden, batch_residual, final_norm,
                                     batch_hidden, batch_normed, mixer_multiplier,
                                     program.final_norm.epsilon);
            } else {
                Layer& next_layer = layers[layer_index + 1];
                encode_residual_norm(batch_hidden, batch_residual,
                                     next_layer.operator_norm, batch_hidden,
                                     batch_normed, mixer_multiplier,
                                     model.graph.final_norm.epsilon);
            }
            normed_ready = true;
            continue;
        }

        encode_residual_norm(batch_hidden, batch_residual, layer.ffn_norm,
                             batch_hidden, batch_normed, mixer_multiplier,
                             program.final_norm.epsilon);
        encode_dense_feed_forward_batch(encoder, layer, rows);
        if (last_layer) {
            encode_residual_norm(batch_operation, batch_hidden, final_norm,
                                 batch_hidden, batch_normed, mixer_multiplier,
                                 program.final_norm.epsilon);
        } else {
            Layer& next_layer = layers[layer_index + 1];
            encode_residual_norm(batch_operation, batch_hidden,
                                 next_layer.operator_norm, batch_hidden,
                                 batch_normed, mixer_multiplier,
                                 model.graph.final_norm.epsilon);
        }
        normed_ready = true;
    }

    if (!normed_ready) {
        encode_rmsnorm_batch(encoder, batch_hidden, final_norm, batch_normed,
                             rows, hidden_width, program.final_norm.epsilon);
    }
    const NSUInteger last_row_offset = static_cast<NSUInteger>(rows - 1) *
        hidden_width * sizeof(float);
    set_buffer(encoder, batch_hidden, 0, last_row_offset);
    set_buffer(encoder, hidden, 1);
    set_bytes(encoder, &hidden_width, sizeof(hidden_width), 2);
    dispatch(encoder, "celeg_copy", hidden_width);
    set_buffer(encoder, batch_normed, 0, last_row_offset);
    set_buffer(encoder, normed, 1);
    set_bytes(encoder, &hidden_width, sizeof(hidden_width), 2);
    dispatch(encoder, "celeg_copy", hidden_width);
    encode_matvec(encoder, embedding, normed, logits);
    position += static_cast<int>(rows);
    if (rope_positions.empty()) {
        for (int32_t& value : next_rope_position) value += static_cast<int32_t>(rows);
    }
}

}
