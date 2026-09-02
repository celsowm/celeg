#include "detail.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace celeg {

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
            set_buffer(encoder, operation, 0);
            set_buffer(encoder, hidden, 1);
            set_bytes(encoder, &hidden_width, sizeof(hidden_width), 2);
            dispatch(encoder, "celeg_copy", hidden_width);
        }

        bool normed_ready = false;
        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            Layer& layer = layers[layer_index];
            const CompiledLayerProgram& program_layer = program.layers[layer_index];
            if (!normed_ready) {
                encode_rmsnorm_save(encoder, hidden, residual, layer.operator_norm, normed,
                               hidden_width, model.graph.final_norm.epsilon);
            }
            normed_ready = false;

            switch (layer.mixer_kind) {
                case Layer::MixerKind::ShortConvolution:
                    encode_short_convolution(encoder, layer);
                    break;
                case Layer::MixerKind::GatedDelta:
                    encode_gated_delta_layer(encoder, layer);
                    break;
                case Layer::MixerKind::Mamba2:
                    encode_mamba2_layer(encoder, layer);
                    break;
                case Layer::MixerKind::Attention:
                    encode_attention(
                        encoder, layer,
                        std::get<CompiledAttentionProgram>(program_layer.mixer),
                        rope_position);
                    break;
            }

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
        if (std::holds_alternative<GatedDeltaNetSpec>(layer.mixer) ||
            std::holds_alternative<Mamba2Spec>(layer.mixer) ||
            std::holds_alternative<MlpBlockSpec>(layer.mixer)) {
            return false;
        }
        if (const auto* attention = std::get_if<CompiledAttentionProgram>(&layer.mixer)) {
            const bool supported_pattern =
                std::holds_alternative<FullCausalPattern>(attention->semantics.pattern) ||
                std::holds_alternative<SlidingWindowPattern>(attention->semantics.pattern);
            const bool supported_bias =
                std::holds_alternative<NoAttentionBiasSpec>(attention->semantics.bias) ||
                std::holds_alternative<AlibiBiasSpec>(attention->semantics.bias) ||
                std::holds_alternative<RelativePositionBiasSpec>(attention->semantics.bias);
            const bool supported_output_transform =
                std::holds_alternative<NoAttentionOutputTransformSpec>(
                    attention->semantics.output_transform) ||
                std::holds_alternative<OrthogonalizeCurrentValueSpec>(
                    attention->semantics.output_transform);
            if (!std::holds_alternative<OrdinaryKvStateSpec>(attention->semantics.state) ||
                !supported_pattern || !supported_bias || !supported_output_transform) {
                return false;
            }
        }
        if (std::holds_alternative<MoeLayerProgram>(layer.feed_forward)) return false;
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

    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        Layer& layer = layers[layer_index];
        const CompiledLayerProgram& program_layer = program.layers[layer_index];
        encode_rmsnorm_batch(encoder, batch_hidden, layer.operator_norm, batch_normed,
                             rows, hidden_width, model.graph.final_norm.epsilon);
        // Preserve the current hidden state as the residual without copying it.
        // Every supported batched mixer writes a complete replacement into
        // batch_hidden, so swapping the two scratch handles makes the old
        // hidden buffer the residual and gives the mixer a disjoint output.
        std::swap(batch_hidden, batch_residual);
        switch (layer.mixer_kind) {
            case Layer::MixerKind::ShortConvolution:
                encode_matmul(encoder, layer.mixer_in, batch_normed, batch_projected, rows);
                encode_short_convolution_batch(encoder, layer, rows, base_position);
                break;
            case Layer::MixerKind::Attention:
                encode_attention_batch(
                    encoder, layer,
                    std::get<CompiledAttentionProgram>(program_layer.mixer),
                    rows, base_position);
                break;
            case Layer::MixerKind::GatedDelta:
            case Layer::MixerKind::Mamba2:
                throw std::logic_error("recurrent Metal mixer reached batched prefill");
        }
        const float mixer_multiplier = 1.0f;
        encode_residual_batch(encoder, batch_hidden, batch_residual, batch_hidden,
                              count, mixer_multiplier);
        if (std::holds_alternative<std::monostate>(program_layer.feed_forward)) {
            continue;
        }
        encode_rmsnorm_batch(encoder, batch_hidden, layer.ffn_norm, batch_normed,
                             rows, hidden_width, program.final_norm.epsilon);
        encode_dense_feed_forward_batch(encoder, layer, rows);
        encode_residual_batch(encoder, batch_operation, batch_hidden, batch_hidden,
                              count, mixer_multiplier);
    }
    encode_rmsnorm_batch(encoder, batch_hidden, final_norm, batch_normed,
                         rows, hidden_width, program.final_norm.epsilon);
    set_buffer(encoder, batch_hidden, 0, static_cast<NSUInteger>(rows - 1) *
        hidden_width * sizeof(float));
    set_buffer(encoder, hidden, 1);
    set_bytes(encoder, &hidden_width, sizeof(hidden_width), 2);
    dispatch(encoder, "celeg_copy", hidden_width);
    encode_matmul(encoder, embedding, batch_normed, logits, 1,
                  static_cast<NSUInteger>(rows - 1) * hidden_width * sizeof(float));
    position += static_cast<int>(rows);
    if (rope_positions.empty()) {
        for (int32_t& value : next_rope_position) value += static_cast<int32_t>(rows);
    }
}

}
