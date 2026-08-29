#include "detail.hpp"

#include <stdexcept>

namespace celeg {

void MetalModel::Impl::encode_token(id<MTLCommandBuffer>& command_buffer,
                                    id<MTLComputeCommandEncoder>& encoder, int32_t token) {
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

        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            Layer& layer = layers[layer_index];
            set_buffer(encoder, hidden, 0);
            set_buffer(encoder, residual, 1);
            set_bytes(encoder, &hidden_width, sizeof(hidden_width), 2);
            dispatch(encoder, "celeg_copy", hidden_width);
            encode_rmsnorm(encoder, hidden, layer.operator_norm, normed,
                           hidden_width, model.graph.final_norm.epsilon);

            if (layer.convolution) {
                encode_short_convolution(encoder, layer);
            } else if (layer.gated_delta) {
                encode_gated_delta_layer(encoder, layer);
            } else if (layer.mamba2) {
                encode_mamba2_layer(encoder, layer);
            } else {
                encode_attention(encoder, layer);
            }

            const uint32_t count = hidden_width;
            const float mixer_multiplier = 1.0f;
            set_buffer(encoder, hidden, 0);
            set_buffer(encoder, residual, 1);
            set_buffer(encoder, hidden, 2);
            set_bytes(encoder, &count, sizeof(count), 3);
            set_bytes(encoder, &mixer_multiplier, sizeof(mixer_multiplier), 4);
            dispatch(encoder, "celeg_residual", count);

            if (std::holds_alternative<std::monostate>(
                    program.layers[layer_index].feed_forward)) {
                continue;
            }

            encode_rmsnorm(encoder, hidden, layer.ffn_norm, normed, hidden_width,
                           program.final_norm.epsilon);
            if (layer.moe) {
                encode_moe(command_buffer, encoder, layer);
            } else {
                encode_dense_feed_forward(encoder, layer);
            }
            set_buffer(encoder, layer.moe ? moe_output : operation, 0);
            set_buffer(encoder, hidden, 1);
            set_buffer(encoder, hidden, 2);
            set_bytes(encoder, &count, sizeof(count), 3);
            set_bytes(encoder, &mixer_multiplier, sizeof(mixer_multiplier), 4);
            dispatch(encoder, "celeg_residual", count);
        }

        encode_rmsnorm(encoder, hidden, final_norm, normed, hidden_width,
                       program.final_norm.epsilon);
        encode_matvec(encoder, embedding, normed, logits);
        ++position;
}

void MetalModel::Impl::run_token(int32_t token) {
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    begin_commands(command_buffer, encoder);
    encode_token(command_buffer, encoder, token);
    finish_commands(command_buffer, encoder);
}

}
