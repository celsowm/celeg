#include "celeg/detail/models/lfm2_layer_decoder.hpp"

#include <stdexcept>

namespace celeg::detail {

Lfm2LayerDecoderResult decode_lfm2_layer_types(
    const CheckpointMetadata& metadata, std::string_view gguf_prefix,
    int initial_key_value_heads) {
    Lfm2LayerDecoderResult result;
    result.num_key_value_heads = initial_key_value_heads;
    if (metadata.is_gguf()) {
        const auto values = std::get<std::vector<int64_t>>(
            metadata.value(std::string(gguf_prefix) + ".attention.head_count_kv"));
        result.mixer_kinds.reserve(values.size());
        for (const int64_t value : values) {
            if (value == 0) {
                result.mixer_kinds.push_back(MixerKind::ShortConvolution);
            } else {
                result.mixer_kinds.push_back(MixerKind::Attention);
                if (result.num_key_value_heads == 0) {
                    result.num_key_value_heads = static_cast<int>(value);
                } else if (result.num_key_value_heads != value) {
                    throw std::runtime_error("heterogeneous KV heads");
                }
            }
        }
    } else {
        for (const std::string& layer : metadata.strings("layer_types")) {
            if (layer == "conv") {
                result.mixer_kinds.push_back(MixerKind::ShortConvolution);
            } else if (layer == "full_attention") {
                result.mixer_kinds.push_back(MixerKind::Attention);
            } else {
                throw std::runtime_error("unsupported LFM2 layer type: " + layer);
            }
        }
    }
    return result;
}

} // namespace celeg::detail
