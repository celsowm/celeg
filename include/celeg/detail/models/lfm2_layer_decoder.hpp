#pragma once

#include "celeg/checkpoint/metadata.hpp"
#include "celeg/model/resolved.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace celeg::detail {

struct Lfm2LayerDecoderResult {
    std::vector<MixerKind> mixer_kinds;
    int num_key_value_heads = 0;
};

Lfm2LayerDecoderResult decode_lfm2_layer_types(
    const CheckpointMetadata& metadata, std::string_view gguf_prefix,
    int initial_key_value_heads);

} // namespace celeg::detail
