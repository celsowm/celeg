#include "celeg/checkpoint/metadata.hpp"
#include "celeg/detail/models/lfm2_layer_decoder.hpp"
#include "support/assertions.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    celeg::CheckpointMetadata safetensors;
    safetensors.values["layer_types"] = std::vector<std::string>{
        "conv", "full_attention", "conv"};
    const auto decoded = celeg::detail::decode_lfm2_layer_types(
        safetensors, "blk", 8);
    CELEG_TEST_CHECK(decoded.mixer_kinds.size() == 3);
    CELEG_TEST_CHECK(decoded.mixer_kinds[0] == celeg::MixerKind::ShortConvolution);
    CELEG_TEST_CHECK(decoded.mixer_kinds[1] == celeg::MixerKind::Attention);
    CELEG_TEST_CHECK(decoded.num_key_value_heads == 8);

    celeg::CheckpointMetadata gguf;
    gguf.source_format = celeg::CheckpointSourceFormat::Gguf;
    gguf.values["blk.attention.head_count_kv"] =
        std::vector<int64_t>{0, 4, 0, 4};
    const auto gguf_decoded = celeg::detail::decode_lfm2_layer_types(
        gguf, "blk", 0);
    CELEG_TEST_CHECK(gguf_decoded.mixer_kinds.size() == 4);
    CELEG_TEST_CHECK(gguf_decoded.num_key_value_heads == 4);

    celeg::CheckpointMetadata heterogeneous;
    heterogeneous.source_format = celeg::CheckpointSourceFormat::Gguf;
    heterogeneous.values["blk.attention.head_count_kv"] =
        std::vector<int64_t>{4, 8};
    bool rejected = false;
    try {
        (void)celeg::detail::decode_lfm2_layer_types(heterogeneous, "blk", 0);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    CELEG_TEST_CHECK(rejected);

    return 0;
}
