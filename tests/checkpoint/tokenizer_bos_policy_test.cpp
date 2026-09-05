// A raw prompt must prepend BOS only when the checkpoint says its tokenizer
// does. Assuming BOS silently corrupts checkpoints that set add_bos_token
// false and reuse one token as both BOS and EOS -- ibm-granite/granite-4.1-3b
// uses <|end_of_text|> for both, so an assumed BOS put an end-of-text marker in
// position 0 and generation degenerated into a single repeated token.
#include "celeg/app/run_preparation.hpp"
#include "celeg/checkpoint/catalog.hpp"
#include "checkpoint/detail/binary_codec.hpp"
#include "support/assertions.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace {

// Writes the smallest checkpoint the safetensors format will open: a config.json
// plus a single empty-header shard.
std::filesystem::path write_fixture(std::string_view name,
                                    std::string_view tokenizer_config) {
    const auto dir = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    std::ofstream(dir / "config.json") << R"({"model_type":"fixture"})";
    if (!tokenizer_config.empty()) {
        std::ofstream(dir / "tokenizer_config.json") << tokenizer_config;
    }
    const std::string header = "{}";
    std::ofstream shard(dir / "model.safetensors", std::ios::binary);
    celeg::binary::write_le(shard, static_cast<uint64_t>(header.size()));
    shard.write(header.data(), static_cast<std::streamsize>(header.size()));
    return dir;
}

celeg::CheckpointMetadata metadata_for(std::string_view name,
                                       std::string_view tokenizer_config) {
    celeg::CheckpointFormatCatalog catalog;
    celeg::add_builtin_checkpoint_formats(catalog);
    catalog.freeze();
    return catalog.open(write_fixture(name, tokenizer_config)).metadata;
}

}

int main() {
    using celeg::app::raw_prompt_takes_bos;

    // A checkpoint that opts out must be honoured.
    const auto declines = metadata_for("celeg_bos_false",
                                       R"({"add_bos_token":false})");
    CELEG_TEST_CHECK(declines.contains("tokenizer.add_bos_token"));
    CELEG_TEST_CHECK(!raw_prompt_takes_bos(declines));

    // An explicit true is honoured too.
    const auto accepts = metadata_for("celeg_bos_true",
                                      R"({"add_bos_token":true})");
    CELEG_TEST_CHECK(raw_prompt_takes_bos(accepts));

    // add_bos_token lives beside chat_template, and used to be readable only
    // when no chat template was present. Parsing must not depend on that.
    const auto with_template = metadata_for(
        "celeg_bos_with_template",
        R"({"add_bos_token":false,"chat_template":"{{ x }}"})");
    CELEG_TEST_CHECK(!raw_prompt_takes_bos(with_template));

    // Silence means BOS, matching Hugging Face's default for tokenizers that
    // omit the field.
    const auto silent = metadata_for("celeg_bos_absent", R"({})");
    CELEG_TEST_CHECK(!silent.contains("tokenizer.add_bos_token"));
    CELEG_TEST_CHECK(raw_prompt_takes_bos(silent));

    const auto no_config = metadata_for("celeg_bos_no_config", "");
    CELEG_TEST_CHECK(raw_prompt_takes_bos(no_config));

    // GGUF spells the same policy differently and must be read there too.
    celeg::CheckpointMetadata gguf;
    gguf.source_format = celeg::CheckpointSourceFormat::Gguf;
    gguf.values["tokenizer.ggml.add_bos_token"] = false;
    CELEG_TEST_CHECK(!raw_prompt_takes_bos(gguf));
    gguf.values["tokenizer.ggml.add_bos_token"] = true;
    CELEG_TEST_CHECK(raw_prompt_takes_bos(gguf));
    return 0;
}
