#include "celeg/text/chat_template.hpp"
#include "celeg/text/tokenizer.hpp"
#include "support/assertions.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

class EvidenceTokenizer final : public celeg::ITokenizer {
public:
    EvidenceTokenizer() {
        ids_ = {{"<|startoftext|>", 1}, {"<|im_start|>", 2},
                {"<|im_end|>", 3}, {"<|tool_call_start|>", 4},
                {"<|tool_call_end|>", 5}};
    }
    std::vector<std::int32_t> encode(std::string_view, bool) const override { return {}; }
    std::string decode(const std::vector<std::int32_t>&, bool) const override { return {}; }
    std::string decode_token(std::int32_t, bool) const override { return {}; }
    std::optional<std::int32_t> token_id(std::string_view text) const override {
        const auto it = ids_.find(std::string(text));
        return it == ids_.end() ? std::nullopt : std::optional{it->second};
    }
    std::int32_t bos_id() const override { return 1; }
    std::int32_t eos_id() const override { return 3; }
    std::int32_t pad_id() const override { return 0; }
private:
    std::unordered_map<std::string, std::int32_t> ids_;
};

} // namespace

int main() {
    const EvidenceTokenizer tokenizer;
    celeg::CheckpointMetadata empty;
    const auto inferred = celeg::resolve_chat_template(empty, tokenizer);
    CELEG_TEST_CHECK(inferred.source_origin() == "tokenizer-inference");
    CELEG_TEST_CHECK(inferred.format(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "Hello"}}, true) ==
        "<|startoftext|><|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n");

    celeg::CheckpointMetadata source;
    source.values["chat_template"] = std::string(
        "{% macro render(message) %}<|im_start|>{{ message.role }}\\n"
        "{{ message.content }}<|im_end|>{% endmacro %}"
        "{% if add_generation_prompt %}<|im_start|>assistant{% endif %}");
    const auto compiled = celeg::resolve_chat_template(source, tokenizer);
    CELEG_TEST_CHECK(compiled.source_origin() == "checkpoint-metadata");
    CELEG_TEST_CHECK(!compiled.fingerprint().empty());

    const std::filesystem::path override_file =
        std::filesystem::temp_directory_path() / "celeg-chat-template-resolution-test.jinja";
    {
        std::ofstream output(override_file, std::ios::binary);
        output << "{% if add_generation_prompt %}<|im_start|>assistant<|im_end|>{% endif %}";
    }
    source.values["chat_template"] = std::string("{% include 'unsafe.jinja' %}");
    const auto overridden = celeg::resolve_chat_template(source, tokenizer, override_file);
    CELEG_TEST_CHECK(overridden.source_origin().starts_with("override:"));
    std::filesystem::remove(override_file);

    bool rejected = false;
    try {
        (void)celeg::resolve_chat_template(source, tokenizer);
    } catch (const std::invalid_argument& error) {
        rejected = std::string(error.what()).find("UnsupportedChatTemplateConstruct") !=
            std::string::npos;
    }
    CELEG_TEST_CHECK(rejected);
    std::cout << "chat_template_resolution_test: ok\n";
}
