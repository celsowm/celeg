#include "celeg/text/chat_template.hpp"
#include "celeg/text/tokenizer.hpp"
#include "support/assertions.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

class Tokenizer final : public celeg::ITokenizer {
public:
    std::vector<std::int32_t> encode(std::string_view, bool) const override { return {}; }
    std::string decode(const std::vector<std::int32_t>&, bool) const override { return {}; }
    std::string decode_token(std::int32_t, bool) const override { return {}; }
    std::optional<std::int32_t> token_id(std::string_view text) const override {
        if (text == "<|startoftext|>" || text == "<|im_start|>" || text == "<|im_end|>" || text == "assistant") return 1;
        return std::nullopt;
    }
    std::int32_t bos_id() const override { return 1; }
    std::int32_t eos_id() const override { return 2; }
    std::int32_t pad_id() const override { return 0; }
};

} // namespace

int main() {
    const Tokenizer tokenizer;
    celeg::CheckpointMetadata metadata;
    metadata.values["chat_template"] = std::string("{% include 'unsafe.jinja' %}");
    const std::filesystem::path override_file = std::filesystem::temp_directory_path() / "celeg-interaction-override.jinja";
    {
        std::ofstream output(override_file, std::ios::binary);
        output << "{% for message in messages %}[{{ message.role }}]{{ message.content }}{% endfor %}";
    }
    const celeg::ResolvedInteraction override = celeg::resolve_interaction(metadata, tokenizer, override_file);
    CELEG_TEST_CHECK(override.source_origin().starts_with("override:"));
    const std::vector<celeg::ChatMessage> messages{{celeg::ChatRole::User, "hello"}};
    CELEG_TEST_CHECK(override.format(messages, false) == "[user]hello");
    std::filesystem::remove(override_file);

    bool rejected = false;
    try { (void)celeg::resolve_interaction(metadata, tokenizer); }
    catch (const std::invalid_argument& error) {
        rejected = std::string(error.what()).find("UnsupportedChatTemplateConstruct") != std::string::npos;
    }
    CELEG_TEST_CHECK(rejected);

    // Validation covers every branch.  An unsupported construct must not be
    // accepted merely because this render would leave the branch inactive.
    metadata.values["chat_template"] = std::string(
        "{% if false %}{{ messages[0].content|unknown_filter }}{% endif %}");
    rejected = false;
    try { (void)celeg::resolve_interaction(metadata, tokenizer); }
    catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        rejected = message.find("UnsupportedChatTemplateConstruct") != std::string::npos &&
            message.find("checkpoint-metadata:1:") != std::string::npos;
    }
    CELEG_TEST_CHECK(rejected);
    std::cout << "chat_template_resolution_test: ok\n";
}
