#include "celeg/text/chat_template.hpp"
#include "celeg/text/tokenizer.hpp"
#include "support/assertions.hpp"

#include <cstdint>
#include <ctime>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

class EvidenceTokenizer final : public celeg::ITokenizer {
public:
    EvidenceTokenizer() : ids_{{"<|startoftext|>", 1}, {"<|im_start|>", 2},
                               {"<|im_end|>", 3}, {"assistant", 4},
                               {"<|tool_call_start|>", 5}, {"<|tool_call_end|>", 6}} {}
    std::vector<std::int32_t> encode(std::string_view, bool) const override { return {}; }
    std::string decode(const std::vector<std::int32_t>& tokens, bool) const override {
        return tokens.size() == 1 && tokens[0] == 1 ? "<|startoftext|>" : "";
    }
    std::string decode_token(std::int32_t, bool) const override { return {}; }
    std::optional<std::int32_t> token_id(std::string_view text) const override {
        const auto found = ids_.find(std::string(text));
        return found == ids_.end() ? std::nullopt : std::optional{found->second};
    }
    std::int32_t bos_id() const override { return 1; }
    std::int32_t eos_id() const override { return 3; }
    std::int32_t pad_id() const override { return 0; }
    int vocab_size() const override { return 7; }
private:
    std::unordered_map<std::string, std::int32_t> ids_;
};

}

int main() {
    const EvidenceTokenizer tokenizer;
    celeg::CheckpointMetadata metadata;
    metadata.values["chat_template"] = std::string(
        "{% macro turn(message) %}<|im_start|>{{ message.role }}\n{{ message.content }}"
        "{% for call in message.tool_calls %}<|tool_call_start|>[{{ call.function.name }}({{ call.function.arguments }})]<|tool_call_end|>{% endfor %}"
        "<|im_end|>\n{% endmacro %}{{ bos_token }}{% for message in messages %}{{ turn(message) }}{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}");
    const celeg::ResolvedInteraction interaction = celeg::resolve_interaction(metadata, tokenizer);
    const std::vector<celeg::ChatMessage> messages{{celeg::ChatRole::System, "You are concise."},
                                                    {celeg::ChatRole::User, "Hello"}};
    CELEG_TEST_CHECK(interaction.format(messages) ==
        "<|startoftext|><|im_start|>system\nYou are concise.<|im_end|>\n"
        "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n");
    CELEG_TEST_CHECK(interaction.tool_call_grammar().has_value());
    const celeg::ToolParseResult parsed = interaction.parse_tool_calls(
        "<|tool_call_start|>[weather({\"city\":\"Sao Paulo\"})]<|tool_call_end|>");
    CELEG_TEST_CHECK(parsed.status == celeg::ToolParseStatus::Complete);
    CELEG_TEST_CHECK(parsed.calls.size() == 1);
    CELEG_TEST_CHECK(parsed.calls[0].name == "weather");

    celeg::CheckpointMetadata no_source;
    const celeg::ResolvedInteraction inferred = celeg::resolve_interaction(no_source, tokenizer);
    CELEG_TEST_CHECK(inferred.source_origin() == "tokenizer-inference");
    CELEG_TEST_CHECK(!inferred.diagnostics().empty());
    const std::vector<celeg::ChatMessage> inferred_messages{{celeg::ChatRole::User, "Hello"}};
    CELEG_TEST_CHECK(inferred.format(inferred_messages).find("<|im_start|>assistant") != std::string::npos);

    // The {% generation %} block must render only when a generation prompt is
    // requested, mirroring add_generation_prompt semantics, with no model-specific
    // branching.
    celeg::CheckpointMetadata generation_template;
    generation_template.values["chat_template"] = std::string(
        "{{ bos_token }}{% for message in messages %}<|im_start|>{{ message.role }}\n"
        "{{ message.content }}<|im_end|>\n{% endfor %}"
        "{% generation %}<|im_start|>assistant\n{% endgeneration %}");
    const celeg::ResolvedInteraction generation_resolved =
        celeg::resolve_interaction(generation_template, tokenizer);
    CELEG_TEST_CHECK(generation_resolved.format(inferred_messages, /*add_generation_prompt=*/false)
                         .find("<|im_start|>assistant") == std::string::npos);
    CELEG_TEST_CHECK(generation_resolved.format(inferred_messages, /*add_generation_prompt=*/true)
                         .find("<|im_start|>assistant") != std::string::npos);

    // strftime_now must render the current local time via a C strftime format
    // rather than emitting the literal format string.
    celeg::CheckpointMetadata strftime_metadata;
    strftime_metadata.values["chat_template"] =
        std::string("Date: {{ strftime_now(\"%d %B %Y\") }}");
    const celeg::ResolvedInteraction strftime_resolved =
        celeg::resolve_interaction(strftime_metadata, tokenizer);
    const std::string strftime_output = strftime_resolved.format({});
    CELEG_TEST_CHECK(strftime_output.find("Date: ") == 0);
    CELEG_TEST_CHECK(strftime_output.find("%d %B %Y") == std::string::npos);
    const std::time_t now = std::time(nullptr);
    std::tm now_tm{};
#ifdef _WIN32
    localtime_s(&now_tm, &now);
#else
    localtime_r(&now, &now_tm);
#endif
    char year[8]{};
    std::strftime(year, sizeof(year), "%Y", &now_tm);
    CELEG_TEST_CHECK(strftime_output.find(year) != std::string::npos);

    std::cout << "chat_template_test: ok\n";
}
