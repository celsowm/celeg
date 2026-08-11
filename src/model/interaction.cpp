#include "celeg/model/interaction.hpp"

#include <stdexcept>

namespace celeg {

std::string resolve_chat_template_id(const CheckpointMetadata& metadata) {
    const std::string source = metadata.string_or(
        "chat_template", metadata.string_or("tokenizer.chat_template", {}));
    if (source.empty()) return {};

    // These markers are interaction evidence from the template language. The
    // selected IDs are reusable protocols registered by the runtime catalog.
    if (source.find("<|im_start|>") != std::string::npos &&
        source.find("<|im_end|>") != std::string::npos &&
        source.find("tools") != std::string::npos &&
        (source.find("function") != std::string::npos ||
         source.find("tool_call") != std::string::npos)) {
        return "chat:thinking-function";
    }
    if (source.find("<|start_header_id|>") != std::string::npos) {
        return "chat:role-envelope";
    }
    if (source.find("<|assistant|>") != std::string::npos &&
        source.find("<|user|>") != std::string::npos) {
        return "chat:turn";
    }
    if (source.find("<role>") != std::string::npos &&
        source.find("<|role_end|>") != std::string::npos &&
        source.find("<arg_key>") != std::string::npos &&
        source.find("<arg_value>") != std::string::npos) {
        return "chat:tagged-role";
    }
    throw std::invalid_argument(
        "UnsupportedChatTemplateConstruct: no registered interaction protocol matches "
        "the checkpoint template evidence");
}

} // namespace celeg
