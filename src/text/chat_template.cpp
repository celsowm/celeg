#include "celeg/text/chat_template.hpp"

#include "celeg/models/gemma4/chat_template.hpp"
#include "celeg/models/granite/chat_template.hpp"
#include "celeg/models/lfm2/chat_template.hpp"
#include "celeg/models/minicpm5/chat_template.hpp"
#include "celeg/models/qwen35/chat_template.hpp"
#include "celeg/models/nemotron_h/chat_template.hpp"
#include "celeg/models/smollm3/chat_template.hpp"

#include <stdexcept>

namespace celeg {

std::string render_chat(std::span<const ChatMessage> messages,
                        const IChatTemplate& chat_template,
                        bool add_generation_prompt) {
    return chat_template.format(messages, add_generation_prompt);
}

std::string render_chat(std::span<const ChatMessage> messages,
                        std::span<const ChatToolDefinition> tools,
                        const IChatTemplate& chat_template,
                        bool add_generation_prompt) {
    return chat_template.format(messages, tools, add_generation_prompt);
}

std::string render_chat(std::span<const ChatMessage> messages,
                        std::span<const ChatToolDefinition> tools,
                        const IChatTemplate& chat_template,
                        bool add_generation_prompt,
                        const ChatTemplateOptions& options) {
    return chat_template.format(messages, tools, add_generation_prompt, options);
}

void ChatProfileCatalog::add(std::string profile_id,
                             std::unique_ptr<IChatTemplate> chat_template,
                             std::unique_ptr<IChatToolCallCodec> tool_call_codec,
                             ChatCapabilities capabilities) {
    if (frozen_) throw std::logic_error("chat profile catalog is frozen");
    if (profile_id.empty() || !chat_template) {
        throw std::invalid_argument("chat profile requires an id and template");
    }
    if (capabilities.native_tool_call_codec && !tool_call_codec) {
        throw std::invalid_argument("chat profile declares a native tool codec but provides none");
    }
    if (capabilities.parallel_tool_calls &&
        (!tool_call_codec || !tool_call_codec->supports_parallel_calls())) {
        throw std::invalid_argument("chat profile declares parallel tool calls without codec support");
    }
    if (tool_call_codec && !capabilities.native_tool_call_codec) {
        throw std::invalid_argument("chat profile provides a tool codec without enabling its capability");
    }
    std::shared_ptr<const IChatToolCallCodec> shared_codec(std::move(tool_call_codec));
    Entry entry{std::move(chat_template), std::move(shared_codec), capabilities};
    if (!entries_.emplace(std::move(profile_id), std::move(entry)).second) {
        throw std::invalid_argument("duplicate chat profile");
    }
}

void ChatProfileCatalog::freeze() { frozen_ = true; }

const IChatTemplate& ChatProfileCatalog::find(std::string_view profile_id) const {
    const auto it = entries_.find(std::string(profile_id));
    if (it == entries_.end()) throw std::invalid_argument("unknown chat profile: " + std::string(profile_id));
    return *it->second.chat_template;
}

ChatCapabilities ChatProfileCatalog::capabilities(std::string_view profile_id) const {
    const auto it = entries_.find(std::string(profile_id));
    if (it == entries_.end()) throw std::invalid_argument("unknown chat profile: " + std::string(profile_id));
    return it->second.capabilities;
}

const IChatToolCallCodec* ChatProfileCatalog::tool_codec(
    std::string_view profile_id) const {
    const auto it = entries_.find(std::string(profile_id));
    if (it == entries_.end()) throw std::invalid_argument(
        "unknown chat profile: " + std::string(profile_id));
    return it->second.tool_call_codec.get();
}

ChatProfileCatalog make_chat_profile_catalog() {
    ChatProfileCatalog catalog;
    add_builtin_chat_profiles(catalog);
    catalog.freeze();
    return catalog;
}

void add_builtin_chat_profiles(ChatProfileCatalog& catalog) {
    add_lfm2_chat_profile(catalog);
    add_granite_chat_profile(catalog);
    add_gemma4_chat_profile(catalog);
    add_minicpm5_chat_profile(catalog);
    add_smollm3_chat_profile(catalog);
    add_qwen35_chat_profile(catalog);
    add_nemotron_h_chat_profile(catalog);
}

} // namespace celeg
