#include "celeg/serve/protocol/mapping.hpp"

#include "celeg/checkpoint/formats/json.hpp"

#include <stdexcept>
#include <unordered_set>

namespace {

struct NormalizedContent {
    std::string text;
    std::vector<celeg::serve::MultimodalImage> images;
};

NormalizedContent normalize_content(
    const std::optional<celeg::serve::protocol::ChatContentDto>& content) {
    if (!content) return {};
    if (std::holds_alternative<std::string>(*content)) {
        return {std::get<std::string>(*content), {}};
    }

    NormalizedContent result;
    for (const auto& part : std::get<std::vector<celeg::serve::protocol::ChatContentPartDto>>(*content)) {
        if (part.type == "text") {
            if (!part.text) throw std::invalid_argument("text content parts require text");
            result.text += *part.text;
        } else if (part.type == "image_url") {
            if (!part.image_url || part.image_url->url.empty()) {
                throw std::invalid_argument("image_url content parts require image_url.url");
            }
            result.text += "<|image|>";
            result.images.push_back({part.image_url->url});
        } else {
            throw std::invalid_argument("unsupported content part type: " + part.type);
        }
    }
    return result;
}

} // namespace

namespace celeg::serve::protocol {

ChatRole role_from_string(const std::string& role) {
    if (role == "system") return ChatRole::System;
    if (role == "developer") return ChatRole::Developer;
    if (role == "user") return ChatRole::User;
    if (role == "assistant") return ChatRole::Assistant;
    if (role == "tool") return ChatRole::Tool;
    throw std::invalid_argument("unknown chat role: " + role);
}

std::string role_to_string(ChatRole role) {
    switch (role) {
        case ChatRole::System: return "system";
        case ChatRole::Developer: return "developer";
        case ChatRole::User: return "user";
        case ChatRole::Assistant: return "assistant";
        case ChatRole::Tool: return "tool";
    }
    throw std::invalid_argument("unknown chat role");
}

void validate_chat_request(const ChatCompletionRequest& request,
                           const ChatCapabilities& capabilities) {
    if (request.messages.empty()) throw std::invalid_argument("messages must not be empty");
    std::unordered_set<std::string> tool_names;
    if (request.tools) {
        for (const ToolDto& tool : *request.tools) {
            if (tool.type != "function") throw std::invalid_argument("tools only support type=function");
            if (tool.function.name.empty()) throw std::invalid_argument("function name must not be empty");
            if (!tool_names.insert(tool.function.name).second) {
                throw std::invalid_argument("duplicate function name: " + tool.function.name);
            }
            if (tool.function.parameters) {
                try { (void)Json::parse(tool.function.parameters->str); }
                catch (const std::exception&) {
                    throw std::invalid_argument("function parameters must be valid JSON");
                }
            }
        }
    }
    if (request.tool_choice && std::holds_alternative<std::string>(*request.tool_choice)) {
        const std::string& choice = std::get<std::string>(*request.tool_choice);
        if (choice != "none" && choice != "auto" && choice != "required" &&
            !choice.starts_with("function:")) {
            throw std::invalid_argument("tool_choice must be none, auto, required, or function:<name>");
        }
        if (choice == "required" && tool_names.empty()) throw std::invalid_argument("tool_choice=required requires tools");
        if (choice.starts_with("function:") && !tool_names.contains(choice.substr(9))) {
            throw std::invalid_argument("tool_choice references an undeclared function");
        }
    }
    if (request.parallel_tool_calls.value_or(false) && !capabilities.parallel_tool_calls) {
        throw std::invalid_argument("parallel_tool_calls is not supported by this chat profile");
    }
    if (request.tools && !capabilities.assistant_tool_calls) {
        throw std::invalid_argument("tools are not supported by this chat profile");
    }
    if (request.tool_choice && std::holds_alternative<ToolChoiceDto>(*request.tool_choice) &&
        !tool_names.contains(std::get<ToolChoiceDto>(*request.tool_choice).function.name)) {
        throw std::invalid_argument("tool_choice references an undeclared function");
    }
    std::unordered_set<std::string> pending_tool_calls;
    for (const ChatMessageDto& message : request.messages) {
        const NormalizedContent normalized = normalize_content(message.content);
        if (!normalized.images.empty() && !capabilities.vision) {
            throw std::invalid_argument("image content is not supported by this model");
        }
        const ChatRole role = role_from_string(message.role);
        const bool role_supported =
            (role == ChatRole::System && capabilities.roles.system) ||
            (role == ChatRole::Developer && capabilities.roles.developer) ||
            (role == ChatRole::User && capabilities.roles.user) ||
            (role == ChatRole::Assistant && capabilities.roles.assistant) ||
            (role == ChatRole::Tool && capabilities.roles.tool);
        if (!role_supported) {
            throw std::invalid_argument("chat role is not supported by this chat profile: " +
                                        message.role);
        }
        if (role == ChatRole::Tool) {
            if (!message.tool_call_id || message.tool_call_id->empty()) throw std::invalid_argument("tool messages require tool_call_id");
            if (!message.content) throw std::invalid_argument("tool messages require content");
        }
        if (role == ChatRole::Assistant && message.tool_calls) {
            if (!capabilities.assistant_tool_calls) throw std::invalid_argument("assistant tool calls are not supported by this chat profile");
            std::unordered_set<std::string> ids;
            for (const ToolCallDto& call : *message.tool_calls) {
                if (call.id.empty() || !ids.insert(call.id).second) throw std::invalid_argument("assistant tool call IDs must be unique");
                if (call.type != "function") throw std::invalid_argument("tool calls only support type=function");
                if (call.function.name.empty()) throw std::invalid_argument("tool call function name must not be empty");
                try { (void)Json::parse(call.function.arguments); }
                catch (const std::exception&) { throw std::invalid_argument("tool call arguments must be valid JSON"); }
                pending_tool_calls.insert(call.id);
            }
        }
        if (role == ChatRole::Assistant && (!message.content &&
            (!message.tool_calls || message.tool_calls->empty()))) {
            throw std::invalid_argument("assistant messages require content or tool_calls");
        }
        if (role == ChatRole::Tool) {
            if (!pending_tool_calls.contains(*message.tool_call_id)) {
                throw std::invalid_argument("tool message references an unknown tool call id");
            }
            pending_tool_calls.erase(*message.tool_call_id);
        }
        if (role != ChatRole::Assistant && !message.content) {
            throw std::invalid_argument("message content is required for this role");
        }
    }
}

ErrorResponseDto error_response(std::string message, std::optional<std::string> param) {
    ErrorResponseDto response;
    response.error.message = std::move(message);
    response.error.param = std::move(param);
    return response;
}

GenerateRequest to_generate_request(const ChatCompletionRequest& request,
                                    const celeg::BpeTokenizer& tokenizer,
                                    const celeg::IChatTemplate& chat_template,
                                    const celeg::ChatCapabilities& capabilities,
                                    std::span<const std::int32_t> eos_token_ids,
                                    const celeg::ChatTemplateOptions& template_options) {
    validate_chat_request(request, capabilities);

    std::vector<celeg::ChatMessage> messages;
    std::vector<celeg::ChatToolDefinition> tools;
    messages.reserve(request.messages.size());
    if (request.tools) {
        tools.reserve(request.tools->size());
        for (const ToolDto& tool : *request.tools) {
            tools.push_back({"function", {
                tool.function.name,
                tool.function.description.value_or(""),
                {tool.function.parameters ? tool.function.parameters->str : "{}"},
                tool.function.strict.value_or(false)}});
        }
    }
    std::vector<celeg::serve::MultimodalImage> images;
    for (const ChatMessageDto& message : request.messages) {
        const NormalizedContent normalized = normalize_content(message.content);
        images.insert(images.end(), normalized.images.begin(), normalized.images.end());
        celeg::ChatMessage mapped{role_from_string(message.role), normalized.text,
                                  {}, message.tool_call_id, std::nullopt};
        if (message.tool_calls) {
            for (const ToolCallDto& call : *message.tool_calls) {
                mapped.tool_calls.push_back({call.id, call.function.name, call.function.arguments});
            }
        }
        messages.push_back(std::move(mapped));
    }
    celeg::validate_conversation(messages);

    celeg::ChatTemplateOptions effective_template_options = template_options;
    if (request.chat_template_kwargs) {
        effective_template_options.enable_thinking =
            request.chat_template_kwargs->enable_thinking;
    }
    const std::string prompt_text = celeg::render_chat(
        messages, tools, chat_template, /*add_generation_prompt=*/true,
        effective_template_options);

    GenerateRequest generate_request;
    generate_request.rendered_prompt = prompt_text;
    generate_request.images = std::move(images);
    if (!generate_request.images.empty()) {
        const auto image_token = tokenizer.token_id("<|image|>");
        if (!image_token) {
            throw std::invalid_argument("Gemma image marker is absent from the tokenizer vocabulary");
        }
        generate_request.image_token_id = *image_token;
    }
    generate_request.prompt_tokens = tokenizer.encode(prompt_text, /*add_bos=*/false);
    generate_request.eos_token_ids.assign(eos_token_ids.begin(), eos_token_ids.end());
    if (request.max_tokens && *request.max_tokens <= 0) {
        throw std::invalid_argument("max_tokens must be positive");
    }
    generate_request.max_output_tokens =
        request.max_tokens ? static_cast<std::size_t>(*request.max_tokens) : 128;
    if (request.temperature) generate_request.generation.temperature = static_cast<float>(*request.temperature);
    if (request.top_p) generate_request.generation.top_p = static_cast<float>(*request.top_p);
    if (request.top_k) generate_request.generation.top_k = *request.top_k;
    if (request.seed) generate_request.generation.seed = *request.seed;
    return generate_request;
}

std::string finish_reason_to_string(FinishReason reason) {
    switch (reason) {
        case FinishReason::None: return "";
        case FinishReason::Stop: return "stop";
        case FinishReason::Length: return "length";
        case FinishReason::Cancelled: return "cancelled";
        case FinishReason::Error: return "error";
        case FinishReason::ToolCalls: return "tool_calls";
    }
    return "";
}

ChatCompletionResponse to_chat_completion_response(const std::string& id,
                                                   const std::string& model,
                                                   std::int64_t created,
                                                   std::size_t prompt_token_count,
                                                   const std::vector<std::int32_t>& completion_tokens,
                                                   FinishReason reason,
                                                   const celeg::BpeTokenizer& tokenizer,
                                                   const celeg::ChatCapabilities& capabilities) {
    ChatCompletionResponse response;
    response.id = id;
    response.model = model;
    response.created = created;

    ChatCompletionChoice choice;
    choice.index = 0;
    const std::string text = tokenizer.decode(completion_tokens);
    if (capabilities.tool_call_codec) {
        const auto parsed = capabilities.tool_call_codec->parse_generation(text);
        if (!parsed.calls.empty()) {
            choice.message.content = parsed.assistant_text;
            choice.message.tool_calls = std::vector<ToolCallDto>{};
            for (const auto& call : parsed.calls) {
                choice.message.tool_calls->push_back({call.id, "function", {call.name, call.arguments}});
            }
            choice.finish_reason = finish_reason_to_string(FinishReason::ToolCalls);
        } else if (parsed.status == celeg::ToolParseStatus::Invalid ||
                   parsed.status == celeg::ToolParseStatus::Incomplete) {
            choice.message.content = parsed.assistant_text.empty() ? text : parsed.assistant_text;
            choice.finish_reason = finish_reason_to_string(FinishReason::Error);
        } else {
            choice.message.content = text;
            choice.finish_reason = finish_reason_to_string(reason);
        }
    } else {
        choice.message.content = text;
        choice.finish_reason = finish_reason_to_string(reason);
    }
    response.choices.push_back(std::move(choice));

    response.usage.prompt_tokens = prompt_token_count;
    response.usage.completion_tokens = completion_tokens.size();
    response.usage.total_tokens = prompt_token_count + completion_tokens.size();
    return response;
}

ChatCompletionChunk to_chat_completion_chunk(const std::string& id,
                                             const std::string& model,
                                             std::int64_t created,
                                             const std::vector<std::int32_t>& new_tokens,
                                             bool include_role,
                                             std::optional<FinishReason> finish,
                                             const celeg::BpeTokenizer& tokenizer,
                                             const celeg::ChatCapabilities& capabilities,
                                             std::string_view accumulated_text) {
    ChatCompletionChunk chunk;
    chunk.id = id;
    chunk.model = model;
    chunk.created = created;

    ChatCompletionChunkChoice choice;
    choice.index = 0;
    if (include_role) choice.delta.role = "assistant";
    if (!new_tokens.empty()) {
        const std::string text = tokenizer.decode(new_tokens);
        if (capabilities.tool_call_codec) {
            const auto parsed = capabilities.tool_call_codec->parse_generation(text);
            if (!parsed.calls.empty()) {
                choice.delta.tool_calls = std::vector<ToolCallDto>{};
                for (const auto& call : parsed.calls) {
                    choice.delta.tool_calls->push_back({call.id, "function", {call.name, call.arguments}});
                }
            } else {
                choice.delta.content = text;
            }
        } else {
            choice.delta.content = text;
        }
    }
    if (new_tokens.empty() && finish && capabilities.tool_call_codec && !accumulated_text.empty()) {
        const auto parsed = capabilities.tool_call_codec->parse_generation(accumulated_text);
        if (!parsed.calls.empty()) {
            choice.delta.content.reset();
            choice.delta.tool_calls = std::vector<ToolCallDto>{};
            for (const auto& call : parsed.calls) {
                choice.delta.tool_calls->push_back({call.id, "function", {call.name, call.arguments}});
            }
        }
    }
    if (finish) choice.finish_reason = finish_reason_to_string(*finish);
    chunk.choices.push_back(std::move(choice));
    return chunk;
}

ChatCompletionChunk to_chat_completion_chunk(
    const std::string& id, const std::string& model, std::int64_t created,
    const celeg::serve::ChatGenerationDelta& delta, bool include_role,
    std::optional<FinishReason> finish) {
    ChatCompletionChunk chunk;
    chunk.id = id;
    chunk.model = model;
    ChatCompletionChunkChoice choice;
    choice.index = 0;
    if (include_role) choice.delta.role = "assistant";
    if (!delta.text.empty()) choice.delta.content = delta.text;
    if (!delta.tool_calls.empty()) {
        choice.delta.tool_calls = std::vector<ToolCallDto>{};
        for (const auto& call : delta.tool_calls) {
            choice.delta.tool_calls->push_back({call.id, "function", {call.name, call.arguments}});
        }
    }
    if (finish) choice.finish_reason = finish_reason_to_string(*finish);
    chunk.choices.push_back(std::move(choice));
    return chunk;
}

TokenizeResponse to_tokenize_response(const TokenizeRequest& request,
                                      const celeg::BpeTokenizer& tokenizer,
                                      const celeg::IChatTemplate& chat_template,
                                      std::size_t max_model_len) {
    if (static_cast<bool>(request.prompt) == static_cast<bool>(request.messages)) {
        throw std::invalid_argument("exactly one of prompt or messages must be set");
    }

    const bool add_special_tokens = request.add_special_tokens.value_or(true);

    TokenizeResponse response;
    if (request.prompt) {
        response.tokens = tokenizer.encode(*request.prompt, /*add_bos=*/add_special_tokens);
    } else {
        std::vector<celeg::ChatMessage> messages;
        messages.reserve(request.messages->size());
        for (const ChatMessageDto& message : *request.messages) {
            messages.push_back({role_from_string(message.role),
                                normalize_content(message.content).text});
        }
        const std::string prompt_text = celeg::render_chat(
            messages, chat_template, /*add_generation_prompt=*/true);
        // The chat template already embeds the model's special tokens, so BOS
        // is never added again here regardless of add_special_tokens (which
        // only applies to the raw-prompt path above).
        response.tokens = tokenizer.encode(prompt_text, /*add_bos=*/false);
    }
    response.count = response.tokens.size();
    response.max_model_len = max_model_len;
    return response;
}

} // namespace celeg::serve::protocol
