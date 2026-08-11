#pragma once

#include "celeg/text/semantic_chat_templates.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace celeg {

enum class ToolCallGrammar : std::uint8_t {
    None,
    DelimitedArguments,
    TaggedObject,
    FunctionXml,
    ParameterXml,
    TaggedJson,
};

struct ToolProtocolSpec {
    ToolCallGrammar grammar = ToolCallGrammar::None;
    bool parallel_calls = false;
    bool assistant_calls = false;
    bool tool_role = false;
    std::string source;
};

struct InteractionSpec {
    ChatTemplateProgram chat;
    ToolProtocolSpec tools;
    std::optional<std::string> vision_pipeline;
};

} // namespace celeg
