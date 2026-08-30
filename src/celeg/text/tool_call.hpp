#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {

struct JsonSchema {
    std::string serialized;
};

struct ToolFunction {
    std::string name;
    std::string description;
    JsonSchema parameters;
    bool strict = false;
};

struct ToolDefinition {
    std::string type = "function";
    ToolFunction function;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments;
};

enum class ToolChoiceMode {
    None,
    Auto,
    Required,
    Specific,
};

struct ToolChoice {
    ToolChoiceMode mode = ToolChoiceMode::Auto;
    std::string function_name;
};

enum class ToolParseStatus {
    NotToolCall,
    Complete,
    Incomplete,
    Invalid,
};

struct ToolParseResult {
    ToolParseStatus status = ToolParseStatus::NotToolCall;
    std::string assistant_text;
    std::vector<ToolCall> calls;
    std::string error;
    std::size_t consumed_bytes = 0;
};

}
