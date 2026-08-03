#include "celeg/text/chat_template.hpp"
#include "support/assertions.hpp"
#include <iostream>
#include <stdexcept>
#include <span>
#include <string>
#include <vector>

int main() {
    auto catalog = celeg::make_chat_profile_catalog();
    const auto& tmpl = catalog.find("lfm2-instruct");

    const std::string user_only = tmpl.format(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "Hello"}}, true);
    CELEG_TEST_CHECK(user_only == "<|startoftext|><|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n");

    const std::string with_system = tmpl.format(
        std::vector<celeg::ChatMessage>{
            {celeg::ChatRole::System, "You are helpful."},
            {celeg::ChatRole::User, "Hi"},
        },
        true);
    CELEG_TEST_CHECK(with_system ==
        "<|startoftext|><|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");

    const std::string multi_turn = tmpl.format(
        std::vector<celeg::ChatMessage>{
            {celeg::ChatRole::System, "You are helpful."},
            {celeg::ChatRole::User, "Hi"},
            {celeg::ChatRole::Assistant, "Hello!"},
            {celeg::ChatRole::User, "How are you?"},
        },
        true);
    CELEG_TEST_CHECK(multi_turn ==
        "<|startoftext|><|im_start|>system\nYou are helpful.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\nHello!<|im_end|>\n"
        "<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n");

    const std::string no_generation_prompt = tmpl.format(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "Hi"}}, false);
    CELEG_TEST_CHECK(no_generation_prompt ==
        "<|startoftext|><|im_start|>user\nHi<|im_end|>\n");

    const std::string tool_turn = tmpl.format(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::Tool, "result"}}, true);
    CELEG_TEST_CHECK(tool_turn.find("<|tool_response_start|>result<|tool_response_end|>") != std::string::npos);
    const auto lfm2_capabilities = catalog.capabilities("lfm2-instruct");
    CELEG_TEST_CHECK(lfm2_capabilities.native_tool_call_codec);
    CELEG_TEST_CHECK(lfm2_capabilities.tool_call_codec != nullptr);
    const auto lfm2_parse = lfm2_capabilities.tool_call_codec->parse_generation(
        "<|tool_call_start|>[weather(city='Paris', unit='C')]<|tool_call_end|>");
    CELEG_TEST_CHECK(lfm2_parse.status == celeg::ToolParseStatus::Complete);
    CELEG_TEST_CHECK(lfm2_parse.calls.size() == 1);
    CELEG_TEST_CHECK(lfm2_parse.calls[0].name == "weather");
    CELEG_TEST_CHECK(lfm2_parse.calls[0].arguments == "{\"city\":\"Paris\",\"unit\":\"C\"}");
    const auto lfm2_incomplete = lfm2_capabilities.tool_call_codec->parse_generation(
        "<|tool_call_start|>[weather(city=\"Paris\")");
    CELEG_TEST_CHECK(lfm2_incomplete.status == celeg::ToolParseStatus::Incomplete);
    const celeg::ToolDefinition tool_definition{
        "function", {"weather", "Weather lookup", {{"{\"type\":\"object\"}"}}, false}};
    const celeg::ToolChoice tool_choice{celeg::ToolChoiceMode::Auto, {}};
    CELEG_TEST_CHECK(lfm2_capabilities.tool_call_codec->render_tool_definitions(
        std::span<const celeg::ToolDefinition>(&tool_definition, 1), tool_choice).find("weather") != std::string::npos);

    const auto& granite = catalog.find("granite-instruct");
    const auto& granite_by_id = catalog.find("granite-instruct");

    const std::string granite_user_only = granite.format(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "Hello"}}, true);
    CELEG_TEST_CHECK(granite_user_only ==
        "<|startoftext|><|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024. "
        "You are Granite, developed by IBM. You are a helpful AI assistant."
        "<|end_of_text|>\n"
        "<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>\n"
        "<|start_of_role|>assistant<|end_of_role|>");

    const std::string granite_with_system = granite.format(
        std::vector<celeg::ChatMessage>{
            {celeg::ChatRole::System, "You are helpful."},
            {celeg::ChatRole::User, "Hi"},
        },
        true);
    CELEG_TEST_CHECK(granite_with_system ==
        "<|startoftext|><|start_of_role|>system<|end_of_role|>You are helpful.<|end_of_text|>\n"
        "<|start_of_role|>user<|end_of_role|>Hi<|end_of_text|>\n"
        "<|start_of_role|>assistant<|end_of_role|>");

    const auto& gemma = catalog.find("gemma4-instruct");
    const auto gemma_capabilities = catalog.capabilities("gemma4-instruct");
    CELEG_TEST_CHECK(gemma_capabilities.vision);
    CELEG_TEST_CHECK(gemma_capabilities.tool_call_codec != nullptr);
    const auto gemma_parse = gemma_capabilities.tool_call_codec->parse_generation(
        "<|tool_call>call:weather{\"city\":\"Paris\"}<tool_call|>");
    CELEG_TEST_CHECK(gemma_parse.status == celeg::ToolParseStatus::Complete);
    CELEG_TEST_CHECK(gemma_parse.calls.size() == 1);
    CELEG_TEST_CHECK(gemma_parse.calls[0].name == "weather");
    CELEG_TEST_CHECK(gemma_parse.calls[0].arguments == "{\"city\":\"Paris\"}");

    const auto& minicpm5 = catalog.find("minicpm5-instruct");
    const auto minicpm5_capabilities = catalog.capabilities("minicpm5-instruct");
    CELEG_TEST_CHECK(!minicpm5_capabilities.vision);
    CELEG_TEST_CHECK(minicpm5_capabilities.developer_messages);
    CELEG_TEST_CHECK(minicpm5_capabilities.parallel_tool_calls);
    CELEG_TEST_CHECK(minicpm5_capabilities.tool_call_codec != nullptr);
    const std::string minicpm5_prompt = minicpm5.format(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "What is the weather?"}}, true);
    CELEG_TEST_CHECK(minicpm5_prompt.find("<bos><|im_start|>user\nWhat is the weather?") == 0);
    CELEG_TEST_CHECK(minicpm5_prompt.find("<|im_start|>assistant\n<think>\n\n</think>\n\n") != std::string::npos);

    const auto minicpm5_tool_parse = minicpm5_capabilities.tool_call_codec->parse_generation(
        "I will check. <function name=\"weather\"><param name=\"city\">\"Paris\"</param>"
        "<param name=\"unit\">\"C\"</param></function>");
    CELEG_TEST_CHECK(minicpm5_tool_parse.status == celeg::ToolParseStatus::Complete);
    CELEG_TEST_CHECK(minicpm5_tool_parse.calls.size() == 1);
    CELEG_TEST_CHECK(minicpm5_tool_parse.calls[0].name == "weather");
    CELEG_TEST_CHECK(minicpm5_tool_parse.calls[0].arguments ==
                     "{\"city\":\"Paris\",\"unit\":\"C\"}");
    const std::string minicpm5_tools = minicpm5.format(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "weather"}},
        std::vector<celeg::ChatToolDefinition>{tool_definition}, true);
    CELEG_TEST_CHECK(minicpm5_tools.find("<tools>") != std::string::npos);
    CELEG_TEST_CHECK(minicpm5_tools.find("\"name\":\"weather\"") != std::string::npos);

    std::cout << "chat_template_test: ok\n";
}
