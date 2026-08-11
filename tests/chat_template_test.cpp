#include "celeg/text/chat_template.hpp"
#include "celeg/text/semantic_chat_templates.hpp"
#include "celeg/text/semantic_chat_templates.hpp"
#include "celeg/text/semantic_chat_templates.hpp"
#include "celeg/text/semantic_chat_templates.hpp"
#include "celeg/text/semantic_chat_templates.hpp"
#include "celeg/text/semantic_chat_templates.hpp"
#include "celeg/text/semantic_chat_templates.hpp"
#include "celeg/text/semantic_chat_templates.hpp"
#include "celeg/text/semantic_chat_templates.hpp"
#include "support/assertions.hpp"
#include <iostream>
#include <stdexcept>
#include <span>
#include <string>
#include <vector>

int main() {
    auto catalog = celeg::make_chat_template_catalog();
    const auto& tmpl = catalog.find("chat:delimited");

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
    bool unsupported_option_rejected = false;
    try {
        tmpl.format(std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "Hello"}},
                    {}, true, celeg::ChatTemplateOptions{false});
    } catch (const std::invalid_argument&) {
        unsupported_option_rejected = true;
    }
    CELEG_TEST_CHECK(unsupported_option_rejected);
    const auto lfm2_capabilities = catalog.capabilities("chat:delimited");
    CELEG_TEST_CHECK(lfm2_capabilities.native_tool_call_codec);
    const auto* lfm2_codec = catalog.tool_codec("chat:delimited");
    CELEG_TEST_CHECK(lfm2_codec != nullptr);
    const auto lfm2_parse = lfm2_codec->parse_generation(
        "<|tool_call_start|>[weather(city='Paris', unit='C')]<|tool_call_end|>");
    CELEG_TEST_CHECK(lfm2_parse.status == celeg::ToolParseStatus::Complete);
    CELEG_TEST_CHECK(lfm2_parse.calls.size() == 1);
    CELEG_TEST_CHECK(lfm2_parse.calls[0].name == "weather");
    CELEG_TEST_CHECK(lfm2_parse.calls[0].arguments == "{\"city\":\"Paris\",\"unit\":\"C\"}");
    const auto lfm2_incomplete = lfm2_codec->parse_generation(
        "<|tool_call_start|>[weather(city=\"Paris\")");
    CELEG_TEST_CHECK(lfm2_incomplete.status == celeg::ToolParseStatus::Incomplete);
    const celeg::ToolDefinition tool_definition{
        "function", {"weather", "Weather lookup", {{"{\"type\":\"object\"}"}}, false}};
    const celeg::ToolChoice tool_choice{celeg::ToolChoiceMode::Auto, {}};
    CELEG_TEST_CHECK(lfm2_codec->render_tool_definitions(
        std::span<const celeg::ToolDefinition>(&tool_definition, 1), tool_choice).find("weather") != std::string::npos);

    const auto& granite = catalog.find("chat:role-envelope");
    const auto& granite_by_id = catalog.find("chat:role-envelope");

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

    const auto& gemma = catalog.find("chat:turn");
    const auto gemma_capabilities = catalog.capabilities("chat:turn");
    CELEG_TEST_CHECK(gemma_capabilities.vision);
    const auto* gemma_codec = catalog.tool_codec("chat:turn");
    CELEG_TEST_CHECK(gemma_codec != nullptr);
    const auto gemma_parse = gemma_codec->parse_generation(
        "<|tool_call>call:weather{\"city\":\"Paris\"}<tool_call|>");
    CELEG_TEST_CHECK(gemma_parse.status == celeg::ToolParseStatus::Complete);
    CELEG_TEST_CHECK(gemma_parse.calls.size() == 1);
    CELEG_TEST_CHECK(gemma_parse.calls[0].name == "weather");
    CELEG_TEST_CHECK(gemma_parse.calls[0].arguments == "{\"city\":\"Paris\"}");

    const auto& qwen35 = catalog.find("chat:vision-role");
    const auto qwen35_capabilities = catalog.capabilities("chat:vision-role");
    CELEG_TEST_CHECK(qwen35_capabilities.vision);
    CELEG_TEST_CHECK(qwen35_capabilities.image_marker == "<|image_pad|>");
    const std::string qwen35_prompt = qwen35.format(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "Describe <|image|>."}}, true);
    CELEG_TEST_CHECK(qwen35_prompt ==
        "<|im_start|>user\nDescribe <|vision_start|><|image_pad|><|vision_end|>.<|im_end|>\n"
        "<|im_start|>assistant\n");

    const auto& muse = catalog.find("chat:patch-role");
    const auto muse_capabilities = catalog.capabilities("chat:patch-role");
    CELEG_TEST_CHECK(muse_capabilities.vision);
    CELEG_TEST_CHECK(!muse_capabilities.video);
    CELEG_TEST_CHECK(muse_capabilities.image_marker == "<|patch|>");
    const std::string muse_prompt = muse.format(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "Describe <|image|>."}}, true);
    CELEG_TEST_CHECK(muse_prompt.find("<|patch|>") != std::string::npos);
    CELEG_TEST_CHECK(muse_prompt.ends_with("<|start|>assistant"));

    const auto& nemotron = catalog.find("chat:thinking-role");
    CELEG_TEST_CHECK(!catalog.capabilities("chat:thinking-role").native_tool_call_codec);
    const std::string nemotron_think = nemotron.format(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "Solve this"}}, true);
    CELEG_TEST_CHECK(nemotron_think.ends_with(
        "<|im_start|>assistant\n<think>\n"));
    const std::string nemotron_no_think = nemotron.format(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "Solve this"}},
        {}, true, celeg::ChatTemplateOptions{false});
    CELEG_TEST_CHECK(nemotron_no_think.ends_with(
        "<|im_start|>assistant\n<think></think>"));

    const auto& minicpm5 = catalog.find("chat:thinking-function");
    const auto minicpm5_capabilities = catalog.capabilities("chat:thinking-function");
    CELEG_TEST_CHECK(!minicpm5_capabilities.vision);
    CELEG_TEST_CHECK(minicpm5_capabilities.roles.developer);
    CELEG_TEST_CHECK(minicpm5_capabilities.parallel_tool_calls);
    const auto* minicpm5_codec = catalog.tool_codec("chat:thinking-function");
    CELEG_TEST_CHECK(minicpm5_codec != nullptr);
    const std::string minicpm5_prompt = minicpm5.format(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "What is the weather?"}}, true);
    CELEG_TEST_CHECK(minicpm5_prompt.find("<bos><|im_start|>user\nWhat is the weather?") == 0);
    CELEG_TEST_CHECK(minicpm5_prompt.find("<|im_start|>assistant\n<think>\n\n</think>\n\n") != std::string::npos);

    const auto minicpm5_tool_parse = minicpm5_codec->parse_generation(
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

    const auto& nanbeige = catalog.find("chat:reasoning-xml");
    const std::string nanbeige_think = nanbeige.format(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "Solve this"}}, true);
    CELEG_TEST_CHECK(nanbeige_think.find("<|im_start|>system\n") == 0);
    CELEG_TEST_CHECK(nanbeige_think.ends_with("<|im_start|>assistant\n<think>\n"));
    const std::string nanbeige_no_think = nanbeige.format(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "Solve this"}},
        {}, true, celeg::ChatTemplateOptions{false});
    CELEG_TEST_CHECK(nanbeige_no_think.ends_with(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    const auto* nanbeige_codec = catalog.tool_codec("chat:reasoning-xml");
    CELEG_TEST_CHECK(nanbeige_codec != nullptr);
    const auto nanbeige_parse = nanbeige_codec->parse_generation(
        "answer <tool_call>\n<function=weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>");
    CELEG_TEST_CHECK(nanbeige_parse.status == celeg::ToolParseStatus::Complete);
    CELEG_TEST_CHECK(nanbeige_parse.calls.size() == 1);
    CELEG_TEST_CHECK(nanbeige_parse.calls[0].name == "weather");
    CELEG_TEST_CHECK(nanbeige_parse.calls[0].arguments == "{\"city\":\"Paris\"}");

    const auto& smollm3 = catalog.find("chat:metadata-thinking");
    const auto smollm3_capabilities = catalog.capabilities("chat:metadata-thinking");
    const auto* smollm3_codec = catalog.tool_codec("chat:metadata-thinking");
    CELEG_TEST_CHECK(smollm3_codec != nullptr);
    const std::string smollm3_think = smollm3.format(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "Solve this"}}, true);
    CELEG_TEST_CHECK(smollm3_think.find("Reasoning Mode: /think") != std::string::npos);
    CELEG_TEST_CHECK(smollm3_think.ends_with("<|im_start|>assistant\n"));
    const std::string smollm3_no_think = smollm3.format(
        std::vector<celeg::ChatMessage>{
            {celeg::ChatRole::System, "/no_think"},
            {celeg::ChatRole::User, "Solve this"}}, true);
    CELEG_TEST_CHECK(smollm3_no_think.find("Reasoning Mode: /no_think") != std::string::npos);
    CELEG_TEST_CHECK(smollm3_no_think.ends_with(
        "<|im_start|>assistant\n<think>\n\n</think>\n"));
    const celeg::ChatTemplateOptions no_think_options{false};
    const std::string smollm3_option_no_think = smollm3.format(
        std::vector<celeg::ChatMessage>{{celeg::ChatRole::User, "Explain"}},
        {}, true, no_think_options);
    CELEG_TEST_CHECK(smollm3_option_no_think.find("Reasoning Mode: /no_think") !=
                     std::string::npos);
    const auto smollm3_parse = smollm3_codec->parse_generation(
        "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>");
    CELEG_TEST_CHECK(smollm3_parse.status == celeg::ToolParseStatus::Complete);
    CELEG_TEST_CHECK(smollm3_parse.calls.size() == 1);
    CELEG_TEST_CHECK(smollm3_parse.calls[0].name == "get_weather");
    CELEG_TEST_CHECK(smollm3_parse.calls[0].arguments == "{\"city\":\"Paris\"}");

    std::cout << "chat_template_test: ok\n";
}
