#include "lfm/serve/protocol/json.hpp"
#include "lfm/serve/protocol/mapping.hpp"
#include "lfm/text/tokenizer.hpp"
#include "support/assertions.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

// Builds a tokenizer whose vocabulary covers exactly the characters used by
// the LFM2 Instruct template rendering of a single "hi" user turn, so the
// full format_chat -> encode pipeline in to_generate_request() is exercised
// end to end with a deterministic, hand-verifiable token sequence.
lfm::BpeTokenizer make_test_tokenizer(const std::filesystem::path& path) {
    std::ofstream out(path);
    out << R"({
      "model": {
        "type": "BPE",
        "vocab": {
          "<|startoftext|>": 0,
          "<|im_start|>": 1,
          "<|im_end|>": 2,
          "u": 3, "s": 4, "e": 5, "r": 6, "h": 7, "i": 8, "a": 9, "n": 10, "t": 11,
          "Ċ": 12
        },
        "merges": []
      },
      "added_tokens": [
        {"id": 0, "content": "<|startoftext|>", "special": true},
        {"id": 1, "content": "<|im_start|>", "special": true},
        {"id": 2, "content": "<|im_end|>", "special": true}
      ]
    })";
    out.close();
    return lfm::BpeTokenizer(path.string());
}

} // namespace

int main() {
    using namespace lfm::serve;
    using namespace lfm::serve::protocol;

    // Role mapping.
    LFM_TEST_CHECK(role_from_string("system") == lfm::ChatRole::System);
    LFM_TEST_CHECK(role_from_string("developer") == lfm::ChatRole::Developer);
    LFM_TEST_CHECK(role_from_string("user") == lfm::ChatRole::User);
    LFM_TEST_CHECK(role_from_string("assistant") == lfm::ChatRole::Assistant);
    LFM_TEST_CHECK(role_from_string("tool") == lfm::ChatRole::Tool);
    bool threw = false;
    try {
        role_from_string("narrator");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    LFM_TEST_CHECK(threw);
    LFM_TEST_CHECK(role_to_string(lfm::ChatRole::User) == "user");

    // Finish reason mapping.
    LFM_TEST_CHECK(finish_reason_to_string(FinishReason::None) == "");
    LFM_TEST_CHECK(finish_reason_to_string(FinishReason::Stop) == "stop");
    LFM_TEST_CHECK(finish_reason_to_string(FinishReason::Length) == "length");
    LFM_TEST_CHECK(finish_reason_to_string(FinishReason::Cancelled) == "cancelled");
    LFM_TEST_CHECK(finish_reason_to_string(FinishReason::Error) == "error");

    // Request JSON round-trip: absent optional fields must not appear on the
    // wire, and present ones must survive a write/read cycle.
    ChatCompletionRequest request;
    request.model = "lfm2.5-test";
    request.messages.push_back({"user", "hi"});
    request.max_tokens = 16;
    request.temperature = 0.5;
    request.top_p = 0.9;
    request.top_k = 40;
    request.seed = 42;

    const std::string request_json = to_json(request);
    LFM_TEST_CHECK(request_json.find("\"stream\"") == std::string::npos);

    const ChatCompletionRequest parsed = from_json<ChatCompletionRequest>(request_json);
    LFM_TEST_CHECK(parsed.model == "lfm2.5-test");
    LFM_TEST_CHECK(parsed.messages.size() == 1);
    LFM_TEST_CHECK(parsed.messages[0].role == "user");
    LFM_TEST_CHECK(parsed.messages[0].content == "hi");
    LFM_TEST_CHECK(parsed.max_tokens && *parsed.max_tokens == 16);
    LFM_TEST_CHECK(parsed.seed && *parsed.seed == 42);
    LFM_TEST_CHECK(!parsed.stream.has_value());

    // Full request -> GenerateRequest pipeline against a real tokenizer.
    const auto tokenizer_path = std::filesystem::temp_directory_path() / "lfm_chat_protocol_test.json";
    const lfm::BpeTokenizer tokenizer = make_test_tokenizer(tokenizer_path);
    std::filesystem::remove(tokenizer_path);

    const GenerateRequest generate_request = to_generate_request(request, tokenizer, /*eos_token_id=*/2);
    LFM_TEST_CHECK(generate_request.eos_token_id == 2);
    LFM_TEST_CHECK(generate_request.max_output_tokens == 16);
    LFM_TEST_CHECK(generate_request.generation.temperature == 0.5f);
    LFM_TEST_CHECK(generate_request.generation.top_p == 0.9f);
    LFM_TEST_CHECK(generate_request.generation.top_k == 40);
    LFM_TEST_CHECK(generate_request.generation.seed == 42);

    // "<|startoftext|><|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    const std::vector<std::int32_t> expected_prompt_tokens = {
        0, 1, 3, 4, 5, 6, 12, 7, 8, 2, 12, 1, 9, 4, 4, 8, 4, 11, 9, 10, 11, 12};
    LFM_TEST_CHECK(generate_request.prompt_tokens == expected_prompt_tokens);

    // Missing messages must be rejected before touching the tokenizer.
    ChatCompletionRequest empty_request;
    empty_request.model = "lfm2.5-test";
    threw = false;
    try {
        to_generate_request(empty_request, tokenizer, 2);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    LFM_TEST_CHECK(threw);

    // Response mapping: decode completion tokens "h","i" -> "hi".
    const std::vector<std::int32_t> completion_tokens = {7, 8};
    const ChatCompletionResponse response = to_chat_completion_response(
        "req-1", "lfm2.5-test", 1000, generate_request.prompt_tokens.size(),
        completion_tokens, FinishReason::Stop, tokenizer);
    LFM_TEST_CHECK(response.id == "req-1");
    LFM_TEST_CHECK(response.object == "chat.completion");
    LFM_TEST_CHECK(response.choices.size() == 1);
    LFM_TEST_CHECK(response.choices[0].message.role == "assistant");
    LFM_TEST_CHECK(response.choices[0].message.content == "hi");
    LFM_TEST_CHECK(response.choices[0].finish_reason == "stop");
    LFM_TEST_CHECK(response.usage.prompt_tokens == generate_request.prompt_tokens.size());
    LFM_TEST_CHECK(response.usage.completion_tokens == 2);
    LFM_TEST_CHECK(response.usage.total_tokens == generate_request.prompt_tokens.size() + 2);

    const std::string response_json = to_json(response);
    const ChatCompletionResponse reparsed = from_json<ChatCompletionResponse>(response_json);
    LFM_TEST_CHECK(reparsed.choices[0].message.content == "hi");
    LFM_TEST_CHECK(reparsed.usage.total_tokens == response.usage.total_tokens);

    // Streaming chunk mapping: first chunk carries the role, later ones don't;
    // only the terminal chunk carries a finish_reason.
    const ChatCompletionChunk first_chunk = to_chat_completion_chunk(
        "req-1", "lfm2.5-test", 1000, completion_tokens, /*include_role=*/true,
        /*finish=*/std::nullopt, tokenizer);
    LFM_TEST_CHECK(first_chunk.choices[0].delta.role && *first_chunk.choices[0].delta.role == "assistant");
    LFM_TEST_CHECK(first_chunk.choices[0].delta.content && *first_chunk.choices[0].delta.content == "hi");
    LFM_TEST_CHECK(!first_chunk.choices[0].finish_reason.has_value());

    const ChatCompletionChunk final_chunk = to_chat_completion_chunk(
        "req-1", "lfm2.5-test", 1000, {}, /*include_role=*/false,
        /*finish=*/FinishReason::Stop, tokenizer);
    LFM_TEST_CHECK(!final_chunk.choices[0].delta.role.has_value());
    LFM_TEST_CHECK(!final_chunk.choices[0].delta.content.has_value());
    LFM_TEST_CHECK(final_chunk.choices[0].finish_reason && *final_chunk.choices[0].finish_reason == "stop");

    std::cout << "chat_protocol_mapping_test: ok\n";
    return 0;
}
