#include "celeg/serve/protocol/json.hpp"
#include "celeg/serve/protocol/mapping.hpp"
#include "celeg/text/tokenizer.hpp"
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
celeg::BpeTokenizer make_test_tokenizer(const std::filesystem::path& path) {
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
    return celeg::BpeTokenizer(path.string());
}

} // namespace

int main() {
    namespace serve = celeg::serve;
    namespace protocol = celeg::serve::protocol;

    // Role mapping.
    CELEG_TEST_CHECK(protocol::role_from_string("system") == celeg::ChatRole::System);
    CELEG_TEST_CHECK(protocol::role_from_string("developer") == celeg::ChatRole::Developer);
    CELEG_TEST_CHECK(protocol::role_from_string("user") == celeg::ChatRole::User);
    CELEG_TEST_CHECK(protocol::role_from_string("assistant") == celeg::ChatRole::Assistant);
    CELEG_TEST_CHECK(protocol::role_from_string("tool") == celeg::ChatRole::Tool);
    bool threw = false;
    try {
        protocol::role_from_string("narrator");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CELEG_TEST_CHECK(threw);
    CELEG_TEST_CHECK(protocol::role_to_string(celeg::ChatRole::User) == "user");

    // Finish reason mapping.
    CELEG_TEST_CHECK(protocol::finish_reason_to_string(serve::FinishReason::None) == "");
    CELEG_TEST_CHECK(protocol::finish_reason_to_string(serve::FinishReason::Stop) == "stop");
    CELEG_TEST_CHECK(protocol::finish_reason_to_string(serve::FinishReason::Length) == "length");
    CELEG_TEST_CHECK(protocol::finish_reason_to_string(serve::FinishReason::Cancelled) == "cancelled");
    CELEG_TEST_CHECK(protocol::finish_reason_to_string(serve::FinishReason::Error) == "error");

    // Request JSON round-trip: absent optional fields must not appear on the
    // wire, and present ones must survive a write/read cycle.
    protocol::ChatCompletionRequest request;
    request.model = "lfm2.5-test";
    request.messages.push_back({"user", "hi"});
    request.max_tokens = 16;
    request.temperature = 0.5;
    request.top_p = 0.9;
    request.top_k = 40;
    request.seed = 42;

    const std::string request_json = protocol::to_json(request);
    CELEG_TEST_CHECK(request_json.find("\"stream\"") == std::string::npos);

    const protocol::ChatCompletionRequest parsed = protocol::from_json<protocol::ChatCompletionRequest>(request_json);
    CELEG_TEST_CHECK(parsed.model == "lfm2.5-test");
    CELEG_TEST_CHECK(parsed.messages.size() == 1);
    CELEG_TEST_CHECK(parsed.messages[0].role == "user");
    CELEG_TEST_CHECK(parsed.messages[0].content == "hi");
    CELEG_TEST_CHECK(parsed.max_tokens && *parsed.max_tokens == 16);
    CELEG_TEST_CHECK(parsed.seed && *parsed.seed == 42);
    CELEG_TEST_CHECK(!parsed.stream.has_value());

    // Full request -> GenerateRequest pipeline against a real tokenizer.
    const auto tokenizer_path = std::filesystem::temp_directory_path() / "celeg_chat_protocol_test.json";
    const celeg::BpeTokenizer tokenizer = make_test_tokenizer(tokenizer_path);
    std::filesystem::remove(tokenizer_path);

    const serve::GenerateRequest generate_request = protocol::to_generate_request(request, tokenizer, /*eos_token_id=*/2);
    CELEG_TEST_CHECK(generate_request.eos_token_id == 2);
    CELEG_TEST_CHECK(generate_request.max_output_tokens == 16);
    CELEG_TEST_CHECK(generate_request.generation.temperature == 0.5f);
    CELEG_TEST_CHECK(generate_request.generation.top_p == 0.9f);
    CELEG_TEST_CHECK(generate_request.generation.top_k == 40);
    CELEG_TEST_CHECK(generate_request.generation.seed == 42);

    // "<|startoftext|><|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    const std::vector<std::int32_t> expected_prompt_tokens = {
        0, 1, 3, 4, 5, 6, 12, 7, 8, 2, 12, 1, 9, 4, 4, 8, 4, 11, 9, 10, 11, 12};
    CELEG_TEST_CHECK(generate_request.prompt_tokens == expected_prompt_tokens);

    // Missing messages must be rejected before touching the tokenizer.
    protocol::ChatCompletionRequest empty_request;
    empty_request.model = "lfm2.5-test";
    threw = false;
    try {
        protocol::to_generate_request(empty_request, tokenizer, 2);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CELEG_TEST_CHECK(threw);

    // Response mapping: decode completion tokens "h","i" -> "hi".
    const std::vector<std::int32_t> completion_tokens = {7, 8};
    const protocol::ChatCompletionResponse response = protocol::to_chat_completion_response(
        "req-1", "lfm2.5-test", 1000, generate_request.prompt_tokens.size(),
        completion_tokens, serve::FinishReason::Stop, tokenizer);
    CELEG_TEST_CHECK(response.id == "req-1");
    CELEG_TEST_CHECK(response.object == "chat.completion");
    CELEG_TEST_CHECK(response.choices.size() == 1);
    CELEG_TEST_CHECK(response.choices[0].message.role == "assistant");
    CELEG_TEST_CHECK(response.choices[0].message.content == "hi");
    CELEG_TEST_CHECK(response.choices[0].finish_reason == "stop");
    CELEG_TEST_CHECK(response.usage.prompt_tokens == generate_request.prompt_tokens.size());
    CELEG_TEST_CHECK(response.usage.completion_tokens == 2);
    CELEG_TEST_CHECK(response.usage.total_tokens == generate_request.prompt_tokens.size() + 2);

    const std::string response_json = protocol::to_json(response);
    const protocol::ChatCompletionResponse reparsed = protocol::from_json<protocol::ChatCompletionResponse>(response_json);
    CELEG_TEST_CHECK(reparsed.choices[0].message.content == "hi");
    CELEG_TEST_CHECK(reparsed.usage.total_tokens == response.usage.total_tokens);

    // Streaming chunk mapping: first chunk carries the role, later ones don't;
    // only the terminal chunk carries a finish_reason.
    const protocol::ChatCompletionChunk first_chunk = protocol::to_chat_completion_chunk(
        "req-1", "lfm2.5-test", 1000, completion_tokens, /*include_role=*/true,
        /*finish=*/std::nullopt, tokenizer);
    CELEG_TEST_CHECK(first_chunk.choices[0].delta.role && *first_chunk.choices[0].delta.role == "assistant");
    CELEG_TEST_CHECK(first_chunk.choices[0].delta.content && *first_chunk.choices[0].delta.content == "hi");
    CELEG_TEST_CHECK(!first_chunk.choices[0].finish_reason.has_value());

    const protocol::ChatCompletionChunk final_chunk = protocol::to_chat_completion_chunk(
        "req-1", "lfm2.5-test", 1000, {}, /*include_role=*/false,
        /*finish=*/serve::FinishReason::Stop, tokenizer);
    CELEG_TEST_CHECK(!final_chunk.choices[0].delta.role.has_value());
    CELEG_TEST_CHECK(!final_chunk.choices[0].delta.content.has_value());
    CELEG_TEST_CHECK(final_chunk.choices[0].finish_reason && *final_chunk.choices[0].finish_reason == "stop");

    std::cout << "chat_protocol_mapping_test: ok\n";
    return 0;
}
