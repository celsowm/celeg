#include "chat_completions.hpp"

#include "celeg/serve/protocol/chat.hpp"
#include "celeg/serve/protocol/json.hpp"
#include "celeg/serve/protocol/mapping.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace celeg::app::serve {

namespace {

namespace protocol = celeg::serve::protocol;
// Large enough for typical base64-encoded images while bounding per-request memory.
constexpr std::size_t kMaxRequestBodyBytes = 32 * 1024 * 1024;
// Context trimming is performed before submission, after exact tokenization.
using celeg::serve::GenerateEvent;
using celeg::serve::GenerateRequest;
using celeg::serve::FinishReason;
using celeg::serve::GenerationDispatcher;
using celeg::serve::IRequestService;
using celeg::serve::RequestId;

std::int64_t now_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string new_request_id() {
    static std::atomic<std::uint64_t> counter{0};
    return "chatcmpl-" + std::to_string(counter.fetch_add(1) + 1);
}

// Detaches the HTTP-facing watcher after a client disconnect mid-stream. The
// dispatcher owns cancellation and drains the request to a terminal state
// without touching the now-invalid HttpResponse.
void forget_after_abort(GenerationDispatcher& dispatcher, RequestId id) {
    dispatcher.cancel(id);
}

} // namespace

void register_chat_completions_route(uWS::App& app,
                                     GenerationDispatcher& dispatcher,
                                     IRequestService& service,
                                     const celeg::ITokenizer& tokenizer,
                                     const celeg::IChatTemplate& chat_template,
                                     const celeg::ChatCapabilities& capabilities,
                                     const celeg::IChatToolCallCodec* tool_codec,
                                     const std::string& model_name,
                                     std::span<const std::int32_t> eos_token_ids,
                                     std::size_t max_context_tokens,
                                     uWS::Loop* loop) {
    const std::vector<std::int32_t> stop_tokens(eos_token_ids.begin(), eos_token_ids.end());
    app.post("/v1/chat/completions", [&dispatcher, &service, &tokenizer, &chat_template, &capabilities, tool_codec, &model_name,
                                       stop_tokens, max_context_tokens, loop](auto* res, auto* /*req*/) {
        const celeg::IChatToolCallCodec* request_tool_codec = tool_codec;
        struct State {
            std::string body;
            std::atomic<bool> aborted{false};
            bool rejected = false;
            std::optional<RequestId> id;
        };
        auto state = std::make_shared<State>();

        res->onAborted([state, &dispatcher]() {
            state->aborted.store(true);
            if (state->id) forget_after_abort(dispatcher, *state->id);
        });

        res->onData([res, state, &dispatcher, &service, &tokenizer, &chat_template, &capabilities,
                     request_tool_codec, &model_name,
                     stop_tokens, max_context_tokens, loop](std::string_view chunk, bool last) {
            if (state->rejected) return;
            if (chunk.size() > kMaxRequestBodyBytes - state->body.size()) {
                state->rejected = true;
                res->writeStatus("413 Payload Too Large")
                    ->writeHeader("Content-Type", "application/json")
                    ->end(protocol::to_json(protocol::error_response("request body exceeds 32 MiB limit")));
                return;
            }
            state->body.append(chunk);
            if (!last) return;

            protocol::ChatCompletionRequest request;
            GenerateRequest generate_request;
            try {
                request = protocol::from_json<protocol::ChatCompletionRequest>(state->body);
                if (request.model.empty()) {
                    throw std::invalid_argument("model is required");
                }
                if (request.model != model_name) {
                    throw std::invalid_argument("model is not available: " + request.model);
                }
                generate_request = protocol::to_generate_request(
                    request, tokenizer, chat_template, capabilities, stop_tokens, {},
                    max_context_tokens, request_tool_codec);
            } catch (const std::exception& error) {
                res->writeStatus("400 Bad Request")
                    ->writeHeader("Content-Type", "application/json")
                    ->end(protocol::to_json(protocol::error_response(error.what())));
                return;
            }

            const bool stream = request.stream && *request.stream;
            const std::size_t prompt_tokens = generate_request.prompt_tokens.size();
            const bool context_window_trimmed = generate_request.context_window_trimmed;
            const std::string id_str = new_request_id();
            const std::int64_t created = now_seconds();

            if (state->aborted.load()) return;
            RequestId id = 0;
            try {
                id = service.submit(std::move(generate_request));
            } catch (const std::exception& error) {
                res->writeStatus("400 Bad Request")
                    ->writeHeader("Content-Type", "application/json")
                    ->end(protocol::to_json(protocol::error_response(error.what())));
                return;
            }
            state->id = id;

            if (!stream) {
                auto completion = std::make_shared<std::vector<std::int32_t>>();
                dispatcher.watch(id, [res, state, completion, id_str, created, prompt_tokens,
                                      &tokenizer, &capabilities, request_tool_codec, &model_name,
                                      context_window_trimmed, loop](const GenerateEvent& event) {
                    completion->insert(completion->end(), event.tokens.begin(), event.tokens.end());
                    if (!event.finished) return;
                    loop->defer([res, state, completion, id_str, created, prompt_tokens,
                                 &tokenizer, &capabilities, request_tool_codec, &model_name, context_window_trimmed,
                                 reason = event.finish_reason, error = event.error] {
                        if (state->aborted.load()) return;
                        if (reason == FinishReason::Error && !error.empty()) {
                            res->writeStatus("500 Internal Server Error")
                                ->writeHeader("Content-Type", "application/json")
                                ->end(protocol::to_json(protocol::error_response(error)));
                            return;
                        }
                        const auto response = protocol::to_chat_completion_response(
                            id_str, model_name, created, prompt_tokens, *completion, reason, tokenizer, request_tool_codec);
                        if (context_window_trimmed) {
                            res->writeHeader("X-Celeg-Context-Trimmed", "true");
                        }
                        res->writeHeader("Content-Type", "application/json")
                            ->end(protocol::to_json(response));
                    });
                });
            } else {
                res->writeHeader("Content-Type", "text/event-stream")
                    ->writeHeader("Cache-Control", "no-cache");
                if (context_window_trimmed) {
                    res->writeHeader("X-Celeg-Context-Trimmed", "true");
                }
                auto first = std::make_shared<bool>(true);
                auto interpreter = std::make_shared<celeg::serve::ChatGenerationInterpreter>(tokenizer, request_tool_codec);
                auto completion_tokens = std::make_shared<std::size_t>(0);
                const bool include_usage = request.stream_options && request.stream_options->include_usage;
                dispatcher.watch(id, [res, state, first, id_str, created, &model_name,
                                      interpreter, completion_tokens, prompt_tokens, include_usage, loop](const GenerateEvent& event) {
                    *completion_tokens += event.tokens.size();
                    const auto delta = interpreter->consume(event.tokens, event.finished);
                    const FinishReason semantic_reason = event.finish_reason == FinishReason::Error
                        ? FinishReason::Error
                        : (delta.finish_reason != FinishReason::None
                            ? delta.finish_reason : event.finish_reason);
                    loop->defer([res, state, first, id_str, created, &model_name,
                                 delta,
                                 finished = event.finished,
                                 reason = semantic_reason, prompt_tokens, include_usage,
                                 completion_token_count = *completion_tokens,
                                 error = event.error] {
                        if (state->aborted.load()) return;
                        if (!delta.text.empty() || !delta.tool_calls.empty() || *first) {
                            const auto chunk = protocol::to_chat_completion_chunk(
                                id_str, model_name, created, delta, *first, std::nullopt);
                            *first = false;
                            res->write("data: " + protocol::to_json(chunk) + "\n\n");
                        }
                        if (finished) {
                            const auto final_chunk = protocol::to_chat_completion_chunk(
                                id_str, model_name, created, celeg::serve::ChatGenerationDelta{}, false, reason);
                            res->write("data: " + protocol::to_json(final_chunk) + "\n\n");
                            if (reason == FinishReason::Error && !error.empty()) {
                                res->write("data: " + protocol::to_json(protocol::error_response(error)) + "\n\n");
                            }
                            if (include_usage) {
                                protocol::ChatCompletionChunk usage_chunk;
                                usage_chunk.id = id_str;
                                usage_chunk.model = model_name;
                                usage_chunk.created = created;
                                usage_chunk.usage = protocol::Usage{
                                    prompt_tokens, completion_token_count,
                                    prompt_tokens + completion_token_count};
                                res->write("data: " + protocol::to_json(usage_chunk) + "\n\n");
                            }
                            res->write(std::string_view("data: [DONE]\n\n"));
                            res->end();
                        }
                    });
                });
            }
        });
    });
}

} // namespace celeg::app::serve
