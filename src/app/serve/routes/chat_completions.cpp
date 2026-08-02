#include "chat_completions.hpp"

#include "celeg/serve/protocol/chat.hpp"
#include "celeg/serve/protocol/json.hpp"
#include "celeg/serve/protocol/mapping.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace celeg::app::serve {

namespace {

namespace protocol = celeg::serve::protocol;
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

// Detaches the HTTP-facing watcher after a client disconnect mid-stream: the
// dispatcher's own dispatch_once() already erases + releases any watched
// request once it reaches a terminal status, so re-watching with a no-op
// callback is enough to let cancellation finish draining without touching
// the now-invalid HttpResponse.
void forget_after_abort(GenerationDispatcher& dispatcher, IRequestService& service, RequestId id) {
    service.cancel(id);
    dispatcher.watch(id, [](const GenerateEvent&) {});
}

} // namespace

void register_chat_completions_route(uWS::App& app,
                                     GenerationDispatcher& dispatcher,
                                     IRequestService& service,
                                     const celeg::BpeTokenizer& tokenizer,
                                     const celeg::IChatTemplate& chat_template,
                                     const celeg::ChatCapabilities& capabilities,
                                     const std::string& model_name,
                                     std::int32_t eos_token_id,
                                     uWS::Loop* loop) {
    app.post("/v1/chat/completions", [&dispatcher, &service, &tokenizer, &chat_template, &capabilities, &model_name, eos_token_id,
                                      loop](auto* res, auto* /*req*/) {
        struct State {
            std::string body;
            std::atomic<bool> aborted{false};
            std::optional<RequestId> id;
        };
        auto state = std::make_shared<State>();

        res->onAborted([state, &dispatcher, &service]() {
            state->aborted.store(true);
            if (state->id) forget_after_abort(dispatcher, service, *state->id);
        });

        res->onData([res, state, &dispatcher, &service, &tokenizer, &chat_template, &capabilities, &model_name,
                     eos_token_id, loop](std::string_view chunk, bool last) {
            state->body.append(chunk);
            if (!last) return;

            protocol::ChatCompletionRequest request;
            GenerateRequest generate_request;
            try {
                request = protocol::from_json<protocol::ChatCompletionRequest>(state->body);
                generate_request = protocol::to_generate_request(
                    request, tokenizer, chat_template, capabilities, eos_token_id);
            } catch (const std::exception& error) {
                res->writeStatus("400 Bad Request")
                    ->writeHeader("Content-Type", "application/json")
                    ->end(protocol::to_json(protocol::error_response(error.what())));
                return;
            }

            const bool stream = request.stream && *request.stream;
            const std::size_t prompt_tokens = generate_request.prompt_tokens.size();
            const std::string id_str = new_request_id();
            const std::int64_t created = now_seconds();

            const RequestId id = service.submit(std::move(generate_request));
            state->id = id;

            if (!stream) {
                auto completion = std::make_shared<std::vector<std::int32_t>>();
                dispatcher.watch(id, [res, state, completion, id_str, created, prompt_tokens,
                                      &tokenizer, &capabilities, &model_name, loop](const GenerateEvent& event) {
                    completion->insert(completion->end(), event.tokens.begin(), event.tokens.end());
                    if (!event.finished) return;
                    loop->defer([res, state, completion, id_str, created, prompt_tokens,
                                 &tokenizer, &capabilities, &model_name, reason = event.finish_reason] {
                        if (state->aborted.load()) return;
                        const auto response = protocol::to_chat_completion_response(
                            id_str, model_name, created, prompt_tokens, *completion, reason, tokenizer, capabilities);
                        res->writeHeader("Content-Type", "application/json")
                            ->end(protocol::to_json(response));
                    });
                });
            } else {
                res->writeHeader("Content-Type", "text/event-stream")
                    ->writeHeader("Cache-Control", "no-cache");
                auto first = std::make_shared<bool>(true);
                auto interpreter = std::make_shared<celeg::serve::ChatGenerationInterpreter>(tokenizer, capabilities);
                dispatcher.watch(id, [res, state, first, id_str, created, &model_name,
                                      interpreter, loop](const GenerateEvent& event) {
                    const auto delta = interpreter->consume(event.tokens, event.finished);
                    const FinishReason semantic_reason =
                        delta.finish_reason != FinishReason::None ? delta.finish_reason : event.finish_reason;
                    loop->defer([res, state, first, id_str, created, &model_name,
                                 delta,
                                 finished = event.finished,
                                 reason = semantic_reason] {
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
