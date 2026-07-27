#include "App.h"

#include "lfm/detail/checkpoint/bootstrap.hpp"
#include "lfm/model/config/config.hpp"
#include "lfm/serve/cpu_inference_service.hpp"
#include "lfm/serve/generation_dispatcher.hpp"
#include "lfm/serve/protocol/json.hpp"
#include "lfm/serve/protocol/mapping.hpp"
#include "lfm/serve/protocol/openapi.hpp"
#include "lfm/text/tokenizer.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using lfm::serve::FinishReason;
using lfm::serve::GenerateEvent;
using lfm::serve::GenerateRequest;
using lfm::serve::GenerationDispatcher;
using lfm::serve::IInferenceService;
using lfm::serve::RequestId;
namespace protocol = lfm::serve::protocol;

struct Args {
    std::string model_dir;
    std::string served_model_name;
    int port = 8080;
    int context = 4096;
    int threads = 0;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + key);
            return argv[i];
        };
        if (key == "--model") args.model_dir = value();
        else if (key == "--served-model-name") args.served_model_name = value();
        else if (key == "--port") args.port = std::stoi(value());
        else if (key == "--context") args.context = std::stoi(value());
        else if (key == "--threads") args.threads = std::stoi(value());
        else if (key == "--help") {
            std::cout << "lfm25-serve --model DIR [--port 8080] [--context 4096] "
                         "[--threads N] [--served-model-name NAME]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.model_dir.empty()) throw std::runtime_error("--model is required");
    return args;
}

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
void forget_after_abort(GenerationDispatcher& dispatcher, IInferenceService& service, RequestId id) {
    service.cancel(id);
    dispatcher.watch(id, [](const GenerateEvent&) {});
}

constexpr std::string_view kIndexHtml = R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><title>lfm25-serve</title>
<style>
body{font-family:system-ui,sans-serif;max-width:640px;margin:2rem auto;padding:0 1rem}
textarea{width:100%;height:5rem}
#out{white-space:pre-wrap;border:1px solid #ccc;padding:0.5rem;min-height:3rem;margin-top:1rem}
</style></head>
<body>
<h1>lfm25-serve</h1>
<p>POST <code>/v1/chat/completions</code> (OpenAI-compatible), browse the <a href="/docs">API docs</a>, or try it here:</p>
<textarea id="prompt">Hello!</textarea><br>
<button id="send">Send</button>
<div id="out"></div>
<script>
document.getElementById('send').onclick = async () => {
  const out = document.getElementById('out');
  out.textContent = '...';
  const res = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      model: 'lfm25',
      messages: [{role: 'user', content: document.getElementById('prompt').value}]
    })
  });
  const data = await res.json();
  out.textContent = data.choices ? data.choices[0].message.content : JSON.stringify(data);
};
</script>
</body></html>
)HTML";

constexpr std::string_view kDocsHtml = R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><title>lfm25-serve API docs</title>
<link rel="stylesheet" href="/docs/swagger-ui.css">
<style>body{margin:0}</style>
</head>
<body>
<div id="swagger-ui"></div>
<script src="/docs/swagger-ui-bundle.js"></script>
<script src="/docs/swagger-ui-standalone-preset.js"></script>
<script>
window.onload = () => {
  window.ui = SwaggerUIBundle({
    url: '/openapi.json',
    dom_id: '#swagger-ui',
    presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
    layout: 'StandaloneLayout'
  });
};
</script>
</body></html>
)HTML";

// Known Swagger UI dist assets served under /docs/<name>. Kept as an
// explicit allowlist (rather than general static-file serving) so a request
// path can never escape LFM_SWAGGER_UI_DIR.
const std::unordered_map<std::string, std::string>& swagger_ui_assets() {
    static const std::unordered_map<std::string, std::string> assets = {
        {"swagger-ui-bundle.js", "application/javascript"},
        {"swagger-ui-standalone-preset.js", "application/javascript"},
        {"swagger-ui.css", "text/css"},
    };
    return assets;
}

std::optional<std::string> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const std::filesystem::path model(args.model_dir);

        const lfm::detail::ModelBootstrap bootstrap = lfm::detail::load_model_bootstrap(model);
        const lfm::ModelConfig& config = bootstrap.config;
        const lfm::IModelVariant& variant = *bootstrap.variant;
        if (args.context > config.max_position_embeddings) {
            throw std::runtime_error("--context exceeds model maximum");
        }

        const lfm::BpeTokenizer tokenizer((model / "tokenizer.json").string(),
            lfm::make_chat_template(variant.chat_template_kind()));

        const std::string model_name =
            args.served_model_name.empty() ? std::string(variant.id()) : args.served_model_name;
        const std::int32_t eos_token_id = config.eos_token_id;

        lfm::CpuModelOptions model_options;
        model_options.threads = static_cast<std::size_t>(args.threads);
        lfm::CpuConcurrentEngineOptions engine_options;

        lfm::serve::CpuInferenceService service(
            (model / "model.safetensors").string(), args.context, model_options, engine_options);

        GenerationDispatcher dispatcher(service);
        dispatcher.start();

        uWS::Loop* loop = uWS::Loop::get();

        uWS::App()
            .get("/", [](auto* res, auto* /*req*/) {
                res->writeHeader("Content-Type", "text/html; charset=utf-8")->end(kIndexHtml);
            })
            .get("/healthz", [](auto* res, auto* /*req*/) { res->end("ok"); })
            .get("/v1/models", [&](auto* res, auto* /*req*/) {
                res->writeHeader("Content-Type", "application/json")
                    ->end("{\"object\":\"list\",\"data\":[{\"id\":\"" + model_name +
                          "\",\"object\":\"model\"}]}");
            })
            .get("/openapi.json", [&](auto* res, auto* /*req*/) {
                res->writeHeader("Content-Type", "application/json")
                    ->end(protocol::build_openapi_spec(model_name));
            })
            .get("/docs", [](auto* res, auto* /*req*/) {
                res->writeHeader("Content-Type", "text/html; charset=utf-8")->end(kDocsHtml);
            })
            .get("/docs/:file", [](auto* res, auto* req) {
                const std::string file(req->getParameter(0));
                const auto& assets = swagger_ui_assets();
                const auto asset = assets.find(file);
                if (asset == assets.end()) {
                    res->writeStatus("404 Not Found")->end("not found");
                    return;
                }
                const auto contents = read_file(std::filesystem::path(LFM_SWAGGER_UI_DIR) / file);
                if (!contents) {
                    res->writeStatus("404 Not Found")->end("not found");
                    return;
                }
                res->writeHeader("Content-Type", asset->second)->end(*contents);
            })
            .post("/v1/chat/completions", [&](auto* res, auto* /*req*/) {
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

                res->onData([res, state, &dispatcher, &service, &tokenizer, &model_name,
                             eos_token_id, loop](std::string_view chunk, bool last) {
                    state->body.append(chunk);
                    if (!last) return;

                    protocol::ChatCompletionRequest request;
                    GenerateRequest generate_request;
                    try {
                        request = protocol::from_json<protocol::ChatCompletionRequest>(state->body);
                        generate_request = protocol::to_generate_request(request, tokenizer, eos_token_id);
                    } catch (const std::exception& error) {
                        res->writeStatus("400 Bad Request")
                            ->writeHeader("Content-Type", "application/json")
                            ->end(std::string("{\"error\":\"") + error.what() + "\"}");
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
                                              &tokenizer, &model_name, loop](const GenerateEvent& event) {
                            completion->insert(completion->end(), event.tokens.begin(), event.tokens.end());
                            if (!event.finished) return;
                            loop->defer([res, state, completion, id_str, created, prompt_tokens,
                                         &tokenizer, &model_name, reason = event.finish_reason] {
                                if (state->aborted.load()) return;
                                const auto response = protocol::to_chat_completion_response(
                                    id_str, model_name, created, prompt_tokens, *completion, reason, tokenizer);
                                res->writeHeader("Content-Type", "application/json")
                                    ->end(protocol::to_json(response));
                            });
                        });
                    } else {
                        res->writeHeader("Content-Type", "text/event-stream")
                            ->writeHeader("Cache-Control", "no-cache");
                        auto first = std::make_shared<bool>(true);
                        dispatcher.watch(id, [res, state, first, id_str, created, &tokenizer, &model_name,
                                              loop](const GenerateEvent& event) {
                            loop->defer([res, state, first, id_str, created, &tokenizer, &model_name,
                                         tokens = event.tokens, finished = event.finished,
                                         reason = event.finish_reason] {
                                if (state->aborted.load()) return;
                                if (!tokens.empty() || *first) {
                                    const auto chunk = protocol::to_chat_completion_chunk(
                                        id_str, model_name, created, tokens, *first, std::nullopt, tokenizer);
                                    *first = false;
                                    res->write("data: " + protocol::to_json(chunk) + "\n\n");
                                }
                                if (finished) {
                                    const auto final_chunk = protocol::to_chat_completion_chunk(
                                        id_str, model_name, created, {}, false, reason, tokenizer);
                                    res->write("data: " + protocol::to_json(final_chunk) + "\n\n");
                                    res->write(std::string_view("data: [DONE]\n\n"));
                                    res->end();
                                }
                            });
                        });
                    }
                });
            })
            .listen(args.port, [&](auto* listen_socket) {
                if (listen_socket) {
                    std::cout << "lfm25-serve listening on http://127.0.0.1:" << args.port
                              << " (model=" << model_name << ")" << std::endl;
                } else {
                    std::cerr << "failed to listen on port " << args.port << std::endl;
                }
            })
            .run();

        dispatcher.stop();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "lfm25-serve: " << error.what() << std::endl;
        return 1;
    }
}
