#include "lfm/serve/protocol/openapi.hpp"

#include <cstddef>
#include <string_view>

namespace lfm::serve::protocol {

namespace {

// Static OpenAPI 3.1 document. __MODEL_NAME__ is substituted at request time
// so the example payloads reflect the model this instance actually serves.
constexpr std::string_view kSpecTemplate = R"JSON({
  "openapi": "3.1.0",
  "info": {
    "title": "lfm25-serve",
    "description": "OpenAI-compatible chat completions API served by lfm25-serve.",
    "version": "0.0.22"
  },
  "servers": [{"url": "/"}],
  "paths": {
    "/healthz": {
      "get": {
        "summary": "Liveness check",
        "operationId": "healthz",
        "responses": {
          "200": {
            "description": "Server is up",
            "content": {"text/plain": {"schema": {"type": "string", "example": "ok"}}}
          }
        }
      }
    },
    "/v1/models": {
      "get": {
        "summary": "List available models",
        "operationId": "listModels",
        "responses": {
          "200": {
            "description": "The served model",
            "content": {
              "application/json": {
                "schema": {"$ref": "#/components/schemas/ModelList"},
                "example": {
                  "object": "list",
                  "data": [{"id": "__MODEL_NAME__", "object": "model"}]
                }
              }
            }
          }
        }
      }
    },
    "/v1/chat/completions": {
      "post": {
        "summary": "Create a chat completion",
        "description": "OpenAI-compatible chat completions endpoint. Set \"stream\": true for a Server-Sent Events response instead of a single JSON object.",
        "operationId": "createChatCompletion",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {"$ref": "#/components/schemas/ChatCompletionRequest"},
              "example": {
                "model": "__MODEL_NAME__",
                "messages": [{"role": "user", "content": "Hello!"}]
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Chat completion result. When \"stream\" is true this is instead a text/event-stream of ChatCompletionChunk objects terminated by \"data: [DONE]\".",
            "content": {
              "application/json": {"schema": {"$ref": "#/components/schemas/ChatCompletionResponse"}},
              "text/event-stream": {"schema": {"$ref": "#/components/schemas/ChatCompletionChunk"}}
            }
          },
          "400": {
            "description": "Malformed request",
            "content": {
              "application/json": {"schema": {"$ref": "#/components/schemas/ErrorResponse"}}
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "ChatMessage": {
        "type": "object",
        "required": ["role", "content"],
        "properties": {
          "role": {"type": "string", "enum": ["system", "developer", "user", "assistant", "tool"]},
          "content": {"type": "string"}
        }
      },
      "ChatCompletionRequest": {
        "type": "object",
        "required": ["model", "messages"],
        "properties": {
          "model": {"type": "string"},
          "messages": {"type": "array", "items": {"$ref": "#/components/schemas/ChatMessage"}},
          "temperature": {"type": "number"},
          "top_p": {"type": "number"},
          "top_k": {"type": "integer"},
          "max_tokens": {"type": "integer"},
          "stream": {"type": "boolean"},
          "seed": {"type": "integer"}
        }
      },
      "ChatCompletionResponseMessage": {
        "type": "object",
        "properties": {
          "role": {"type": "string", "example": "assistant"},
          "content": {"type": "string"}
        }
      },
      "ChatCompletionChoice": {
        "type": "object",
        "properties": {
          "index": {"type": "integer"},
          "message": {"$ref": "#/components/schemas/ChatCompletionResponseMessage"},
          "finish_reason": {"type": "string", "enum": ["", "stop", "length", "cancelled", "error"]}
        }
      },
      "Usage": {
        "type": "object",
        "properties": {
          "prompt_tokens": {"type": "integer"},
          "completion_tokens": {"type": "integer"},
          "total_tokens": {"type": "integer"}
        }
      },
      "ChatCompletionResponse": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "object": {"type": "string", "example": "chat.completion"},
          "created": {"type": "integer"},
          "model": {"type": "string"},
          "choices": {"type": "array", "items": {"$ref": "#/components/schemas/ChatCompletionChoice"}},
          "usage": {"$ref": "#/components/schemas/Usage"}
        }
      },
      "ChatCompletionChunkDelta": {
        "type": "object",
        "properties": {
          "role": {"type": "string"},
          "content": {"type": "string"}
        }
      },
      "ChatCompletionChunkChoice": {
        "type": "object",
        "properties": {
          "index": {"type": "integer"},
          "delta": {"$ref": "#/components/schemas/ChatCompletionChunkDelta"},
          "finish_reason": {"type": "string"}
        }
      },
      "ChatCompletionChunk": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "object": {"type": "string", "example": "chat.completion.chunk"},
          "created": {"type": "integer"},
          "model": {"type": "string"},
          "choices": {"type": "array", "items": {"$ref": "#/components/schemas/ChatCompletionChunkChoice"}}
        }
      },
      "ModelList": {
        "type": "object",
        "properties": {
          "object": {"type": "string", "example": "list"},
          "data": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "id": {"type": "string"},
                "object": {"type": "string", "example": "model"}
              }
            }
          }
        }
      },
      "ErrorResponse": {
        "type": "object",
        "properties": {
          "error": {"type": "string"}
        }
      }
    }
  }
})JSON";

} // namespace

std::string build_openapi_spec(const std::string& model_name) {
    std::string spec(kSpecTemplate);
    std::size_t pos = 0;
    while ((pos = spec.find("__MODEL_NAME__", pos)) != std::string::npos) {
        spec.replace(pos, std::string("__MODEL_NAME__").size(), model_name);
        pos += model_name.size();
    }
    return spec;
}

} // namespace lfm::serve::protocol
