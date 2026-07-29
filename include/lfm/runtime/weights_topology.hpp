#pragma once

#include "lfm/model/config/shape.hpp"

#include <cstdint>
#include <string_view>
#include <vector>

namespace lfm::weights {

enum class TensorRole : std::uint8_t { TokenEmbedding, LanguageModelHead, FinalNorm };

struct TensorSpec {
    TensorRole role;
    std::string_view canonical_name;
    std::vector<std::string_view> alternatives;
    std::vector<std::int64_t> shape;
    bool optional = false;
};

inline TensorSpec token_embedding(const ModelShape& shape) {
    return {TensorRole::TokenEmbedding, "model.embed_tokens.weight",
            {"model.embedding.weight"}, {shape.vocab_size, shape.hidden}, false};
}

inline TensorSpec language_model_head(const ModelShape& shape, bool tied) {
    return {TensorRole::LanguageModelHead, "model.lm_head.weight", {},
            {shape.vocab_size, shape.hidden}, tied};
}

inline TensorSpec final_norm(const ModelShape& shape) {
    return {TensorRole::FinalNorm, "model.embedding_norm.weight",
            {"model.norm.weight", "model.final_norm.weight"}, {shape.hidden}, false};
}

} // namespace lfm::weights
