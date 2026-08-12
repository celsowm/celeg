#pragma once

#include "celeg/model/resolved.hpp"

#include <cstdint>
#include <string_view>
#include <vector>

namespace celeg::weights {

enum class TensorRole : std::uint8_t { TokenEmbedding, LanguageModelHead, FinalNorm };

struct TensorSpec {
    TensorRole role;
    std::string_view canonical_name;
    std::vector<std::string_view> alternatives;
    std::vector<std::int64_t> shape;
    bool optional = false;
};

inline TensorSpec token_embedding(const CheckpointDimensions& dims,
                                  const ExecutionTopology& shape) {
    return {TensorRole::TokenEmbedding, "model.embed_tokens.weight",
            {"model.embedding.weight"}, {dims.vocab_size, shape.hidden}, false};
}

inline TensorSpec language_model_head(const CheckpointDimensions& dims,
                                     const ExecutionTopology& shape, bool tied) {
    return {TensorRole::LanguageModelHead, "model.lm_head.weight", {},
            {dims.vocab_size, shape.hidden}, tied};
}

inline TensorSpec final_norm(const ExecutionTopology& shape) {
    return {TensorRole::FinalNorm, "model.embedding_norm.weight",
            {"model.norm.weight", "model.final_norm.weight"}, {shape.hidden}, false};
}

} // namespace celeg::weights
