#pragma once

#include "celeg/serve/types.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <stdexcept>

namespace celeg::serve {

inline void materialize_visual_prompt(GenerateRequest& request,
                                      const VisualEmbeddingProvider& provider) {
    if (request.images.empty()) return;
    if (!provider) throw std::invalid_argument("visual input provider is not configured");
    if (request.image_token_id < 0) {
        throw std::invalid_argument("visual request has no image token id");
    }

    PromptEmbedding& prompt_embedding = request.prompt_embedding;
    std::size_t search_from = 0;
    for (const MultimodalImage& image : request.images) {
        const VisualEmbedding embedding = provider->encode(image.data_url);
        if (embedding.width <= 0 || embedding.token_count() == 0) {
            throw std::invalid_argument("visual provider returned an empty embedding");
        }
        if (prompt_embedding.width == 0) prompt_embedding.width = embedding.width;
        if (prompt_embedding.width != embedding.width) {
            throw std::invalid_argument("visual embeddings have different widths");
        }

        const auto marker = std::find(request.prompt_tokens.begin() +
                                          static_cast<std::ptrdiff_t>(search_from),
                                      request.prompt_tokens.end(),
                                      request.image_token_id);
        if (marker == request.prompt_tokens.end()) {
            throw std::invalid_argument("visual input has no matching image marker");
        }
        const std::size_t position = static_cast<std::size_t>(
            std::distance(request.prompt_tokens.begin(), marker));
        const std::size_t count = embedding.token_count();
        request.prompt_tokens.insert(request.prompt_tokens.begin() +
                                         static_cast<std::ptrdiff_t>(position + 1),
                                     count - 1, request.image_token_id);
        for (std::size_t index = 0; index < count; ++index) {
            prompt_embedding.positions.push_back(position + index);
        }
        prompt_embedding.values.insert(prompt_embedding.values.end(),
                                       embedding.values.begin(), embedding.values.end());
        search_from = position + count;
    }
    request.images.clear();
}

} // namespace celeg::serve
