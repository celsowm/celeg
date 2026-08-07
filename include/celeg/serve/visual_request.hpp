#pragma once

#include "celeg/serve/types.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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

    const std::vector<int32_t> source_tokens = request.prompt_tokens;
    std::vector<int32_t> materialized_tokens;
    materialized_tokens.reserve(source_tokens.size());
    std::vector<std::array<int32_t, 3>> rope_positions;
    rope_positions.reserve(source_tokens.size());
    PromptEmbedding prompt_embedding;
    std::size_t image_index = 0;
    int32_t logical_position = 0;

    for (const int32_t token : source_tokens) {
        if (token != request.image_token_id) {
            materialized_tokens.push_back(token);
            rope_positions.push_back({logical_position, logical_position, logical_position});
            ++logical_position;
            continue;
        }
        if (image_index >= request.images.size()) {
            throw std::invalid_argument("visual prompt has more image markers than images");
        }
        const VisualEmbedding embedding = provider->encode(
            request.images[image_index++].data_url);
        if (embedding.width <= 0 || embedding.token_count() == 0) {
            throw std::invalid_argument("visual provider returned an empty embedding");
        }
        if (embedding.rope_positions.empty() && embedding.token_count() > 1) {
            // Providers without M-RoPE metadata retain the old scalar behavior.
            // The local positions also define the logical span of the image.
        } else if (!embedding.rope_positions.empty() &&
                   embedding.rope_positions.size() != embedding.token_count()) {
            throw std::invalid_argument("visual provider returned invalid M-RoPE positions");
        }
        if (prompt_embedding.width == 0) prompt_embedding.width = embedding.width;
        if (prompt_embedding.width != embedding.width) {
            throw std::invalid_argument("visual embeddings have different widths");
        }

        const std::size_t count = embedding.token_count();
        const std::size_t physical_position = materialized_tokens.size();
        for (std::size_t index = 0; index < count; ++index) {
            materialized_tokens.push_back(request.image_token_id);
            prompt_embedding.positions.push_back(physical_position + index);
            const float* row = embedding.values.data() + index *
                static_cast<std::size_t>(embedding.width);
            prompt_embedding.values.insert(prompt_embedding.values.end(), row,
                                           row + embedding.width);
            std::array<int32_t, 3> local = embedding.rope_positions.empty()
                ? std::array<int32_t, 3>{static_cast<int32_t>(index),
                                         static_cast<int32_t>(index),
                                         static_cast<int32_t>(index)}
                : embedding.rope_positions[index];
            rope_positions.push_back({local[0] + logical_position,
                                      local[1] + logical_position,
                                      local[2] + logical_position});
        }
        int32_t logical_span = static_cast<int32_t>(count);
        if (!embedding.rope_positions.empty()) {
            logical_span = 0;
            for (const auto& position : embedding.rope_positions) {
                logical_span = std::max(logical_span,
                    std::max(position[0], std::max(position[1], position[2])) + 1);
            }
        }
        logical_position += logical_span;
    }

    if (image_index != request.images.size()) {
        throw std::invalid_argument("visual input has fewer image markers than images");
    }
    request.prompt_tokens = std::move(materialized_tokens);
    prompt_embedding.rope_positions = std::move(rope_positions);
    prompt_embedding.next_rope_position = {logical_position, logical_position, logical_position};
    prompt_embedding.has_rope_positions = true;
    request.prompt_embedding = std::move(prompt_embedding);
    request.images.clear();
}

} // namespace celeg::serve
