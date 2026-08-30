#pragma once

#include "celeg/model/runtime_types.hpp"

#include <memory>
#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace celeg {

struct VisualEmbedding {
    int width = 0;
    std::vector<float> values;
    std::vector<std::array<int32_t, 3>> rope_positions;

    std::size_t token_count() const {
        if (width <= 0 || values.size() % static_cast<std::size_t>(width) != 0) {
            return 0;
        }
        return values.size() / static_cast<std::size_t>(width);
    }
};

class IVisualEmbeddingProvider {
public:
    virtual ~IVisualEmbeddingProvider() = default;
    virtual VisualEmbedding encode(std::string_view data_url) const = 0;
};

using VisualEmbeddingProvider = std::shared_ptr<const IVisualEmbeddingProvider>;

}
