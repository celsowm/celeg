#pragma once

#include "celeg/model/visual_embeddings.hpp"

#include <filesystem>

namespace celeg {

VisualEmbeddingProvider make_gemma4_visual_embedding_provider(
    const std::filesystem::path& projector_path);

} // namespace celeg
