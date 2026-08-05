#pragma once

#include "celeg/model/visual_embeddings.hpp"

#include <filesystem>
#include <memory>

namespace celeg {

class IVisionProviderFactory;

VisualEmbeddingProvider make_gemma4_visual_embedding_provider(
    const std::filesystem::path& projector_path);

std::unique_ptr<IVisionProviderFactory> make_gemma4_vision_provider_factory();

} // namespace celeg
