#pragma once

#include "celeg/model/visual_embeddings.hpp"

#include <filesystem>
#include <memory>

namespace celeg {

class IVisionProviderFactory;
class IWeightRepository;

VisualEmbeddingProvider make_qwen35_visual_embedding_provider(
    std::shared_ptr<const IWeightRepository> repository);

VisualEmbeddingProvider make_qwen35_visual_embedding_provider(
    const std::filesystem::path& model_path);
std::unique_ptr<IVisionProviderFactory> make_qwen35_vision_provider_factory();

} // namespace celeg
