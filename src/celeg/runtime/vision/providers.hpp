#pragma once

#include "celeg/model/visual_embeddings.hpp"

#include <array>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

namespace celeg {

class IVisionProviderFactory;
class IWeightRepository;

struct ImageResizeSpec {
    int width = 0;
    int height = 0;
    bool preserve_aspect = false;
};

struct ImageNormalizationSpec {
    std::array<float, 3> mean{0.0f, 0.0f, 0.0f};
    std::array<float, 3> standard_deviation{1.0f, 1.0f, 1.0f};
};

struct VisionPipelineSpec {
    ImageResizeSpec resize;
    ImageNormalizationSpec normalization;
    int patch_size = 0;
    int embedding_width = 0;
    std::string projection_layout;
    std::string source;
};

VisualEmbeddingProvider make_gguf_visual_embedding_provider(
    const std::filesystem::path& projector_path);

VisualEmbeddingProvider make_safetensor_visual_embedding_provider(
    std::shared_ptr<const IWeightRepository> repository);
VisualEmbeddingProvider make_safetensor_visual_embedding_provider(
    const std::filesystem::path& model_path);

VisualEmbeddingProvider make_patch_visual_embedding_provider(
    std::shared_ptr<const IWeightRepository> repository);
VisualEmbeddingProvider make_patch_visual_embedding_provider(
    const std::filesystem::path& model_path);

std::unique_ptr<IVisionProviderFactory> make_gguf_vision_provider_factory();
std::unique_ptr<IVisionProviderFactory> make_safetensor_vision_provider_factory();
std::unique_ptr<IVisionProviderFactory> make_patch_vision_provider_factory();

}
