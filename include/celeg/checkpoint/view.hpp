#pragma once

#include "celeg/checkpoint/metadata.hpp"

#include <filesystem>
#include <memory>

namespace celeg {

class GgufFile;
class IWeightRepository;

// Fully opened, format-normalized checkpoint. Format-specific readers retain
// ownership of their repository and optional GGUF mapping here; consumers only
// see normalized metadata and the common tensor repository boundary.
struct CheckpointView {
    CheckpointMetadata metadata;
    std::shared_ptr<IWeightRepository> repository;
    std::shared_ptr<GgufFile> gguf;
    std::filesystem::path path;
};

} // namespace celeg
