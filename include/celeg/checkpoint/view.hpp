#pragma once

#include "celeg/checkpoint/metadata.hpp"
#include "celeg/checkpoint/tokenizer.hpp"

#include <filesystem>
#include <memory>

namespace celeg {

class IWeightRepository;

// Fully opened, format-normalized checkpoint. Format-specific readers retain
// ownership of their repository here; consumers only see normalized metadata
// and the common tensor repository boundary. Behavior that genuinely differs
// by concrete format must be expressed as a capability query on `repository`
// (see `require_locatable_tensor_repository` / `require_random_access_tensor_reader`
// in celeg/checkpoint/weight_repository.hpp), not a format flag here.
struct CheckpointView {
    CheckpointMetadata metadata;
    std::shared_ptr<IWeightRepository> repository;
    std::filesystem::path path;
    std::shared_ptr<const TokenizerData> tokenizer;
};

} // namespace celeg
