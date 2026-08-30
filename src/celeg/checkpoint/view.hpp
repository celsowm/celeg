#pragma once

#include "celeg/checkpoint/metadata.hpp"
#include "celeg/checkpoint/tokenizer.hpp"

#include <filesystem>
#include <memory>

namespace celeg {

class IWeightRepository;

struct CheckpointView {
    CheckpointMetadata metadata;
    std::shared_ptr<IWeightRepository> repository;
    std::filesystem::path path;
    std::shared_ptr<const TokenizerData> tokenizer;
};

}
