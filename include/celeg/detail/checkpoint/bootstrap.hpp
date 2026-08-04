#pragma once

#include "celeg/model/architecture.hpp"
#include "celeg/runtime/context.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace celeg::detail {

struct ModelBootstrap {
    CheckpointView checkpoint;
    ResolvedModel model;
};

ModelBootstrap load_model_bootstrap(const std::filesystem::path& model_path);
ModelBootstrap load_model_bootstrap(const std::filesystem::path& model_path,
                                    const RuntimeContext& runtime);

} // namespace celeg::detail
