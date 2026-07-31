#pragma once

#include "celeg/model/architecture.hpp"
#include "celeg/checkpoint/formats/gguf.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace celeg::detail {

struct ModelBootstrap {
    CheckpointView checkpoint;
    ResolvedModel model;
    std::shared_ptr<GgufFile> gguf_file;

    bool is_gguf() const { return model.is_gguf; }
};

bool path_is_gguf(const std::filesystem::path& model_path);
std::filesystem::path config_path_for(const std::filesystem::path& model_path);
ModelBootstrap load_model_bootstrap(const std::filesystem::path& model_path);

} // namespace celeg::detail
