#pragma once

#include "lfm/model/config/config.hpp"
#include "lfm/checkpoint/formats/gguf.hpp"
#include "lfm/model/config/shape.hpp"
#include "lfm/model/config/variant.hpp"
#include "lfm/model/architecture.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace lfm::detail {

struct ModelBootstrap {
    bool is_gguf = false;
    ModelConfig config;
    ModelShape shape;
    const IModelVariant* variant = nullptr;
    const IArchitectureProvider* architecture_provider = nullptr;
    std::shared_ptr<GgufFile> gguf_file;
};

bool path_is_gguf(const std::filesystem::path& model_path);
std::filesystem::path config_path_for(const std::filesystem::path& model_path);
ModelBootstrap load_model_bootstrap(const std::filesystem::path& model_path);

} // namespace lfm::detail
