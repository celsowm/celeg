#include "lfm/detail/checkpoint/bootstrap.hpp"

#include <stdexcept>

namespace lfm::detail {

bool path_is_gguf(const std::filesystem::path& model_path) {
    if (std::filesystem::is_directory(model_path)) return false;
    return model_path.extension() == ".gguf";
}

std::filesystem::path config_path_for(const std::filesystem::path& model_path) {
    if (std::filesystem::is_directory(model_path)) {
        return model_path / "config.json";
    }
    return model_path.parent_path() / "config.json";
}

ModelBootstrap load_model_bootstrap(const std::filesystem::path& model_path) {
    register_builtin_variants();
    register_builtin_architecture_providers();

    ModelBootstrap bootstrap;
    bootstrap.is_gguf = path_is_gguf(model_path);
    if (bootstrap.is_gguf) {
        bootstrap.gguf_file = std::make_shared<GgufFile>(model_path.string());
        bootstrap.config = ModelConfig::from_gguf(*bootstrap.gguf_file);
    } else {
        const std::filesystem::path config_path = config_path_for(model_path);
        if (!std::filesystem::exists(config_path)) {
            throw std::runtime_error(
                "config.json not found alongside checkpoint: " + config_path.string());
        }
        bootstrap.config = ModelConfig::load(config_path.string());
    }
    bootstrap.shape = ModelShape::from_config(bootstrap.config);
    bootstrap.architecture_provider = &ArchitectureRegistry::instance().select(
        bootstrap.config);
    bootstrap.variant = &ModelVariantRegistry::instance().select(
        bootstrap.shape, bootstrap.config.repo_hint);
    bootstrap.shape = bootstrap.variant->resolve_shape(bootstrap.shape);
    return bootstrap;
}

} // namespace lfm::detail
