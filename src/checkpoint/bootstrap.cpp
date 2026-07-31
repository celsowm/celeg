#include "celeg/detail/checkpoint/bootstrap.hpp"

#include "celeg/checkpoint/formats/json.hpp"
#include "celeg/checkpoint/repositories/gguf.hpp"
#include "celeg/checkpoint/repositories/safetensors.hpp"

#include <stdexcept>

namespace celeg::detail {

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
    ModelBootstrap bootstrap;
    bootstrap.checkpoint.path = model_path;
    const bool is_gguf = path_is_gguf(model_path);
    if (is_gguf) {
        bootstrap.gguf_file = std::make_shared<GgufFile>(model_path.string());
        bootstrap.checkpoint.metadata = CheckpointMetadata::from_gguf(*bootstrap.gguf_file);
        bootstrap.checkpoint.repository = std::make_shared<GgufRepository>(bootstrap.gguf_file);
        bootstrap.checkpoint.gguf = bootstrap.gguf_file;
    } else {
        const std::filesystem::path config_path = config_path_for(model_path);
        if (!std::filesystem::exists(config_path)) {
            throw std::runtime_error(
                "config.json not found alongside checkpoint: " + config_path.string());
        }
        const Json root = Json::parse_file(config_path.string());
        bootstrap.checkpoint.metadata = CheckpointMetadata::from_json(root);
        bootstrap.checkpoint.repository = std::make_shared<SafeTensorRepository>(model_path);
    }
    static const std::shared_ptr<const ArchitectureCatalog> catalog =
        create_builtin_architecture_catalog();
    const IArchitecture& architecture = catalog->select(bootstrap.checkpoint.metadata);
    bootstrap.model = architecture.resolve(bootstrap.checkpoint);
    return bootstrap;
}

} // namespace celeg::detail
