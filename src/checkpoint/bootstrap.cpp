#include "celeg/detail/checkpoint/bootstrap.hpp"

#include "celeg/checkpoint/catalog.hpp"

#include <stdexcept>

namespace celeg::detail {

ModelBootstrap load_model_bootstrap(const std::filesystem::path& model_path) {
    ModelBootstrap bootstrap;
    static const std::shared_ptr<const CheckpointFormatCatalog> formats =
        create_builtin_checkpoint_format_catalog();
    bootstrap.checkpoint = formats->open(model_path);
    static const std::shared_ptr<const ArchitectureCatalog> catalog =
        create_builtin_architecture_catalog();
    const IArchitecture& architecture = catalog->select(bootstrap.checkpoint.metadata);
    bootstrap.model = architecture.resolve(bootstrap.checkpoint);
    return bootstrap;
}

} // namespace celeg::detail
