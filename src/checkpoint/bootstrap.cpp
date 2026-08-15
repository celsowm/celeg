#include "celeg/detail/checkpoint/bootstrap.hpp"

#include "celeg/checkpoint/catalog.hpp"

#include <stdexcept>

namespace celeg::detail {

ModelBootstrap load_model_bootstrap(const std::filesystem::path& model_path) {
    return load_model_bootstrap(model_path, *create_builtin_runtime_context());
}

ModelBootstrap load_model_bootstrap(const std::filesystem::path& model_path,
                                   const RuntimeContext& runtime) {
    ModelBootstrap bootstrap;
    bootstrap.checkpoint = runtime.checkpoint_formats().open(model_path);
    const IArchitecture& architecture =
        runtime.architectures().select(bootstrap.checkpoint.metadata);
    bootstrap.model = architecture.resolve(bootstrap.checkpoint);
    return bootstrap;
}

}
