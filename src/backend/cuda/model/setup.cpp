#include "celeg/detail/model/impl.hpp"
#include "celeg/detail/checkpoint/bootstrap.hpp"

#include <filesystem>
#include <stdexcept>
namespace celeg {

Model::Impl::Impl(const std::string& model_path,
                   int max_context,
                   ModelOptions options,
                   GenerationConfig generation)
    : ModelResources(ExecutionPlan::compile(options, max_context)),
      SessionState(generation),
      stream_(),
      max_context_(max_context) {
    generation_.validate();
    if (max_context_ <= 0) {
        throw std::invalid_argument("max_context must be positive");
    }
    const detail::ModelBootstrap bootstrap =
        detail::load_model_bootstrap(std::filesystem::path(model_path));
    configure_model(bootstrap);
    allocate_celeg_resources();
    load_checkpoint_weights(model_path, bootstrap);
    if (options_.cuda_graph ||
        options_.gemm_backend == GemmBackend::CublasLt) {
        warmup_decode_gemms();
    }
    if (options_.fast_attention) {
        warmup_prefill_attention_gemm();
    }
    local_kv_cache_available_ = options_.allocate_local_kv_cache;
    reset(options_.allocate_local_kv_cache);
}

} // namespace celeg

