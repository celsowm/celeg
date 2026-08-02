#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/detail/checkpoint/bootstrap.hpp"

#include <filesystem>
#include <stdexcept>
namespace celeg {

CudaCompiledModel::CudaCompiledModel(const std::string& model_path,
                   int max_context,
                   CudaModelOptions options,
                   GenerationConfig generation)
    : resources_(CudaExecutionPlan::compile(options, max_context)),
      session_(generation),
      stream_(),
      max_context_(max_context) {
    session_.generation_.validate();
    if (max_context_ <= 0) {
        throw std::invalid_argument("max_context must be positive");
    }
    const detail::ModelBootstrap bootstrap =
        detail::load_model_bootstrap(std::filesystem::path(model_path));
    configure_model(bootstrap);
    allocate_celeg_resources();
    load_checkpoint_weights(model_path, bootstrap);
    if (resources_.options_.cuda_graph ||
        resources_.options_.gemm_backend == GemmBackend::CublasLt) {
        warmup_decode_gemms();
    }
    if (resources_.options_.fast_attention) {
        warmup_prefill_attention_gemm();
    }
    local_kv_cache_available_ = resources_.options_.allocate_local_kv_cache;
    reset(resources_.options_.allocate_local_kv_cache);
}

} // namespace celeg

