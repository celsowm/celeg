#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/backend/cuda/kernels/mmq.hpp"

#include <filesystem>
#include <stdexcept>
namespace celeg {

CudaCompiledModel::CudaCompiledModel(const std::string& model_path,
                   int max_context,
                   CudaModelOptions options,
                   GenerationConfig generation,
                   std::shared_ptr<const RuntimeContext> runtime)
    : resources_(CudaExecutionPlan::compile(
          options, max_context, discover_cuda_device_capabilities())),
      runtime_(runtime ? std::move(runtime) : create_builtin_runtime_context()),
      session_(generation),
      stream_(),
      max_context_(max_context) {
    session_.generation_.validate();
    if (max_context_ <= 0) {
        throw std::invalid_argument("max_context must be positive");
    }
    const detail::ModelBootstrap bootstrap = detail::load_model_bootstrap(
        std::filesystem::path(model_path), *runtime_);
    configure_model(bootstrap);
    allocate_celeg_resources();
    workspace_.residency_workspace_ = ExpertResidencyWorkspace{
        &workspace_.router_done_event_, &workspace_.ffn_done_event_,
        &workspace_.promote_done_event_, &workspace_.prefetch_done_event_,
        &workspace_.cold_expert_host_, &workspace_.cold_scores_host_,
        &workspace_.active_expert_host_,
        &workspace_.loaded_leases_, &workspace_.io_futures_,
        &workspace_.prefetch_idx_, &workspace_.prefetch_ranked_,
        &workspace_.prefetch_scores_};
    workspace_.cold_expert_host_.reserve(
        static_cast<size_t>(resources_.shape_.num_experts));
    workspace_.cold_scores_host_.reserve(
        static_cast<size_t>(resources_.shape_.num_experts));
    workspace_.active_expert_host_.reserve(
        static_cast<size_t>(resources_.shape_.num_experts));
    workspace_.loaded_leases_.reserve(
        static_cast<size_t>(resources_.shape_.num_experts));
    workspace_.io_futures_.reserve(
        static_cast<size_t>(resources_.shape_.num_experts));
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

