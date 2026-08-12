#include "celeg/backend/cuda/weight_setup.hpp"

#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/detail/checkpoint/bootstrap.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/backend/cuda/weight_setup_support.hpp"
#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "celeg/runtime/weights_topology.hpp"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <cstdlib>
#include <string_view>

namespace celeg {
namespace {

std::string tensor_name(std::span<const TensorRequest> requests, TensorRole role,
                        int layer = -1) {
    return cuda_tensor_name(requests, role, layer);
}

} // namespace

void CudaWeightSetup::load(CudaCompiledModel& model,
                           const std::string& model_path,
                           const detail::ModelBootstrap& bootstrap,
                           LayerLoader load_layers) {
    const char* managed = std::getenv("CELEG_CUDA_MANAGED_WEIGHTS");
    CudaManagedWeightAllocationScope managed_scope(
        managed && std::string_view(managed) == "1");
    model.resources_.weights_ = WeightLoader::acquire(
        model_path, model.resources_.options_.weight_mode,
        model.resources_.options_.expert_offload.fingerprint() +
        "|mtp=" + (model.resources_.options_.enable_mtp ? "1" : "0") +
        "|mtp_steps=" +
            std::to_string(model.resources_.options_.mtp_speculative_tokens));
    model.resources_.weight_loader_ = std::make_unique<WeightLoader>(
        model.resources_.weights_, model.resources_.options_.weight_mode);

    // Keep the shared checkpoint locked while global resources and all layer
    // views are assembled. Other sessions can then reuse an immutable,
    // completely initialized weight set.
    std::unique_lock<std::mutex> shared_weights_lock(model.resources_.weights_->mutex);
    model.resources_.weights_->repo = bootstrap.checkpoint.repository;
    const IWeightRepository& repo = *model.resources_.weights_->repo;
    model.resources_.embedding_ = model.resources_.weight_loader_->load_linear_weight(
        repo, tensor_name(model.resources_.model_.weight_plan.requests,
                          TensorRole::TokenEmbedding),
        {model.resources_.dims_.vocab_size, model.resources_.shape_.hidden});
    const NormSpec& final_norm = model.resources_.program_.final_norm;
    const std::string final_name = final_norm.weightless()
        ? std::string{} : tensor_name(model.resources_.model_.weight_plan.requests,
                                      TensorRole::FinalNorm);
    model.resources_.final_norm_ = model.resources_.weight_loader_->load_rms_norm_weight(
        repo, final_name, {model.resources_.shape_.hidden}, final_norm.weight_kind);
    if (model.resources_.program_.embedding_transform.post_norm) {
        const NormSpec& embedding_norm =
            *model.resources_.program_.embedding_transform.post_norm;
        const std::string embedding_name = embedding_norm.weightless()
            ? std::string{} : final_name;
        model.resources_.embedding_norm_ =
            model.resources_.weight_loader_->load_rms_norm_weight(
                repo, embedding_name, {model.resources_.shape_.hidden},
                embedding_norm.weight_kind);
    }

    if (model.resources_.program_.per_layer_input.enabled) {
        const int input_size = model.resources_.program_.per_layer_input.input_size;
        CudaPerLayerInputResources per_layer;
        per_layer.plan = model.resources_.program_.per_layer_input;
        per_layer.embedding = model.resources_.weight_loader_->load_linear_weight(
            repo, tensor_name(model.resources_.model_.weight_plan.requests,
                              TensorRole::PerLayerEmbedding),
            {model.resources_.dims_.vocab_size,
             static_cast<int>(per_layer.plan.packed_width)});
        per_layer.context_projection = model.resources_.weight_loader_->load_linear_weight(
            repo, tensor_name(model.resources_.model_.weight_plan.requests,
                              TensorRole::PerLayerContextProjection),
            {static_cast<int>(per_layer.plan.packed_width), model.resources_.shape_.hidden});
        per_layer.projection_norm = model.resources_.weight_loader_->load_weight(
            repo, tensor_name(model.resources_.model_.weight_plan.requests,
                              TensorRole::PerLayerProjectionNorm),
            {input_size});
        per_layer.embedding_layout = make_cuda_embedding_layout(
            model.resources_.options_.weight_mode, *per_layer.embedding,
            "per-layer embedding");
        per_layer.validate();
        model.resources_.per_layer_input_ = std::move(per_layer);
    }

    model.resources_.weight_layout_ = make_cuda_embedding_layout(
        model.resources_.options_.weight_mode, *model.resources_.embedding_, "embedding");
    const std::string lm_head_name = tensor_name(
        model.resources_.model_.weight_plan.requests,
        TensorRole::LanguageModelHead);
    if (!model.resources_.model_.capabilities.tied_embeddings &&
        repo.contains(lm_head_name)) {
        model.resources_.lm_head_ = model.resources_.weight_loader_->load_linear_weight(
            repo, lm_head_name,
        {model.resources_.dims_.vocab_size, model.resources_.shape_.hidden});
    }

    load_layers(repo);
}

} // namespace celeg
