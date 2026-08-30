#include "backend/cuda/weight_setup.hpp"

#include "detail/compiled_model.hpp"
#include "checkpoint/detail/bootstrap.hpp"
#include "backend/cuda/weight_layout.hpp"
#include "backend/cuda/weight_setup_support.hpp"
#include "backend/cuda/weights_loader.hpp"
#include "celeg/model/weights/quantization.hpp"

#include <memory>
#include <mutex>
#include <stdexcept>

namespace celeg {
namespace {

std::string tensor_name(std::span<const TensorRequest> requests, TensorRole role,
                        int layer = -1) {
    return cuda_tensor_name(requests, role, layer);
}

}

void CudaWeightSetup::load(CudaCompiledModel& model,
                           const std::string& model_path,
                           const detail::ModelBootstrap& bootstrap,
                           LayerLoader load_layers) {
    model.resources_.weights_ = model.weight_cache_.acquire(
        model_path, model.resources_.options().weight_mode,
        model.resources_.options().expert_offload.fingerprint() +
        "|mtp=" + (model.resources_.options().enable_mtp ? "1" : "0") +
        "|mtp_steps=" +
            std::to_string(model.resources_.options().mtp_speculative_tokens),
        model.resources_.options().managed_weights);
    model.resources_.weight_loader_ = std::make_unique<WeightLoader>(
        model.resources_.weights_, model.resources_.options().weight_mode);

    std::unique_lock<std::mutex> shared_weights_lock(model.resources_.weights_->mutex);
    model.resources_.weights_->repo = bootstrap.checkpoint.repository;
    const IWeightRepository& repo = *model.resources_.weights_->repo;
    model.resources_.embedding_ = model.resources_.weight_loader_->load_linear_weight(
        repo, tensor_name(model.resources_.model_.weight_plan.requests,
                          TensorRole::TokenEmbedding),
        {model.resources_.dims().vocab_size, model.resources_.program_.hidden});
    const NormSpec& final_norm = model.resources_.program_.final_norm;
    const std::string final_name = final_norm.weightless()
        ? std::string{} : tensor_name(model.resources_.model_.weight_plan.requests,
                                      TensorRole::FinalNorm);
    model.resources_.final_norm_ = model.resources_.weight_loader_->load_rms_norm_weight(
        repo, final_name, {model.resources_.program_.hidden}, final_norm.weight_kind);
    if (model.resources_.program_.embedding_transform.post_norm) {
        const NormSpec& embedding_norm =
            *model.resources_.program_.embedding_transform.post_norm;
        const std::string embedding_name = embedding_norm.weightless()
            ? std::string{} : final_name;
        model.resources_.embedding_norm_ =
            model.resources_.weight_loader_->load_rms_norm_weight(
                repo, embedding_name, {model.resources_.program_.hidden},
                embedding_norm.weight_kind);
    }

    if (model.resources_.program_.per_layer_input.enabled) {
        const int input_size = model.resources_.program_.per_layer_input.input_size;
        CudaPerLayerInputResources per_layer;
        per_layer.plan = model.resources_.program_.per_layer_input;
        per_layer.embedding = model.resources_.weight_loader_->load_linear_weight(
            repo, tensor_name(model.resources_.model_.weight_plan.requests,
                              TensorRole::PerLayerEmbedding),
            {model.resources_.dims().vocab_size,
             static_cast<int>(per_layer.plan.packed_width)});
        per_layer.context_projection = model.resources_.weight_loader_->load_linear_weight(
            repo, tensor_name(model.resources_.model_.weight_plan.requests,
                              TensorRole::PerLayerContextProjection),
            {static_cast<int>(per_layer.plan.packed_width), model.resources_.program_.hidden});
        per_layer.projection_norm = model.resources_.weight_loader_->load_weight(
            repo, tensor_name(model.resources_.model_.weight_plan.requests,
                              TensorRole::PerLayerProjectionNorm),
            {input_size});
        per_layer.embedding_layout = make_cuda_embedding_layout(
            model.resources_.options().weight_mode, *per_layer.embedding,
            "per-layer embedding");
        per_layer.validate();
        model.resources_.per_layer_input_ = std::move(per_layer);
    }

    model.resources_.weight_layout_ = make_cuda_embedding_layout(
        model.resources_.options().weight_mode, *model.resources_.embedding_, "embedding");
    const std::string lm_head_name = tensor_name(
        model.resources_.model_.weight_plan.requests,
        TensorRole::LanguageModelHead);
    if (!model.resources_.model_.graph.tied_embeddings &&
        repo.contains(lm_head_name)) {
        model.resources_.lm_head_ = model.resources_.weight_loader_->load_linear_weight(
            repo, lm_head_name,
        {model.resources_.dims().vocab_size, model.resources_.program_.hidden});
    }

    load_layers(repo);
}

}
