#pragma once

#include "celeg/model/resolved.hpp"
#include "celeg/model/program.hpp"
#include "backend/cuda/execution_plan.hpp"
#include "backend/cuda/weight_layout.hpp"
#include "backend/cuda/weights_loader.hpp"
#include "celeg/model/weights/roles.hpp"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace celeg {

struct CudaPerLayerInputResources {
    PerLayerInputPlan plan;
    const LinearWeight* embedding = nullptr;
    const LinearWeight* context_projection = nullptr;
    const __nv_bfloat16* projection_norm = nullptr;
    std::unique_ptr<IWeightLayout> embedding_layout;

    void validate() const {
        plan.validate();
        if (plan.enabled && (!embedding || !context_projection || !projection_norm ||
                             !embedding_layout)) {
            throw std::invalid_argument("incomplete CUDA per-layer input resources");
        }
    }
};

struct CudaMtpResources {
    bool enabled = false;
    int layer_count = 0;
    const LinearWeight* fc = nullptr;
    const __nv_bfloat16* pre_fc_norm_embedding = nullptr;
    const __nv_bfloat16* pre_fc_norm_hidden = nullptr;
    const __nv_bfloat16* norm = nullptr;
    const LinearWeight* logits = nullptr;
    std::vector<Layer> layers;

    bool available() const {
        return enabled && layer_count > 0 && fc != nullptr &&
            pre_fc_norm_embedding != nullptr && pre_fc_norm_hidden != nullptr &&
            norm != nullptr && logits != nullptr &&
            static_cast<int>(layers.size()) == layer_count;
    }
};

struct CudaModelResources {
    explicit CudaModelResources(CudaExecutionPlan execution_plan)
        : plan_(std::move(execution_plan)), options_(plan_.options()),
          shape_(topology_.exec), dims_(topology_.dims) {}

    CudaExecutionPlan plan_;
    CudaModelOptions options_;
    ResolvedModel model_;
    CompiledModelProgram program_;
    RuntimeTopology topology_;
    ExecutionTopology& shape_;
    CheckpointDimensions& dims_;
    std::string model_identity_;
    std::shared_ptr<SharedModelWeights> weights_;
    std::unique_ptr<WeightLoader> weight_loader_;
    std::vector<Layer> layers_;
    CudaMtpResources mtp_;
    const LinearWeight* embedding_ = nullptr;
    std::optional<CudaPerLayerInputResources> per_layer_input_;
    const LinearWeight* lm_head_ = nullptr;
    const __nv_bfloat16* embedding_norm_ = nullptr;
    const __nv_bfloat16* final_norm_ = nullptr;
    std::unique_ptr<IWeightLayout> weight_layout_;
};

}
