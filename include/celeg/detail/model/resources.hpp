#pragma once

#include "celeg/model/resolved.hpp"
#include "celeg/model/program.hpp"
#include "celeg/backend/cuda/execution_plan.hpp"
#include "celeg/backend/cuda/weight_layout.hpp"
#include "celeg/backend/cuda/weights_loader.hpp"
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

// CUDA model resource owner for immutable/shared topology decisions. Device
// weights and backend caches are attached by setup, while request-local
// position, phase, and metrics live in SessionState.
struct CudaModelResources {
    explicit CudaModelResources(CudaExecutionPlan execution_plan)
        : plan_(std::move(execution_plan)), options_(plan_.options()) {}

    CudaExecutionPlan plan_;
    CudaModelOptions options_;
    ResolvedModel model_;
    CompiledModelProgram program_;
    RuntimeTopology shape_;
    std::string model_identity_;
    std::shared_ptr<SharedModelWeights> weights_;
    std::unique_ptr<WeightLoader> weight_loader_;
    std::vector<Layer> layers_;
    const LinearWeight* embedding_ = nullptr;
    std::optional<CudaPerLayerInputResources> per_layer_input_;
    const LinearWeight* lm_head_ = nullptr;
    const __nv_bfloat16* final_norm_ = nullptr;
    std::unique_ptr<IWeightLayout> weight_layout_;
};

} // namespace celeg
