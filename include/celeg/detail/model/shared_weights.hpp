#pragma once

// Process-wide shared weight arena and expert residency/offload state.
//
// This is the widest of the model implementation headers: it is where weight
// ownership meets expert offload planning, the host expert cache and the
// residency coordinator. Only components that manage weight lifetime
// (`WeightLoader`, residency, expert sources) should include it — the
// forward pass binds views from `linear_weights.hpp` / `layer_state.hpp`
// instead.

#include "celeg/detail/model/device_weights.hpp"
#include "celeg/backend/cuda/moe/offload.hpp"
#include "celeg/backend/cuda/moe/expert_residency.hpp"
#include "celeg/backend/cuda/utils.cuh"
#include "celeg/checkpoint/formats/safetensors.hpp"
#include "celeg/runtime/cache/host_expert_cache.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace celeg {

struct ResidencyController {
    std::mutex mutex;
    std::unique_ptr<ExpertLayerCache> cache;
    std::unique_ptr<CudaStream> transfer_stream;

    struct InflightTransfer {
        ExpertHostLease lease;
        std::unique_ptr<CudaEvent> event;
    };
    std::vector<InflightTransfer> inflight_transfers;
};

class CudaExpertSource;

// Process-wide shared weight arena. Multiple CudaModel sessions on the same
// device + checkpoint + weight_mode share one instance to avoid duplicate
// GPU allocations.
struct SharedModelWeights {
    std::mutex mutex;
    std::string residency_fingerprint;
    WeightMap tensors;

    std::shared_ptr<IWeightRepository> repo;

    ExpertOffloadPlan expert_offload_plan;
    HostExpertStore host_expert_store;
    std::vector<std::unique_ptr<ResidencyController>> expert_controllers;
    std::vector<std::vector<ExpertLocation>> expert_catalog;
    std::unique_ptr<HostExpertCache> host_expert_cache;
    std::shared_ptr<CudaExpertSource> expert_source;
    std::unique_ptr<ExpertSidecar> expert_sidecar;
    std::unique_ptr<ExpertIoManager> expert_io_manager;
    ModelUsageStats usage_stats;
    std::string usage_profile_path;

    std::shared_ptr<CudaExpertResidencyCoordinator> residency_coordinator;

    size_t memory_bytes() const;
};

} // namespace celeg
