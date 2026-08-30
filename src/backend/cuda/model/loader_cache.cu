#include "backend/cuda/weights_loader.hpp"
#include "backend/cuda/moe/expert_source.hpp"

#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace celeg {

std::shared_ptr<SharedModelWeights> CudaWeightCache::acquire(
    const std::string& model_path, WeightMode weight_mode,
    const std::string& residency_fingerprint, bool managed_weights) {
    int device_id = 0;
    CELEG_CUDA(cudaGetDevice(&device_id));
    std::ostringstream key_builder;
    key_builder << device_id << ':' << static_cast<int>(weight_mode) << ':'
                << (managed_weights ? "managed" : "device") << ':'
                << std::filesystem::weakly_canonical(model_path).string();
    const std::string key = key_builder.str();
    std::lock_guard<std::mutex> lock(mutex_);
    std::shared_ptr<SharedModelWeights> weights;
    if (const auto found = entries_.find(key); found != entries_.end())
        weights = found->second.lock();
    if (weights && weights->residency_fingerprint != residency_fingerprint)
        throw std::invalid_argument("incompatible CUDA MoE residency options for shared checkpoint weights");
    if (!weights) {
        weights = std::make_shared<SharedModelWeights>();
        weights->residency_fingerprint = residency_fingerprint;
        weights->residency_coordinator =
            std::make_shared<CudaExpertResidencyCoordinator>(*weights);
        weights->expert_source = std::make_shared<CudaExpertSource>(*weights);
        entries_[key] = weights;
    }
    return weights;
}

WeightLoader::WeightLoader(std::shared_ptr<SharedModelWeights> weights,
                           WeightMode weight_mode)
    : weights_(std::move(weights)), weight_mode_(weight_mode),
      weight_mode_resolver_([weight_mode](const std::string&) { return weight_mode; }) {
    if (!weights_) throw std::invalid_argument("WeightLoader requires non-null weights");
}

WeightLoader::WeightLoader(std::shared_ptr<SharedModelWeights> weights,
                           WeightModeResolver weight_mode_resolver)
    : weights_(std::move(weights)), weight_mode_(weight_mode_resolver({})),
      weight_mode_resolver_(std::move(weight_mode_resolver)) {
    if (!weights_) throw std::invalid_argument("WeightLoader requires non-null weights");
    if (!weight_mode_resolver_) {
        throw std::invalid_argument("WeightLoader requires a non-null weight mode resolver");
    }
}

}
