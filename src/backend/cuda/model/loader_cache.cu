#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/backend/cuda/moe/expert_source.hpp"

#include <filesystem>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace celeg {

std::shared_ptr<SharedModelWeights> WeightLoader::acquire(
    const std::string& model_path, WeightMode weight_mode,
    const std::string& residency_fingerprint) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::weak_ptr<SharedModelWeights>> cache;
    int device_id = 0;
    CELEG_CUDA(cudaGetDevice(&device_id));
    std::ostringstream key_builder;
    key_builder << device_id << ':' << static_cast<int>(weight_mode) << ':'
                << (g_cuda_managed_weight_allocations ? "managed" : "device") << ':'
                << std::filesystem::weakly_canonical(model_path).string();
    const std::string key = key_builder.str();
    std::lock_guard<std::mutex> lock(cache_mutex);
    std::shared_ptr<SharedModelWeights> weights;
    if (const auto found = cache.find(key); found != cache.end())
        weights = found->second.lock();
    if (weights && weights->residency_fingerprint != residency_fingerprint)
        throw std::invalid_argument("incompatible CUDA MoE residency options for shared checkpoint weights");
    if (!weights) {
        weights = std::make_shared<SharedModelWeights>();
        weights->residency_fingerprint = residency_fingerprint;
        weights->residency_coordinator =
            std::make_shared<CudaExpertResidencyCoordinator>(*weights);
        weights->expert_source = std::make_shared<CudaExpertSource>(*weights);
        cache[key] = weights;
    }
    return weights;
}

WeightLoader::WeightLoader(std::shared_ptr<SharedModelWeights> weights,
                           WeightMode weight_mode)
    : weights_(std::move(weights)), weight_mode_(weight_mode) {
    if (!weights_) throw std::invalid_argument("WeightLoader requires non-null weights");
}

} // namespace celeg


