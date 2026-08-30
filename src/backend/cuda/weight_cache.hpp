#pragma once

#include "model/detail/shared_weights.hpp"
#include "runtime_types.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace celeg {

class CudaWeightCache final {
public:
    std::shared_ptr<SharedModelWeights> acquire(
        const std::string& model_path,
        WeightMode weight_mode,
        const std::string& residency_fingerprint,
        bool managed_weights);

private:
    std::mutex mutex_;
    std::unordered_map<std::string, std::weak_ptr<SharedModelWeights>> entries_;
};

}
