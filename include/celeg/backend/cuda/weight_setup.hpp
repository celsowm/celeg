#pragma once

#include <functional>
#include <string>

namespace celeg {

struct CudaCompiledModel;
class IWeightRepository;

namespace detail {
struct ModelBootstrap;
}

class CudaWeightSetup {
public:
    using LayerLoader = std::function<void(const IWeightRepository&)>;

    static void load(CudaCompiledModel& model,
                     const std::string& model_path,
                     const detail::ModelBootstrap& bootstrap,
                     LayerLoader load_layers);
};

void load_mtp_weights(CudaCompiledModel& model, const IWeightRepository& repository);

} // namespace celeg
