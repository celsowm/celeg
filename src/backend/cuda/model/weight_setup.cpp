#include "detail/compiled_model.hpp"
#include "checkpoint/detail/bootstrap.hpp"
#include "backend/cuda/weight_setup.hpp"
#include "attention_weight_setup.hpp"
#include "layer_weight_setup.hpp"
#include "non_attention_weight_setup.hpp"

#include <vector>

namespace celeg {

void CudaCompiledModel::load_checkpoint_weights(
    const std::string& model_path,
    const detail::ModelBootstrap& bootstrap) {
    CudaWeightSetup::load(*this, model_path, bootstrap,
        [this](const IWeightRepository& repo) {
            prepare_cuda_layer_weight_resources(*this);
            resources_.layers_.reserve(
                static_cast<size_t>(resources_.shape().num_hidden_layers));

            std::vector<int> shared_owner(2, -1);
            for (int layer_index = 0;
                 layer_index < resources_.shape().num_hidden_layers;
                 ++layer_index) {
                const CompiledLayerProgram& semantics = resources_.program_.layers.at(
                    static_cast<size_t>(layer_index));
                const LayerCommon common_layer = bind_cuda_layer_common(
                    *this, repo, semantics, layer_index);

                if (bind_cuda_attention_layer(
                        *this, repo, semantics, layer_index,
                        common_layer, shared_owner)) {
                    continue;
                }
                bind_cuda_non_attention_layer(
                    *this, repo, semantics, layer_index, common_layer);
            }

            load_mtp_weights(*this, repo);
        });
}

}
