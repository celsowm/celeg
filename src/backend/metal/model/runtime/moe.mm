#include "detail.hpp"
#include "moe_router.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace celeg {

void MetalModel::Impl::encode_moe(
    id<MTLCommandBuffer>& command_buffer,
    id<MTLComputeCommandEncoder>& encoder, Layer& layer) {
    const uint32_t hidden_width = static_cast<uint32_t>(model.graph.hidden);
    finish_commands(command_buffer, encoder);
    std::vector<float> router_logits(
        static_cast<size_t>(layer.moe->router.expert_count));
    for (int expert = 0; expert < layer.moe->router.expert_count; ++expert) {
        float sum = 0.0f;
        const size_t base = static_cast<size_t>(expert) * hidden_width;
        for (uint32_t index = 0; index < hidden_width; ++index) {
            sum += layer.moe->router_weight[base + index] *
                   static_cast<const float*>(normed.contents)[index];
        }
        router_logits[static_cast<size_t>(expert)] = sum;
    }
    const MetalMoeRoute route =
        route_metal_moe(layer.moe->router, router_logits, layer.moe->router_bias);
    struct ExpertBatch {
        Linear gate;
        Linear up;
        Linear down;
    };
    std::vector<ExpertBatch> batches;
    batches.reserve(route.experts.size());
    for (int expert : route.experts) {
        const Layer::Expert& names = layer.moe->experts[static_cast<size_t>(expert)];
        batches.push_back(
            {load_linear_source(names.gate_name, layer.intermediate, hidden_width),
             load_linear_source(names.up_name, layer.intermediate, hidden_width),
             load_linear_source(names.down_name, hidden_width, layer.intermediate)});
    }
    std::memset(moe_output.contents, 0, static_cast<size_t>(hidden_width) * sizeof(float));
    begin_commands(command_buffer, encoder);
    const uint32_t intermediate = static_cast<uint32_t>(layer.intermediate);
    for (size_t route_index = 0; route_index < route.experts.size(); ++route_index) {
        const ExpertBatch& batch = batches[route_index];
        encode_matvec(encoder, batch.gate, normed, gate_up, 0);
        encode_matvec(encoder, batch.up, normed, gate_up,
                      static_cast<NSUInteger>(layer.intermediate) * sizeof(float));
        set_buffer(encoder, gate_up, 0);
        set_buffer(encoder, activated, 1);
        set_bytes(encoder, &intermediate, sizeof(intermediate), 2);
        dispatch(encoder, "celeg_swiglu", intermediate);
        encode_matvec(encoder, batch.down, activated, operation);
        encode_weighted_add(encoder, operation, moe_output, hidden_width,
                            route.weights[route_index]);
    }
}

}
