#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/backend/cuda/moe/expert_residency.hpp"
#include "celeg/checkpoint/tensor_names.hpp"
#include <cstring>
#include <stdexcept>
#include <vector>

namespace celeg {
WeightLoader::HostExpertLayer WeightLoader::load_moe_experts_host(
    const IWeightRepository& repo, int layer,
    int num_experts, int moe_intermediate, int hidden,
    HostExpertStore& store, ExpertHostMode host_mode) {
    if (num_experts <= 0 || moe_intermediate <= 0 || hidden <= 0) {
        throw std::runtime_error("invalid MoE expert dimensions for layer " +
                                 std::to_string(layer));
    }
    const size_t moe_inter = static_cast<size_t>(moe_intermediate);
    const size_t hidden_c = static_cast<size_t>(hidden);
    const size_t gate_up_elems = 2 * moe_inter * hidden_c;
    const size_t down_elems = hidden_c * moe_inter;
    const size_t gate_up_bytes = gate_up_elems * sizeof(__nv_bfloat16);
    const size_t down_bytes = down_elems * sizeof(__nv_bfloat16);
    const size_t w_bytes = moe_inter * hidden_c * sizeof(__nv_bfloat16);

    HostExpertLayer result;
    result.gate_up_bytes = gate_up_bytes;
    result.down_bytes = down_bytes;
    result.gate_up_host_dev.resize(static_cast<size_t>(num_experts));
    result.down_host_dev.resize(static_cast<size_t>(num_experts));

    const size_t layer_gate_up_bytes = gate_up_bytes * static_cast<size_t>(num_experts);
    const size_t layer_down_bytes = down_bytes * static_cast<size_t>(num_experts);

    // Mapped mode: allocate one persistent arena per layer (a single pinned +
    // registered region) and point each expert into it. This avoids 704 separate
    // cudaHostAlloc calls that can exhaust the driver's pinned-memory pool.
    // Pinned mode: copy each expert into its own pinned allocation.
    const __nv_bfloat16* gate_up_base = nullptr;
    const __nv_bfloat16* down_base = nullptr;
    if (host_mode == ExpertHostMode::Mapped) {
        gate_up_base = static_cast<const __nv_bfloat16*>(
            store.alloc_mapped(layer_gate_up_bytes));
        down_base = static_cast<const __nv_bfloat16*>(
            store.alloc_mapped(layer_down_bytes));
        for (int e = 0; e < num_experts; ++e) {
            result.gate_up_host_dev[static_cast<size_t>(e)] =
                gate_up_base + static_cast<size_t>(e) * (gate_up_bytes / sizeof(__nv_bfloat16));
            result.down_host_dev[static_cast<size_t>(e)] =
                down_base + static_cast<size_t>(e) * (down_bytes / sizeof(__nv_bfloat16));
        }
    }

    // Reusable host staging buffer: assemble each expert's packed layout
    // (gate_up = [w1; w3], down = [w2]).
    std::vector<__nv_bfloat16> gate_up_stage(gate_up_elems);
    std::vector<__nv_bfloat16> down_stage(down_elems);
    std::vector<__nv_bfloat16> decoded_stage;

    for (int e = 0; e < num_experts; ++e) {
        const std::string w1_name = layer_name(
            layer, "feed_forward.experts." + std::to_string(e) + ".w1.weight");
        const std::string w3_name = layer_name(
            layer, "feed_forward.experts." + std::to_string(e) + ".w3.weight");
        const std::string w2_name = layer_name(
            layer, "feed_forward.experts." + std::to_string(e) + ".w2.weight");
        const HostTensorView w1 = repo.tensor(w1_name);
        const HostTensorView w3 = repo.tensor(w3_name);
        const HostTensorView w2 = repo.tensor(w2_name);
        if (w1.shape != std::vector<int64_t>{moe_intermediate, hidden} ||
            w3.shape != std::vector<int64_t>{moe_intermediate, hidden} ||
            w2.shape != std::vector<int64_t>{hidden, moe_intermediate}) {
            throw std::runtime_error("unexpected MoE expert tensor shape for " + w1_name);
        }
        const auto copy_or_dequantize = [&](const HostTensorView& tensor,
                                            __nv_bfloat16* destination,
                                            size_t bytes,
                                            const std::string& name) {
            if (tensor.dtype == TensorDType::BF16) {
                if (tensor.bytes != bytes) {
                    throw std::runtime_error("invalid BF16 MoE expert bytes: " + name);
                }
                std::memcpy(destination, tensor.data, bytes);
                return;
            }
            if (tensor.dtype == TensorDType::Quantized) {
                dequantize_gguf_to_bf16(tensor, decoded_stage);
                if (decoded_stage.size() * sizeof(__nv_bfloat16) != bytes) {
                    throw std::runtime_error("invalid GGUF MoE expert size: " + name);
                }
                std::memcpy(destination, decoded_stage.data(), bytes);
                return;
            }
            throw std::runtime_error("unsupported MoE expert dtype: " + name);
        };
        copy_or_dequantize(w1, gate_up_stage.data(), w_bytes, w1_name);
        copy_or_dequantize(w3, gate_up_stage.data() + moe_inter * hidden_c,
                           w_bytes, w3_name);
        copy_or_dequantize(w2, down_stage.data(), down_bytes, w2_name);

        if (host_mode == ExpertHostMode::Mapped) {
            // Copy the packed expert into its slot in the persistent arena.
            const size_t gu_off = static_cast<size_t>(e) * gate_up_elems;
            const size_t dw_off = static_cast<size_t>(e) * down_elems;
            std::memcpy(const_cast<__nv_bfloat16*>(gate_up_base) + gu_off,
                        gate_up_stage.data(), gate_up_bytes);
            std::memcpy(const_cast<__nv_bfloat16*>(down_base) + dw_off,
                        down_stage.data(), down_bytes);
        } else {
            result.gate_up_host_dev[static_cast<size_t>(e)] =
                static_cast<const __nv_bfloat16*>(
                    store.store_pinned_copy(gate_up_stage.data(), gate_up_bytes));
            result.down_host_dev[static_cast<size_t>(e)] =
                static_cast<const __nv_bfloat16*>(
                    store.store_pinned_copy(down_stage.data(), down_bytes));
        }
    }
    return result;
}

std::vector<ExpertLocation> WeightLoader::build_expert_catalog(
    const IWeightRepository& repo, int layer,
    int num_experts, int moe_intermediate, int hidden) {
    if (num_experts <= 0 || moe_intermediate <= 0 || hidden <= 0) {
        throw std::runtime_error("invalid MoE expert dimensions for layer " +
                                 std::to_string(layer));
    }
    std::vector<ExpertLocation> catalog(static_cast<size_t>(num_experts));
    const auto& locator = require_locatable_tensor_repository(repo);
    for (int e = 0; e < num_experts; ++e) {
        const std::string w1_name = layer_name(
            layer, "feed_forward.experts." + std::to_string(e) + ".w1.weight");
        const std::string w3_name = layer_name(
            layer, "feed_forward.experts." + std::to_string(e) + ".w3.weight");
        const std::string w2_name = layer_name(
            layer, "feed_forward.experts." + std::to_string(e) + ".w2.weight");

        TensorLocator w1_loc = locator.locate(w1_name);
        TensorLocator w3_loc = locator.locate(w3_name);
        TensorLocator w2_loc = locator.locate(w2_name);

        if (w1_loc.shape != std::vector<int64_t>{moe_intermediate, hidden} ||
            w3_loc.shape != std::vector<int64_t>{moe_intermediate, hidden} ||
            w2_loc.shape != std::vector<int64_t>{hidden, moe_intermediate}) {
            throw std::runtime_error("unexpected MoE expert tensor shape for " + w1_name);
        }

        catalog[static_cast<size_t>(e)] = ExpertLocation{w1_loc, w2_loc, w3_loc};
    }
    return catalog;
}


} // namespace celeg
