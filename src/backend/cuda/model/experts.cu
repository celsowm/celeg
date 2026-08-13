#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/backend/cuda/moe/expert_residency.hpp"
#include "celeg/checkpoint/tensor_names.hpp"
#include "celeg/checkpoint/packed/int4.hpp"
#include <cstring>
#include <stdexcept>
#include <vector>

namespace celeg {

namespace {

std::vector<__nv_bfloat16> load_named_expert_matrix(
    const IWeightRepository& repo, const std::string& name,
    const std::vector<int64_t>& shape) {
    if (has_packed_int4_matrix(repo, name)) {
        const PackedInt4Matrix packed = load_packed_int4_matrix(repo, name, shape);
        const std::vector<float> values = dequantize_packed_int4(packed);
        std::vector<__nv_bfloat16> result(values.size());
        for (size_t i = 0; i < values.size(); ++i) result[i] = __float2bfloat16(values[i]);
        return result;
    }
    const HostTensorView tensor = repo.tensor(name);
    const size_t count = static_cast<size_t>(shape[0]) * static_cast<size_t>(shape[1]);
    if (tensor.dtype != TensorDType::BF16 || tensor.shape != shape ||
        tensor.bytes != count * sizeof(__nv_bfloat16)) {
        throw std::runtime_error("invalid named MoE expert tensor: " + name);
    }
    const auto* source = reinterpret_cast<const __nv_bfloat16*>(tensor.data);
    return std::vector<__nv_bfloat16>(source, source + count);
}

} // namespace

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
    __nv_bfloat16* gate_up_base = nullptr;
    __nv_bfloat16* down_base = nullptr;
    const __nv_bfloat16* gate_up_device_base = nullptr;
    const __nv_bfloat16* down_device_base = nullptr;
    if (host_mode == ExpertHostMode::Mapped) {
        const HostExpertStore::MappedRange gate_up_range =
            store.alloc_mapped(layer_gate_up_bytes);
        const HostExpertStore::MappedRange down_range =
            store.alloc_mapped(layer_down_bytes);
        gate_up_base = static_cast<__nv_bfloat16*>(gate_up_range.host);
        down_base = static_cast<__nv_bfloat16*>(down_range.host);
        gate_up_device_base = static_cast<const __nv_bfloat16*>(gate_up_range.device);
        down_device_base = static_cast<const __nv_bfloat16*>(down_range.device);
        for (int e = 0; e < num_experts; ++e) {
            result.gate_up_host_dev[static_cast<size_t>(e)] =
                gate_up_device_base + static_cast<size_t>(e) *
                    (gate_up_bytes / sizeof(__nv_bfloat16));
            result.down_host_dev[static_cast<size_t>(e)] =
                down_device_base + static_cast<size_t>(e) *
                    (down_bytes / sizeof(__nv_bfloat16));
        }
    }

    // Reusable host staging buffer: assemble each expert's packed layout
    // (gate_up = [w1; w3], down = [w2]).
    std::vector<__nv_bfloat16> gate_up_stage(gate_up_elems);
    std::vector<__nv_bfloat16> down_stage(down_elems);
    std::vector<__nv_bfloat16> decoded_stage;
    const std::string packed_gate_up_name = "model.language_model.layers." +
        std::to_string(layer) + ".mlp.experts.gate_up_proj";
    const std::string packed_down_name = "model.language_model.layers." +
        std::to_string(layer) + ".mlp.experts.down_proj";
    const bool packed_experts = repo.contains(packed_gate_up_name) &&
        repo.contains(packed_down_name);
    HostTensorView packed_gate_up;
    HostTensorView packed_down;
    if (packed_experts) {
        packed_gate_up = repo.tensor(packed_gate_up_name);
        packed_down = repo.tensor(packed_down_name);
        const bool gu_shape = packed_gate_up.shape == std::vector<int64_t>{
            num_experts, 2 * moe_intermediate, hidden};
        const bool gu_flat_shape = packed_gate_up.shape == std::vector<int64_t>{
            num_experts * 2 * moe_intermediate, hidden};
        const bool down_shape = packed_down.shape == std::vector<int64_t>{
            num_experts, hidden, moe_intermediate};
        const bool down_flat_shape = packed_down.shape == std::vector<int64_t>{
            num_experts * hidden, moe_intermediate};
        if (!gu_shape && !gu_flat_shape || !down_shape && !down_flat_shape ||
            packed_gate_up.dtype != TensorDType::BF16 ||
            packed_down.dtype != TensorDType::BF16 ||
            packed_gate_up.bytes != gate_up_elems * static_cast<size_t>(num_experts) *
                sizeof(__nv_bfloat16) ||
            packed_down.bytes != down_elems * static_cast<size_t>(num_experts) *
                sizeof(__nv_bfloat16)) {
            throw std::runtime_error("packed experts must be BF16 [E, rows, cols]");
        }
    }

    for (int e = 0; e < num_experts; ++e) {
        if (packed_experts) {
            const size_t gu_offset = static_cast<size_t>(e) * gate_up_elems;
            const size_t down_offset = static_cast<size_t>(e) * down_elems;
            std::memcpy(gate_up_stage.data(),
                        reinterpret_cast<const __nv_bfloat16*>(packed_gate_up.data) + gu_offset,
                        gate_up_bytes);
            std::memcpy(down_stage.data(),
                        reinterpret_cast<const __nv_bfloat16*>(packed_down.data) + down_offset,
                        down_bytes);
        } else {
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
        }

        if (host_mode == ExpertHostMode::Mapped) {
            // Copy the packed expert into its slot in the persistent arena.
            const size_t gu_off = static_cast<size_t>(e) * gate_up_elems;
            const size_t dw_off = static_cast<size_t>(e) * down_elems;
            std::memcpy(gate_up_base + gu_off,
                        gate_up_stage.data(), gate_up_bytes);
            std::memcpy(down_base + dw_off,
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

WeightLoader::HostExpertLayer WeightLoader::load_moe_experts_host_named(
    const IWeightRepository& repo, const std::string& experts_prefix,
    const std::string& gate_name, const std::string& up_name,
    const std::string& down_name, int num_experts, int moe_intermediate,
    int hidden, HostExpertStore& store, ExpertHostMode host_mode) {
    if (num_experts <= 0 || moe_intermediate <= 0 || hidden <= 0) {
        throw std::runtime_error("invalid named MoE expert dimensions");
    }
    const size_t inter = static_cast<size_t>(moe_intermediate);
    const size_t hidden_count = static_cast<size_t>(hidden);
    const size_t gate_up_elems = 2 * inter * hidden_count;
    const size_t down_elems = hidden_count * inter;
    const size_t projection_bytes = inter * hidden_count * sizeof(__nv_bfloat16);
    const size_t gate_up_bytes = gate_up_elems * sizeof(__nv_bfloat16);
    const size_t down_bytes = down_elems * sizeof(__nv_bfloat16);

    HostExpertLayer result;
    result.gate_up_bytes = gate_up_bytes;
    result.down_bytes = down_bytes;
    result.gate_up_host_dev.resize(static_cast<size_t>(num_experts));
    result.down_host_dev.resize(static_cast<size_t>(num_experts));

    __nv_bfloat16* gate_up_base = nullptr;
    __nv_bfloat16* down_base = nullptr;
    const __nv_bfloat16* gate_up_device_base = nullptr;
    const __nv_bfloat16* down_device_base = nullptr;
    if (host_mode == ExpertHostMode::Mapped) {
        const HostExpertStore::MappedRange gate_up_range =
            store.alloc_mapped(gate_up_bytes * static_cast<size_t>(num_experts));
        const HostExpertStore::MappedRange down_range =
            store.alloc_mapped(down_bytes * static_cast<size_t>(num_experts));
        gate_up_base = static_cast<__nv_bfloat16*>(gate_up_range.host);
        down_base = static_cast<__nv_bfloat16*>(down_range.host);
        gate_up_device_base = static_cast<const __nv_bfloat16*>(gate_up_range.device);
        down_device_base = static_cast<const __nv_bfloat16*>(down_range.device);
        for (int expert = 0; expert < num_experts; ++expert) {
            result.gate_up_host_dev[static_cast<size_t>(expert)] =
                gate_up_device_base + static_cast<size_t>(expert) * gate_up_elems;
            result.down_host_dev[static_cast<size_t>(expert)] =
                down_device_base + static_cast<size_t>(expert) * down_elems;
        }
    }

    std::vector<__nv_bfloat16> gate_up_stage(gate_up_elems);
    std::vector<__nv_bfloat16> down_stage(down_elems);
    for (int expert = 0; expert < num_experts; ++expert) {
        const std::string gate = experts_prefix + "." + std::to_string(expert) +
            "." + gate_name + ".weight";
        const std::string up = experts_prefix + "." + std::to_string(expert) +
            "." + up_name + ".weight";
        const std::string down = experts_prefix + "." + std::to_string(expert) +
            "." + down_name + ".weight";
        const std::vector<int64_t> projection_shape{moe_intermediate, hidden};
        const std::vector<__nv_bfloat16> gate_tensor = load_named_expert_matrix(
            repo, gate, projection_shape);
        const std::vector<__nv_bfloat16> up_tensor = load_named_expert_matrix(
            repo, up, projection_shape);
        const std::vector<__nv_bfloat16> down_tensor = load_named_expert_matrix(
            repo, down, {hidden, moe_intermediate});
        std::memcpy(gate_up_stage.data(), gate_tensor.data(), projection_bytes);
        std::memcpy(gate_up_stage.data() + inter * hidden_count,
                    up_tensor.data(), projection_bytes);
        std::memcpy(down_stage.data(), down_tensor.data(), down_bytes);
        if (host_mode == ExpertHostMode::Mapped) {
            std::memcpy(gate_up_base + static_cast<size_t>(expert) * gate_up_elems,
                        gate_up_stage.data(), gate_up_bytes);
            std::memcpy(down_base + static_cast<size_t>(expert) * down_elems,
                        down_stage.data(), down_bytes);
        } else {
            result.gate_up_host_dev[static_cast<size_t>(expert)] =
                static_cast<const __nv_bfloat16*>(
                    store.store_pinned_copy(gate_up_stage.data(), gate_up_bytes));
            result.down_host_dev[static_cast<size_t>(expert)] =
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
    const std::string packed_gate_up_name = "model.language_model.layers." +
        std::to_string(layer) + ".mlp.experts.gate_up_proj";
    const std::string packed_down_name = "model.language_model.layers." +
        std::to_string(layer) + ".mlp.experts.down_proj";
    const bool packed_experts = repo.contains(packed_gate_up_name) &&
        repo.contains(packed_down_name);
    if (packed_experts) {
        const TensorLocator gate_up = locator.locate(packed_gate_up_name);
        const TensorLocator down = locator.locate(packed_down_name);
        const std::vector<int64_t> gate_up_shape{
            num_experts, 2 * moe_intermediate, hidden};
        const std::vector<int64_t> gate_up_flat_shape{
            num_experts * 2 * moe_intermediate, hidden};
        const std::vector<int64_t> down_shape{
            num_experts, hidden, moe_intermediate};
        const std::vector<int64_t> down_flat_shape{
            num_experts * hidden, moe_intermediate};
        if ((gate_up.shape != gate_up_shape && gate_up.shape != gate_up_flat_shape) ||
            (down.shape != down_shape && down.shape != down_flat_shape) ||
            gate_up.dtype != TensorDType::BF16 || down.dtype != TensorDType::BF16) {
            throw std::invalid_argument(
                "packed experts must be BF16 [E, rows, cols]");
        }
        const size_t gate_up_expert_bytes = static_cast<size_t>(2 * moe_intermediate) *
            static_cast<size_t>(hidden) * sizeof(__nv_bfloat16);
        const size_t down_expert_bytes = static_cast<size_t>(hidden) *
            static_cast<size_t>(moe_intermediate) * sizeof(__nv_bfloat16);
        for (int e = 0; e < num_experts; ++e) {
            TensorLocator w1 = gate_up;
            TensorLocator w3 = gate_up;
            TensorLocator w2 = down;
            const std::uint64_t gate_up_offset = static_cast<std::uint64_t>(e) *
                gate_up_expert_bytes;
            w1.absolute_offset += gate_up_offset;
            w1.bytes = down_expert_bytes;
            w1.shape = {moe_intermediate, hidden};
            w3.absolute_offset = w1.absolute_offset + down_expert_bytes;
            w3.bytes = down_expert_bytes;
            w3.shape = {moe_intermediate, hidden};
            w2.absolute_offset += static_cast<std::uint64_t>(e) * down_expert_bytes;
            w2.bytes = down_expert_bytes;
            w2.shape = {hidden, moe_intermediate};
            catalog[static_cast<size_t>(e)] = ExpertLocation{w1, w2, w3};
        }
        return catalog;
    }
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

std::vector<ExpertLocation> WeightLoader::build_expert_catalog_named(
    const IWeightRepository& repo, const std::string& experts_prefix,
    const std::string& gate_name, const std::string& up_name,
    const std::string& down_name, int num_experts, int moe_intermediate,
    int hidden) {
    if (num_experts <= 0 || moe_intermediate <= 0 || hidden <= 0) {
        throw std::runtime_error("invalid named MoE catalog dimensions");
    }
    const auto& locator = require_locatable_tensor_repository(repo);
    std::vector<ExpertLocation> result(static_cast<size_t>(num_experts));
    for (int expert = 0; expert < num_experts; ++expert) {
        const std::string gate = experts_prefix + "." + std::to_string(expert) +
            "." + gate_name + ".weight";
        const std::string up = experts_prefix + "." + std::to_string(expert) +
            "." + up_name + ".weight";
        const std::string down = experts_prefix + "." + std::to_string(expert) +
            "." + down_name + ".weight";
        TensorLocator gate_loc = locator.locate(gate);
        TensorLocator up_loc = locator.locate(up);
        TensorLocator down_loc = locator.locate(down);
        if (gate_loc.shape != std::vector<int64_t>{moe_intermediate, hidden} ||
            up_loc.shape != std::vector<int64_t>{moe_intermediate, hidden} ||
            down_loc.shape != std::vector<int64_t>{hidden, moe_intermediate} ||
            gate_loc.dtype != TensorDType::BF16 || up_loc.dtype != TensorDType::BF16 ||
            down_loc.dtype != TensorDType::BF16) {
            throw std::runtime_error("invalid named MoE catalog tensor: " + gate);
        }
        result[static_cast<size_t>(expert)] = ExpertLocation{
            gate_loc, down_loc, up_loc};
    }
    return result;
}


} // namespace celeg
