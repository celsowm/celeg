#include "backend/cuda/weights_loader.hpp"
#include "backend/cuda/moe/expert_residency.hpp"
#include "celeg/checkpoint/tensor_names.hpp"
#include "celeg/checkpoint/packed/int4.hpp"
#include <cstring>
#include <stdexcept>
#include <vector>

namespace celeg {

namespace {

void validate_expert_name_counts(const MoeExpertTensorNames& names,
                                 int num_experts) {
    if (names.packed()) return;
    if (static_cast<int>(names.gate.size()) != num_experts ||
        static_cast<int>(names.up.size()) != num_experts ||
        static_cast<int>(names.down.size()) != num_experts) {
        throw std::runtime_error(
            "resolved routed-expert name count does not match the expert count");
    }
}

/// Copies one individually resolved routed-expert tensor into the BF16
/// staging buffer, dequantizing when the checkpoint stores it as a
/// checkpoint-packed int4 pair or as native GGUF blocks. The storage family
/// is decided by the tensor's dtype/capability, never by its name spelling.
void copy_or_dequantize(const IWeightRepository& repo, const std::string& name,
                        const std::vector<int64_t>& shape,
                        __nv_bfloat16* destination, size_t bytes,
                        std::vector<float>& decoded_stage) {
    if (has_packed_int4_matrix(repo, name)) {
        const PackedInt4Matrix packed = load_packed_int4_matrix(repo, name, shape);
        decoded_stage = dequantize_packed_int4(packed);
        if (decoded_stage.size() * sizeof(__nv_bfloat16) != bytes) {
            throw std::runtime_error("invalid packed int4 MoE expert size: " + name);
        }
        for (size_t i = 0; i < decoded_stage.size(); ++i) {
            destination[i] = __float2bfloat16(decoded_stage[i]);
        }
        return;
    }
    const HostTensorView tensor = repo.tensor(name);
    if (tensor.dtype == TensorDType::BF16) {
        if (tensor.bytes != bytes) {
            throw std::runtime_error("invalid BF16 MoE expert bytes: " + name);
        }
        std::memcpy(destination, tensor.data, bytes);
        return;
    }
    if (tensor.dtype == TensorDType::Quantized) {
        std::vector<__nv_bfloat16> decoded;
        dequantize_gguf_to_bf16(tensor, decoded);
        if (decoded.size() * sizeof(__nv_bfloat16) != bytes) {
            throw std::runtime_error("invalid GGUF MoE expert size: " + name);
        }
        std::memcpy(destination, decoded.data(), bytes);
        return;
    }
    throw std::runtime_error("unsupported MoE expert dtype: " + name);
}

}

WeightLoader::HostExpertLayer WeightLoader::load_moe_experts_host(
    const IWeightRepository& repo, const MoeExpertTensorNames& names,
    int num_experts, int moe_intermediate, int hidden,
    HostExpertStore& store, ExpertHostMode host_mode) {
    if (num_experts <= 0 || moe_intermediate <= 0 || hidden <= 0) {
        throw std::runtime_error("invalid MoE expert dimensions");
    }
    validate_expert_name_counts(names, num_experts);
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

    std::vector<__nv_bfloat16> gate_up_stage(gate_up_elems);
    std::vector<__nv_bfloat16> down_stage(down_elems);
    std::vector<float> decoded_stage;
    const std::vector<int64_t> projection_shape{moe_intermediate, hidden};
    const std::vector<int64_t> down_shape{hidden, moe_intermediate};
    HostTensorView packed_gate_up;
    HostTensorView packed_down;
    if (names.packed()) {
        packed_gate_up = repo.tensor(names.packed_gate_up);
        packed_down = repo.tensor(names.packed_down);
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
        if (names.packed()) {
            const size_t gu_offset = static_cast<size_t>(e) * gate_up_elems;
            const size_t down_offset = static_cast<size_t>(e) * down_elems;
            std::memcpy(gate_up_stage.data(),
                        reinterpret_cast<const __nv_bfloat16*>(packed_gate_up.data) + gu_offset,
                        gate_up_bytes);
            std::memcpy(down_stage.data(),
                        reinterpret_cast<const __nv_bfloat16*>(packed_down.data) + down_offset,
                        down_bytes);
        } else {
            copy_or_dequantize(repo, names.gate[static_cast<size_t>(e)],
                               projection_shape, gate_up_stage.data(), w_bytes,
                               decoded_stage);
            copy_or_dequantize(repo, names.up[static_cast<size_t>(e)],
                               projection_shape,
                               gate_up_stage.data() + moe_inter * hidden_c,
                               w_bytes, decoded_stage);
            copy_or_dequantize(repo, names.down[static_cast<size_t>(e)],
                               down_shape, down_stage.data(), down_bytes,
                               decoded_stage);
        }

        if (host_mode == ExpertHostMode::Mapped) {
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

std::vector<ExpertLocation> WeightLoader::build_expert_catalog(
    const IWeightRepository& repo, const MoeExpertTensorNames& names,
    int num_experts, int moe_intermediate, int hidden) {
    if (num_experts <= 0 || moe_intermediate <= 0 || hidden <= 0) {
        throw std::runtime_error("invalid MoE catalog dimensions");
    }
    validate_expert_name_counts(names, num_experts);
    std::vector<ExpertLocation> catalog(static_cast<size_t>(num_experts));
    const auto& locator = require_locatable_tensor_repository(repo);
    if (names.packed()) {
        const TensorLocator gate_up = locator.locate(names.packed_gate_up);
        const TensorLocator down = locator.locate(names.packed_down);
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
        const size_t index = static_cast<size_t>(e);
        TensorLocator gate_loc = locator.locate(names.gate[index]);
        TensorLocator up_loc = locator.locate(names.up[index]);
        TensorLocator down_loc = locator.locate(names.down[index]);
        if (gate_loc.shape != std::vector<int64_t>{moe_intermediate, hidden} ||
            up_loc.shape != std::vector<int64_t>{moe_intermediate, hidden} ||
            down_loc.shape != std::vector<int64_t>{hidden, moe_intermediate} ||
            gate_loc.dtype != TensorDType::BF16 || up_loc.dtype != TensorDType::BF16 ||
            down_loc.dtype != TensorDType::BF16) {
            throw std::runtime_error(
                "invalid routed-expert catalog tensor: " + names.gate[index]);
        }
        catalog[index] = ExpertLocation{gate_loc, down_loc, up_loc};
    }
    return catalog;
}

}
