#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/backend/cuda/kernels/gguf.cuh"
#include "celeg/checkpoint/formats/gguf.hpp"
#include "celeg/checkpoint/tensor_names.hpp"
#include "celeg/checkpoint/packed/int4.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace celeg {

namespace {

void validate_individual_names(const MoeExpertTensorNames& names,
                               int num_experts, const char* projection) {
    if (names.packed()) {
        throw std::invalid_argument(std::string(projection) +
            " loading requires individually resolved routed-expert names");
    }
    if (static_cast<int>(names.gate.size()) != num_experts ||
        static_cast<int>(names.up.size()) != num_experts ||
        static_cast<int>(names.down.size()) != num_experts) {
        throw std::invalid_argument(std::string(projection) +
            " name count does not match the routed-expert count");
    }
}

void validate_expert_dimensions(int experts, int intermediate, int hidden,
                               const char* projection) {
    if (experts <= 0 || intermediate <= 0 || hidden <= 0) {
        throw std::runtime_error("invalid MoE " + std::string(projection) +
                                 " dimensions");
    }
}

size_t gguf_row_bytes(int columns, GgmlType type, const std::string& name) {
    // Only MMQ-capable types can stay packed as expert blocks on the device;
    // everything else must have been routed to the host-decoded path already.
    if (!cuda_gguf_native_mmq(type)) {
        throw std::runtime_error("unsupported GGUF MoE quantization: " + name +
                                 " (" + ggml_type_name(type) + ")");
    }
    const GgmlTypeTrait trait = ggml_type_trait(type);
    if (columns <= 0 || columns % trait.block_size != 0) {
        throw std::runtime_error("GGUF MoE width is not block-aligned: " + name);
    }
    return static_cast<size_t>(columns) / static_cast<size_t>(trait.block_size) *
        trait.type_size;
}

/// Expert weights of a type with no native MMQ kernel are decoded on the
/// host at load time instead of being kept packed on the device.
bool is_host_decoded_moe_type(GgmlType type) {
    return ggml_row_decoder(type).has_value() && !cuda_gguf_native_mmq(type);
}

void validate_named_bf16(const HostTensorView& tensor,
                         const std::vector<int64_t>& shape,
                         size_t bytes, const std::string& name) {
    if (tensor.dtype != TensorDType::BF16 || tensor.shape != shape ||
        tensor.bytes != bytes) {
        throw std::runtime_error("inconsistent BF16 named MoE tensor: " + name);
    }
}

std::vector<__nv_bfloat16> load_bf16_or_dequantized(
    const IWeightRepository& repo, const std::string& name,
    const std::vector<int64_t>& shape) {
    if (has_packed_int4_matrix(repo, name)) {
        const PackedInt4Matrix packed = load_packed_int4_matrix(repo, name, shape);
        const std::vector<float> values = dequantize_packed_int4(packed);
        std::vector<__nv_bfloat16> result(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            result[i] = __float2bfloat16(values[i]);
        }
        return result;
    }
    const HostTensorView tensor = repo.tensor(name);
    const size_t bytes = static_cast<size_t>(shape[0]) * static_cast<size_t>(shape[1]) *
        sizeof(__nv_bfloat16);
    validate_named_bf16(tensor, shape, bytes, name);
    const auto* source = reinterpret_cast<const __nv_bfloat16*>(tensor.data);
    return std::vector<__nv_bfloat16>(source, source + bytes / sizeof(__nv_bfloat16));
}

}

const ExpertLinearWeight* WeightLoader::load_moe_gate_up(
    const IWeightRepository& repo, const MoeExpertTensorNames& names,
    int num_experts, int moe_intermediate, int hidden) {
    validate_expert_dimensions(num_experts, moe_intermediate, hidden, "gate_up");
    validate_individual_names(names, num_experts, "MoE gate/up");
    const std::string cache_key = names.gate.front() + ".moe.gate_up";
    if (const auto cached = expert_cache_.find(cache_key);
        cached != expert_cache_.end()) return &cached->second;

    const size_t rows = static_cast<size_t>(2 * moe_intermediate);
    const std::vector<int64_t> shape = {moe_intermediate, hidden};
    /// Checkpoint-packed int4 tensors are addressed by a virtual base name
    /// the repository does not contain directly, so probe that convention
    /// before fetching tensor views.
    const bool packed_int4 =
        has_packed_int4_matrix(repo, names.gate.front()) ||
        has_packed_int4_matrix(repo, names.up.front());
    const HostTensorView first_gate = packed_int4
        ? HostTensorView{} : repo.tensor(names.gate.front());
    const HostTensorView first_up = packed_int4
        ? HostTensorView{} : repo.tensor(names.up.front());

    if (first_gate.dtype == TensorDType::Quantized ||
        first_up.dtype == TensorDType::Quantized) {
        const GgmlType first_gate_type =
            ggml_type_from_block_encoding(first_gate.block_encoding);
        const GgmlType first_up_type =
            ggml_type_from_block_encoding(first_up.block_encoding);
        if (first_gate.dtype != TensorDType::Quantized ||
            first_up.dtype != TensorDType::Quantized ||
            first_gate_type != first_up_type ||
            first_gate.shape != shape || first_up.shape != shape) {
            throw std::runtime_error("incompatible quantized MoE gate/up tensors");
        }
        if (is_host_decoded_moe_type(first_gate_type)) {
            const size_t per_expert = rows * static_cast<size_t>(hidden);
            DeviceWeight weight;
            weight.shape = {num_experts, static_cast<int>(rows), hidden};
            weight.bf16_storage.reset(static_cast<size_t>(num_experts) * per_expert);
            for (int e = 0; e < num_experts; ++e) {
                const HostTensorView gate = repo.tensor(names.gate[static_cast<size_t>(e)]);
                const HostTensorView up = repo.tensor(names.up[static_cast<size_t>(e)]);
                if (gate.dtype != TensorDType::Quantized ||
                    up.dtype != TensorDType::Quantized ||
                    ggml_type_from_block_encoding(gate.block_encoding) != first_gate_type ||
                    ggml_type_from_block_encoding(up.block_encoding) != first_gate_type ||
                    gate.shape != shape || up.shape != shape) {
                    throw std::runtime_error("inconsistent quantized MoE gate/up tensor");
                }
                std::vector<__nv_bfloat16> gate_values;
                std::vector<__nv_bfloat16> up_values;
                dequantize_gguf_to_bf16(gate, gate_values);
                dequantize_gguf_to_bf16(up, up_values);
                __nv_bfloat16* dst = weight.bf16_storage.data() +
                    static_cast<size_t>(e) * per_expert;
                CELEG_CUDA(cudaMemcpy(dst, gate_values.data(),
                                      gate_values.size() * sizeof(__nv_bfloat16),
                                       cudaMemcpyHostToDevice));
                CELEG_CUDA(cudaMemcpy(dst + static_cast<size_t>(moe_intermediate) * hidden,
                                      up_values.data(),
                                      up_values.size() * sizeof(__nv_bfloat16),
                                      cudaMemcpyHostToDevice));
            }
            ExpertLinearWeight view;
            view.kind = ExpertStorageKind::Bf16;
            view.bf16 = weight.bf16_storage.data();
            view.experts = num_experts;
            view.rows_per_expert = static_cast<int>(rows);
            view.cols = hidden;
            auto [it, inserted] = weights_->tensors.emplace(cache_key, std::move(weight));
            if (!inserted) throw std::runtime_error("duplicate expert weight: " + cache_key);
            expert_cache_.emplace(cache_key, view);
            return &expert_cache_.find(cache_key)->second;
        }
        const size_t row_bytes = gguf_row_bytes(hidden, first_gate_type,
                                                names.gate.front());
        const size_t source_bytes = static_cast<size_t>(moe_intermediate) * row_bytes;
        const size_t expert_bytes = rows * row_bytes;
        DeviceWeight weight;
        weight.shape = {num_experts, static_cast<int>(rows), hidden};
        weight.gguf_expert_storage.reset(static_cast<size_t>(num_experts) * expert_bytes);
        for (int e = 0; e < num_experts; ++e) {
            const HostTensorView gate = repo.tensor(names.gate[static_cast<size_t>(e)]);
            const HostTensorView up = repo.tensor(names.up[static_cast<size_t>(e)]);
            if (gate.dtype != TensorDType::Quantized ||
                up.dtype != TensorDType::Quantized ||
                ggml_type_from_block_encoding(gate.block_encoding) != first_gate_type ||
                ggml_type_from_block_encoding(up.block_encoding) != first_gate_type ||
                gate.shape != shape || up.shape != shape ||
                gate.bytes != source_bytes || up.bytes != source_bytes) {
                throw std::runtime_error("inconsistent quantized MoE gate/up tensor");
            }
            uint8_t* dst = weight.gguf_expert_storage.data() +
                static_cast<size_t>(e) * expert_bytes;
            CELEG_CUDA(cudaMemcpy(dst, gate.data, gate.bytes, cudaMemcpyHostToDevice));
            CELEG_CUDA(cudaMemcpy(dst + gate.bytes, up.data, up.bytes,
                                cudaMemcpyHostToDevice));
        }
        const ExpertStorageKind kind = first_gate_type == GgmlType::Q4_K
            ? ExpertStorageKind::Q4_K : ExpertStorageKind::Q6_K;
        ExpertLinearWeight view;
        view.kind = kind;
        view.gguf_blocks = weight.gguf_expert_storage.data();
        view.gguf_type = first_gate_type;
        view.gguf_row_bytes = row_bytes;
        view.gguf_expert_stride = expert_bytes;
        view.experts = num_experts;
        view.rows_per_expert = static_cast<int>(rows);
        view.cols = hidden;
        auto [it, inserted] = weights_->tensors.emplace(cache_key, std::move(weight));
        if (!inserted) throw std::runtime_error("duplicate expert weight: " + cache_key);
        expert_cache_.emplace(cache_key, view);
        return &expert_cache_.find(cache_key)->second;
    }

    const size_t per_expert = rows * static_cast<size_t>(hidden);
    DeviceWeight weight;
    weight.shape = {num_experts, static_cast<int>(rows), hidden};
    weight.bf16_storage.reset(static_cast<size_t>(num_experts) * per_expert);
    const size_t projection_bytes = static_cast<size_t>(moe_intermediate) * hidden *
        sizeof(__nv_bfloat16);
    for (int e = 0; e < num_experts; ++e) {
        const std::vector<__nv_bfloat16> gate = load_bf16_or_dequantized(
            repo, names.gate[static_cast<size_t>(e)], shape);
        const std::vector<__nv_bfloat16> up = load_bf16_or_dequantized(
            repo, names.up[static_cast<size_t>(e)], shape);
        __nv_bfloat16* dst = weight.bf16_storage.data() +
            static_cast<size_t>(e) * per_expert;
        CELEG_CUDA(cudaMemcpy(dst, gate.data(), projection_bytes,
                              cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dst + static_cast<size_t>(moe_intermediate) * hidden,
                              up.data(), projection_bytes, cudaMemcpyHostToDevice));
    }
    ExpertLinearWeight view;
    view.kind = ExpertStorageKind::Bf16;
    view.bf16 = weight.bf16_storage.data();
    view.experts = num_experts;
    view.rows_per_expert = static_cast<int>(rows);
    view.cols = hidden;
    auto [it, inserted] = weights_->tensors.emplace(cache_key, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate expert weight: " + cache_key);
    expert_cache_.emplace(cache_key, view);
    return &expert_cache_.find(cache_key)->second;
}

const ExpertLinearWeight* WeightLoader::load_moe_down(
    const IWeightRepository& repo, const MoeExpertTensorNames& names,
    int num_experts, int moe_intermediate, int hidden) {
    validate_expert_dimensions(num_experts, moe_intermediate, hidden, "down");
    validate_individual_names(names, num_experts, "MoE down");
    const std::string cache_key = names.down.front() + ".moe.down";
    if (const auto cached = expert_cache_.find(cache_key);
        cached != expert_cache_.end()) return &cached->second;

    const std::vector<int64_t> shape = {hidden, moe_intermediate};
    /// Same virtual-name convention as the gate/up loader: probe the int4
    /// packing before touching repository tensor views.
    const bool packed_int4 = has_packed_int4_matrix(repo, names.down.front());
    const HostTensorView first = packed_int4
        ? HostTensorView{} : repo.tensor(names.down.front());
    if (!packed_int4 && first.shape != shape) {
        throw std::runtime_error("invalid MoE down tensor");
    }
    if (first.dtype == TensorDType::Quantized) {
        const GgmlType first_type = ggml_type_from_block_encoding(first.block_encoding);
        if (is_host_decoded_moe_type(first_type)) {
            const size_t per_expert = static_cast<size_t>(hidden) * moe_intermediate;
            DeviceWeight weight;
            weight.shape = {num_experts, hidden, moe_intermediate};
            weight.bf16_storage.reset(static_cast<size_t>(num_experts) * per_expert);
            for (int e = 0; e < num_experts; ++e) {
                const HostTensorView tensor = repo.tensor(names.down[static_cast<size_t>(e)]);
                if (tensor.dtype != TensorDType::Quantized ||
                    ggml_type_from_block_encoding(tensor.block_encoding) != first_type ||
                    tensor.shape != shape) {
                    throw std::runtime_error("inconsistent quantized MoE down tensor");
                }
                std::vector<__nv_bfloat16> decoded;
                dequantize_gguf_to_bf16(tensor, decoded);
                CELEG_CUDA(cudaMemcpy(weight.bf16_storage.data() +
                                      static_cast<size_t>(e) * per_expert,
                                      decoded.data(), decoded.size() * sizeof(__nv_bfloat16),
                                      cudaMemcpyHostToDevice));
            }
            ExpertLinearWeight view;
            view.kind = ExpertStorageKind::Bf16;
            view.bf16 = weight.bf16_storage.data();
            view.experts = num_experts;
            view.rows_per_expert = hidden;
            view.cols = moe_intermediate;
            auto [it, inserted] = weights_->tensors.emplace(cache_key, std::move(weight));
            if (!inserted) throw std::runtime_error("duplicate expert weight: " + cache_key);
            expert_cache_.emplace(cache_key, view);
            return &expert_cache_.find(cache_key)->second;
        }
        const size_t row_bytes = gguf_row_bytes(moe_intermediate, first_type,
                                                names.down.front());
        const size_t expert_bytes = static_cast<size_t>(hidden) * row_bytes;
        DeviceWeight weight;
        weight.shape = {num_experts, hidden, moe_intermediate};
        weight.gguf_expert_storage.reset(static_cast<size_t>(num_experts) * expert_bytes);
        for (int e = 0; e < num_experts; ++e) {
            const HostTensorView tensor = repo.tensor(names.down[static_cast<size_t>(e)]);
            if (tensor.dtype != TensorDType::Quantized ||
                ggml_type_from_block_encoding(tensor.block_encoding) != first_type ||
                tensor.shape != shape ||
                tensor.bytes != expert_bytes) {
                throw std::runtime_error("inconsistent quantized MoE down tensor");
            }
            CELEG_CUDA(cudaMemcpy(weight.gguf_expert_storage.data() +
                                static_cast<size_t>(e) * expert_bytes,
                                tensor.data, tensor.bytes, cudaMemcpyHostToDevice));
        }
        ExpertLinearWeight view;
        view.kind = first_type == GgmlType::Q4_K
            ? ExpertStorageKind::Q4_K : ExpertStorageKind::Q6_K;
        view.gguf_blocks = weight.gguf_expert_storage.data();
        view.gguf_type = first_type;
        view.gguf_row_bytes = row_bytes;
        view.gguf_expert_stride = expert_bytes;
        view.experts = num_experts;
        view.rows_per_expert = hidden;
        view.cols = moe_intermediate;
        auto [it, inserted] = weights_->tensors.emplace(cache_key, std::move(weight));
        if (!inserted) throw std::runtime_error("duplicate expert weight: " + cache_key);
        expert_cache_.emplace(cache_key, view);
        return &expert_cache_.find(cache_key)->second;
    }

    const size_t per_expert = static_cast<size_t>(hidden) * moe_intermediate;
    const size_t bytes = per_expert * sizeof(__nv_bfloat16);
    DeviceWeight weight;
    weight.shape = {num_experts, hidden, moe_intermediate};
    weight.bf16_storage.reset(static_cast<size_t>(num_experts) * per_expert);
    for (int e = 0; e < num_experts; ++e) {
        const std::vector<__nv_bfloat16> tensor = load_bf16_or_dequantized(
            repo, names.down[static_cast<size_t>(e)], shape);
        CELEG_CUDA(cudaMemcpy(weight.bf16_storage.data() +
                              static_cast<size_t>(e) * per_expert,
                              tensor.data(), bytes, cudaMemcpyHostToDevice));
    }
    ExpertLinearWeight view;
    view.kind = ExpertStorageKind::Bf16;
    view.bf16 = weight.bf16_storage.data();
    view.experts = num_experts;
    view.rows_per_expert = hidden;
    view.cols = moe_intermediate;
    auto [it, inserted] = weights_->tensors.emplace(cache_key, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate expert weight: " + cache_key);
    expert_cache_.emplace(cache_key, view);
    return &expert_cache_.find(cache_key)->second;
}

}
