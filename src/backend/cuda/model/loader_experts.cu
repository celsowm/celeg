#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/checkpoint/formats/gguf.hpp"
#include "celeg/checkpoint/tensor_names.hpp"
#include "celeg/checkpoint/packed/int4.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace celeg {

namespace {

std::string expert_name(int layer, int expert, const char* tensor) {
    return layer_name(layer, "feed_forward.experts." +
        std::to_string(expert) + "." + tensor + ".weight");
}

void validate_expert_dimensions(int layer, int experts, int intermediate, int hidden,
                               const char* projection) {
    if (experts <= 0 || intermediate <= 0 || hidden <= 0) {
        throw std::runtime_error("invalid MoE " + std::string(projection) +
                                 " dimensions for layer " + std::to_string(layer));
    }
}

size_t gguf_row_bytes(int columns, GgmlType type, const std::string& name) {
    if (type != GgmlType::Q4_K && type != GgmlType::Q6_K) {
        throw std::runtime_error("unsupported GGUF MoE quantization: " + name);
    }
    const GgmlTypeTrait trait = ggml_type_trait(type);
    if (columns <= 0 || columns % trait.block_size != 0) {
        throw std::runtime_error("GGUF MoE width is not block-aligned: " + name);
    }
    return static_cast<size_t>(columns) / static_cast<size_t>(trait.block_size) *
        trait.type_size;
}

bool is_host_decoded_moe_type(GgmlType type) {
    return type == GgmlType::Q2_K || type == GgmlType::Q3_K;
}

std::string named_expert_name(const std::string& prefix, int expert,
                              const std::string& projection) {
    return prefix + "." + std::to_string(expert) + "." + projection + ".weight";
}

void validate_named_bf16(const HostTensorView& tensor,
                         const std::vector<int64_t>& shape,
                         size_t bytes, const std::string& name) {
    if (tensor.dtype != TensorDType::BF16 || tensor.shape != shape ||
        tensor.bytes != bytes) {
        throw std::runtime_error("inconsistent BF16 named MoE tensor: " + name);
    }
}

std::vector<__nv_bfloat16> load_named_bf16(
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
    const IWeightRepository& repo, int layer,
    int num_experts, int moe_intermediate, int hidden) {
    const std::string cache_key = layer_name(layer, "moe.gate_up");
    if (const auto cached = expert_cache_.find(cache_key);
        cached != expert_cache_.end()) return &cached->second;
    validate_expert_dimensions(layer, num_experts, moe_intermediate, hidden, "gate_up");

    const size_t rows = static_cast<size_t>(2 * moe_intermediate);
    const std::string first_w1_name = expert_name(layer, 0, "w1");
    const std::string first_w3_name = expert_name(layer, 0, "w3");
    const HostTensorView first_w1 = repo.tensor(first_w1_name);
    const HostTensorView first_w3 = repo.tensor(first_w3_name);
    const std::vector<int64_t> shape = {moe_intermediate, hidden};

    if (first_w1.dtype == TensorDType::Quantized ||
        first_w3.dtype == TensorDType::Quantized) {
        const GgmlType first_w1_ggml_type = ggml_type_from_block_encoding(first_w1.block_encoding);
        const GgmlType first_w3_ggml_type = ggml_type_from_block_encoding(first_w3.block_encoding);
        if (first_w1.dtype != TensorDType::Quantized ||
            first_w3.dtype != TensorDType::Quantized ||
            first_w1_ggml_type != first_w3_ggml_type ||
            first_w1.shape != shape || first_w3.shape != shape) {
            throw std::runtime_error("incompatible quantized MoE gate/up tensors");
        }
        if (is_host_decoded_moe_type(first_w1_ggml_type)) {
            const size_t per_expert = rows * static_cast<size_t>(hidden);
            DeviceWeight weight;
            weight.shape = {num_experts, static_cast<int>(rows), hidden};
            weight.bf16_storage.reset(static_cast<size_t>(num_experts) * per_expert);
            for (int e = 0; e < num_experts; ++e) {
                const HostTensorView w1 = repo.tensor(expert_name(layer, e, "w1"));
                const HostTensorView w3 = repo.tensor(expert_name(layer, e, "w3"));
                if (w1.dtype != TensorDType::Quantized ||
                    w3.dtype != TensorDType::Quantized ||
                    ggml_type_from_block_encoding(w1.block_encoding) != first_w1_ggml_type ||
                    ggml_type_from_block_encoding(w3.block_encoding) != first_w1_ggml_type ||
                    w1.shape != shape || w3.shape != shape) {
                    throw std::runtime_error("inconsistent quantized MoE gate/up tensor");
                }
                std::vector<__nv_bfloat16> gate;
                std::vector<__nv_bfloat16> up;
                dequantize_gguf_to_bf16(w1, gate);
                dequantize_gguf_to_bf16(w3, up);
                __nv_bfloat16* dst = weight.bf16_storage.data() +
                    static_cast<size_t>(e) * per_expert;
                CELEG_CUDA(cudaMemcpy(dst, gate.data(), gate.size() * sizeof(__nv_bfloat16),
                                       cudaMemcpyHostToDevice));
                CELEG_CUDA(cudaMemcpy(dst + static_cast<size_t>(moe_intermediate) * hidden,
                                      up.data(), up.size() * sizeof(__nv_bfloat16),
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
        const size_t row_bytes = gguf_row_bytes(hidden, first_w1_ggml_type, first_w1_name);
        const size_t source_bytes = static_cast<size_t>(moe_intermediate) * row_bytes;
        const size_t expert_bytes = rows * row_bytes;
        DeviceWeight weight;
        weight.shape = {num_experts, static_cast<int>(rows), hidden};
        weight.gguf_expert_storage.reset(static_cast<size_t>(num_experts) * expert_bytes);
        for (int e = 0; e < num_experts; ++e) {
            const HostTensorView w1 = repo.tensor(expert_name(layer, e, "w1"));
            const HostTensorView w3 = repo.tensor(expert_name(layer, e, "w3"));
            if (w1.dtype != TensorDType::Quantized ||
                w3.dtype != TensorDType::Quantized ||
                ggml_type_from_block_encoding(w1.block_encoding) != first_w1_ggml_type ||
                ggml_type_from_block_encoding(w3.block_encoding) != first_w1_ggml_type ||
                w1.shape != shape || w3.shape != shape ||
                w1.bytes != source_bytes || w3.bytes != source_bytes) {
                throw std::runtime_error("inconsistent quantized MoE gate/up tensor");
            }
            uint8_t* dst = weight.gguf_expert_storage.data() +
                static_cast<size_t>(e) * expert_bytes;
            CELEG_CUDA(cudaMemcpy(dst, w1.data, w1.bytes, cudaMemcpyHostToDevice));
            CELEG_CUDA(cudaMemcpy(dst + w1.bytes, w3.data, w3.bytes,
                                cudaMemcpyHostToDevice));
        }
        const ExpertStorageKind kind = first_w1_ggml_type == GgmlType::Q4_K
            ? ExpertStorageKind::Q4_K : ExpertStorageKind::Q6_K;
        ExpertLinearWeight view;
        view.kind = kind;
        view.gguf_blocks = weight.gguf_expert_storage.data();
        view.gguf_type = first_w1_ggml_type;
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
    const size_t source_bytes = static_cast<size_t>(moe_intermediate) * hidden *
        sizeof(__nv_bfloat16);
    for (int e = 0; e < num_experts; ++e) {
        const HostTensorView w1 = repo.tensor(expert_name(layer, e, "w1"));
        const HostTensorView w3 = repo.tensor(expert_name(layer, e, "w3"));
        if (w1.dtype != TensorDType::BF16 || w3.dtype != TensorDType::BF16 ||
            w1.shape != shape || w3.shape != shape ||
            w1.bytes != source_bytes || w3.bytes != source_bytes) {
            throw std::runtime_error("inconsistent BF16 MoE gate/up tensor");
        }
        __nv_bfloat16* dst = weight.bf16_storage.data() +
            static_cast<size_t>(e) * per_expert;
        CELEG_CUDA(cudaMemcpy(dst, w1.data, source_bytes, cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(dst + static_cast<size_t>(moe_intermediate) * hidden,
                            w3.data, source_bytes, cudaMemcpyHostToDevice));
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
    const IWeightRepository& repo, int layer,
    int num_experts, int moe_intermediate, int hidden) {
    const std::string cache_key = layer_name(layer, "moe.down");
    if (const auto cached = expert_cache_.find(cache_key);
        cached != expert_cache_.end()) return &cached->second;
    validate_expert_dimensions(layer, num_experts, moe_intermediate, hidden, "down");

    const std::vector<int64_t> shape = {hidden, moe_intermediate};
    const std::string first_name = expert_name(layer, 0, "w2");
    const HostTensorView first = repo.tensor(first_name);
    if (first.shape != shape) throw std::runtime_error("invalid MoE down tensor");

    if (first.dtype == TensorDType::Quantized) {
        const GgmlType first_ggml_type = ggml_type_from_block_encoding(first.block_encoding);
        if (is_host_decoded_moe_type(first_ggml_type)) {
            const size_t per_expert = static_cast<size_t>(hidden) * moe_intermediate;
            DeviceWeight weight;
            weight.shape = {num_experts, hidden, moe_intermediate};
            weight.bf16_storage.reset(static_cast<size_t>(num_experts) * per_expert);
            for (int e = 0; e < num_experts; ++e) {
                const HostTensorView tensor = repo.tensor(expert_name(layer, e, "w2"));
                if (tensor.dtype != TensorDType::Quantized ||
                    ggml_type_from_block_encoding(tensor.block_encoding) != first_ggml_type ||
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
        const size_t row_bytes = gguf_row_bytes(moe_intermediate, first_ggml_type,
                                                first_name);
        const size_t expert_bytes = static_cast<size_t>(hidden) * row_bytes;
        DeviceWeight weight;
        weight.shape = {num_experts, hidden, moe_intermediate};
        weight.gguf_expert_storage.reset(static_cast<size_t>(num_experts) * expert_bytes);
        for (int e = 0; e < num_experts; ++e) {
            const HostTensorView tensor = repo.tensor(expert_name(layer, e, "w2"));
            if (tensor.dtype != TensorDType::Quantized ||
                ggml_type_from_block_encoding(tensor.block_encoding) != first_ggml_type ||
                tensor.shape != shape ||
                tensor.bytes != expert_bytes) {
                throw std::runtime_error("inconsistent quantized MoE down tensor");
            }
            CELEG_CUDA(cudaMemcpy(weight.gguf_expert_storage.data() +
                                static_cast<size_t>(e) * expert_bytes,
                                tensor.data, tensor.bytes, cudaMemcpyHostToDevice));
        }
        ExpertLinearWeight view;
        view.kind = first_ggml_type == GgmlType::Q4_K
            ? ExpertStorageKind::Q4_K : ExpertStorageKind::Q6_K;
        view.gguf_blocks = weight.gguf_expert_storage.data();
        view.gguf_type = first_ggml_type;
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

    if (first.dtype != TensorDType::BF16) {
        throw std::runtime_error("MoE down tensors must be BF16 or Q4_K/Q6_K");
    }
    const size_t per_expert = static_cast<size_t>(hidden) * moe_intermediate;
    const size_t expert_bytes = per_expert * sizeof(__nv_bfloat16);
    DeviceWeight weight;
    weight.shape = {num_experts, hidden, moe_intermediate};
    weight.bf16_storage.reset(static_cast<size_t>(num_experts) * per_expert);
    for (int e = 0; e < num_experts; ++e) {
        const HostTensorView tensor = repo.tensor(expert_name(layer, e, "w2"));
        if (tensor.dtype != TensorDType::BF16 || tensor.shape != shape ||
            tensor.bytes != expert_bytes) {
            throw std::runtime_error("inconsistent BF16 MoE down tensor");
        }
        CELEG_CUDA(cudaMemcpy(weight.bf16_storage.data() +
                            static_cast<size_t>(e) * per_expert,
                            tensor.data, tensor.bytes, cudaMemcpyHostToDevice));
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

const ExpertLinearWeight* WeightLoader::load_moe_gate_up_named(
    const IWeightRepository& repo, const std::string& experts_prefix,
    const std::string& gate_name, const std::string& up_name,
    int num_experts, int moe_intermediate, int hidden) {
    validate_expert_dimensions(0, num_experts, moe_intermediate, hidden, "named_gate_up");
    const std::string cache_key = experts_prefix + ".packed_gate_up";
    if (const auto cached = expert_cache_.find(cache_key);
        cached != expert_cache_.end()) return &cached->second;

    const size_t per_projection = static_cast<size_t>(moe_intermediate) * hidden;
    const size_t per_expert = 2 * per_projection;
    const size_t projection_bytes = per_projection * sizeof(__nv_bfloat16);
    DeviceWeight weight;
    weight.shape = {num_experts, 2 * moe_intermediate, hidden};
    weight.bf16_storage.reset(static_cast<size_t>(num_experts) * per_expert);
    for (int expert = 0; expert < num_experts; ++expert) {
        const std::string gate = named_expert_name(experts_prefix, expert, gate_name);
        const std::string up = named_expert_name(experts_prefix, expert, up_name);
        const std::vector<__nv_bfloat16> gate_tensor = load_named_bf16(
            repo, gate, {moe_intermediate, hidden});
        const std::vector<__nv_bfloat16> up_tensor = load_named_bf16(
            repo, up, {moe_intermediate, hidden});
        __nv_bfloat16* destination = weight.bf16_storage.data() +
            static_cast<size_t>(expert) * per_expert;
        CELEG_CUDA(cudaMemcpy(destination, gate_tensor.data(), projection_bytes,
                              cudaMemcpyHostToDevice));
        CELEG_CUDA(cudaMemcpy(destination + per_projection, up_tensor.data(),
                              projection_bytes, cudaMemcpyHostToDevice));
    }
    ExpertLinearWeight view;
    view.kind = ExpertStorageKind::Bf16;
    view.bf16 = weight.bf16_storage.data();
    view.experts = num_experts;
    view.rows_per_expert = 2 * moe_intermediate;
    view.cols = hidden;
    auto [it, inserted] = weights_->tensors.emplace(cache_key, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate named expert weight: " + cache_key);
    expert_cache_.emplace(cache_key, view);
    return &expert_cache_.find(cache_key)->second;
}

const ExpertLinearWeight* WeightLoader::load_moe_down_named(
    const IWeightRepository& repo, const std::string& experts_prefix,
    const std::string& down_name, int num_experts, int moe_intermediate,
    int hidden) {
    validate_expert_dimensions(0, num_experts, moe_intermediate, hidden, "named_down");
    const std::string cache_key = experts_prefix + ".packed_down";
    if (const auto cached = expert_cache_.find(cache_key);
        cached != expert_cache_.end()) return &cached->second;

    const size_t per_expert = static_cast<size_t>(hidden) * moe_intermediate;
    const size_t bytes = per_expert * sizeof(__nv_bfloat16);
    DeviceWeight weight;
    weight.shape = {num_experts, hidden, moe_intermediate};
    weight.bf16_storage.reset(static_cast<size_t>(num_experts) * per_expert);
    for (int expert = 0; expert < num_experts; ++expert) {
        const std::string name = named_expert_name(experts_prefix, expert, down_name);
        const std::vector<__nv_bfloat16> tensor = load_named_bf16(
            repo, name, {hidden, moe_intermediate});
        CELEG_CUDA(cudaMemcpy(weight.bf16_storage.data() +
                              static_cast<size_t>(expert) * per_expert,
                              tensor.data(), bytes, cudaMemcpyHostToDevice));
    }
    ExpertLinearWeight view;
    view.kind = ExpertStorageKind::Bf16;
    view.bf16 = weight.bf16_storage.data();
    view.experts = num_experts;
    view.rows_per_expert = hidden;
    view.cols = moe_intermediate;
    auto [it, inserted] = weights_->tensors.emplace(cache_key, std::move(weight));
    if (!inserted) throw std::runtime_error("duplicate named expert weight: " + cache_key);
    expert_cache_.emplace(cache_key, view);
    return &expert_cache_.find(cache_key)->second;
}

}
