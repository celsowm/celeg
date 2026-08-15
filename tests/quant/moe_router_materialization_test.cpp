
#include "celeg/backend/cuda/weights_loader.hpp"
#include "celeg/detail/model/device_weights.hpp"
#include "celeg/detail/model/shared_weights.hpp"
#include "celeg/checkpoint/formats/safetensors.hpp"
#include "celeg/checkpoint/formats/gguf.hpp"

#include "support/assertions.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using celeg::DeviceWeight;
using celeg::IWeightRepository;
using celeg::LinearStorageKind;
using celeg::LinearWeight;
using celeg::SharedModelWeights;
using celeg::TensorDType;
using celeg::WeightLoader;
using celeg::WeightMode;
using celeg::HostTensorView;

class FakeRouterRepo : public IWeightRepository {
public:
    FakeRouterRepo(int experts, int hidden, const std::vector<__nv_bfloat16>& bytes)
        : experts_(experts), hidden_(hidden), bytes_(bytes) {}

    bool contains(std::string_view tensor) const override {
        return tensor == router_name(0);
    }
    HostTensorView tensor(std::string_view tensor) const override {
        if (!contains(tensor)) {
            throw std::runtime_error("FakeRouterRepo unknown tensor: " +
                                     std::string(tensor));
        }
        HostTensorView view{};
        view.dtype = TensorDType::BF16;
        view.shape = std::vector<int64_t>{static_cast<int64_t>(experts_),
                                         static_cast<int64_t>(hidden_)};
        view.data = reinterpret_cast<const std::byte*>(bytes_.data());
        view.bytes = bytes_.size() * sizeof(__nv_bfloat16);
        return view;
    }
    std::vector<std::string> names() const override { return {router_name(0)}; }

private:
    static std::string router_name(int layer) {
        return "model.layers." + std::to_string(layer) +
               ".feed_forward.gate.weight";
    }
    int experts_ = 0;
    int hidden_ = 0;
    std::vector<__nv_bfloat16> bytes_;
};

struct RouterMaterialization {
    LinearStorageKind kind;
    bool bf16_nonnull;
};

RouterMaterialization materialize_router(WeightMode mode, int experts, int hidden,
                                         const std::vector<__nv_bfloat16>& data,
                                         cudaStream_t /*stream*/) {
    auto shared = std::make_shared<SharedModelWeights>();
    WeightLoader loader(shared, mode);
    FakeRouterRepo repo(experts, hidden, data);

    const LinearWeight* router = loader.load_router_weight(repo, 0, experts, hidden);
    CELEG_TEST_CHECK(router != nullptr);
    CELEG_TEST_CHECK(router->rows == experts);
    CELEG_TEST_CHECK(router->cols == hidden);
    CELEG_TEST_CHECK(!router->quantized());
    RouterMaterialization result{
        router->kind,
        router->bf16 != nullptr,
    };
    CELEG_TEST_CHECK(result.kind == LinearStorageKind::Bf16);
    CELEG_TEST_CHECK(result.bf16_nonnull);
    return result;
}

class FakeQuantizedRouterRepo final : public IWeightRepository {
public:
    FakeQuantizedRouterRepo(int experts, int hidden,
                            std::vector<std::byte> blocks)
        : experts_(experts), hidden_(hidden), blocks_(std::move(blocks)) {}

    bool contains(std::string_view tensor) const override {
        return tensor == "model.layers.0.feed_forward.gate.weight";
    }

    HostTensorView tensor(std::string_view tensor) const override {
        if (!contains(tensor)) {
            throw std::runtime_error("FakeQuantizedRouterRepo unknown tensor");
        }
        HostTensorView view{};
        view.dtype = TensorDType::Quantized;
        view.shape = {experts_, hidden_};
        view.data = blocks_.data();
        view.bytes = blocks_.size();
        view.block_encoding = celeg::block_encoding_from_ggml_type(celeg::GgmlType::Q4_K);
        return view;
    }

    std::vector<std::string> names() const override {
        return {"model.layers.0.feed_forward.gate.weight"};
    }

private:
    int experts_ = 0;
    int hidden_ = 0;
    std::vector<std::byte> blocks_;
};

void verify_native_q4_router_is_materialized_as_bf16() {
    constexpr int experts = 1;
    constexpr int hidden = 256;
    const celeg::GgmlTypeTrait trait = celeg::ggml_type_trait(celeg::GgmlType::Q4_K);
    std::vector<std::byte> blocks(static_cast<size_t>(experts) * trait.type_size);

    const __half d = __float2half(1.0f);
    const __half dmin = __float2half(0.0f);
    std::memcpy(blocks.data(), &d, sizeof(d));
    std::memcpy(blocks.data() + sizeof(d), &dmin, sizeof(dmin));
    auto* bytes = reinterpret_cast<uint8_t*>(blocks.data());
    bytes[4] = 1;
    std::fill(bytes + 16, bytes + trait.type_size, uint8_t{1});

    auto weights = std::make_shared<SharedModelWeights>();
    WeightLoader loader(weights, WeightMode::NativeGguf);
    FakeQuantizedRouterRepo repo(experts, hidden, std::move(blocks));
    const LinearWeight* router = loader.load_router_weight(
        repo, 0, experts, hidden);
    CELEG_TEST_CHECK(router != nullptr);
    CELEG_TEST_CHECK(router->kind == LinearStorageKind::Bf16);
    CELEG_TEST_CHECK(router->bf16 != nullptr);
    CELEG_TEST_CHECK(!router->quantized());
}

}

int main() {
    constexpr int E = 4, H = 8;
    std::vector<__nv_bfloat16> data(static_cast<size_t>(E) * H);
    for (size_t i = 0; i < data.size(); ++i) {
        const float v = (static_cast<float>(i % 7) - 3.0f) * 0.25f;
        data[i] = __float2bfloat16(v);
    }

    for (auto mode : {WeightMode::Bf16, WeightMode::Int8, WeightMode::Int4,
                      WeightMode::NativeGguf}) {
        const auto m = materialize_router(mode, E, H, data, nullptr);
        const char* mode_name = mode == WeightMode::Bf16 ? "bf16" :
                                mode == WeightMode::Int8 ? "int8" :
                                mode == WeightMode::Int4 ? "int4" : "native_gguf";
        std::cout << "mode=" << mode_name
                  << " kind=" << static_cast<int>(m.kind)
                  << " bf16_nonnull=" << (m.bf16_nonnull ? "yes" : "no") << "\n";
        CELEG_TEST_CHECK(m.kind == LinearStorageKind::Bf16);
        CELEG_TEST_CHECK(m.bf16_nonnull);
    }

    verify_native_q4_router_is_materialized_as_bf16();

    std::cout << "moe_router_materialization_test: ok\n";
    return 0;
}
