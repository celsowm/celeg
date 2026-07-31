#include "celeg/model/weights/layout.hpp"
#include "celeg/backend/cuda/kernels/embedding.hpp"
#include "celeg/backend/cuda/kernels/gguf.cuh"
#include "celeg/model/execution/runtime_types.hpp"
#include <stdexcept>

namespace celeg {

namespace {

class Bf16WeightLayout final : public IWeightLayout {
public:
    Bf16WeightLayout(const __nv_bfloat16* table, const float* /*scales*/)
        : table_(table) {}

    void embed_token(int32_t token, __nv_bfloat16* out, int hidden,
                    cudaStream_t stream) override {
        launch_embedding(token, table_, out, hidden, stream);
    }
    void embed_batch(const int32_t* tokens, int rows, __nv_bfloat16* out,
                    int hidden, cudaStream_t stream) override {
        launch_embedding_batch(tokens, rows, table_, out, hidden, stream);
    }
    void embed_token_device(const int32_t* token, __nv_bfloat16* out,
                           int hidden, cudaStream_t stream) override {
        launch_embedding_device(token, table_, out, hidden, stream);
    }

private:
    const __nv_bfloat16* table_;
};

class Int8WeightLayout final : public IWeightLayout {
public:
    Int8WeightLayout(const int8_t* table, const float* scales)
        : table_(table), scales_(scales) {}

    void embed_token(int32_t token, __nv_bfloat16* out, int hidden,
                    cudaStream_t stream) override {
        launch_embedding_int8(token, table_, scales_, out, hidden, stream);
    }
    void embed_batch(const int32_t* tokens, int rows, __nv_bfloat16* out,
                    int hidden, cudaStream_t stream) override {
        launch_embedding_int8_batch(tokens, rows, table_, scales_,
                                    out, hidden, stream);
    }
    void embed_token_device(const int32_t* token, __nv_bfloat16* out,
                           int hidden, cudaStream_t stream) override {
        launch_embedding_int8_device(token, table_, scales_, out, hidden,
                                     stream);
    }

private:
    const int8_t* table_;
    const float* scales_;
};

class Int4WeightLayout final : public IWeightLayout {
public:
    Int4WeightLayout(const uint8_t* table, const float* scales)
        : table_(table), scales_(scales) {}

    void embed_token(int32_t token, __nv_bfloat16* out, int hidden,
                    cudaStream_t stream) override {
        launch_embedding_int4(token, table_, scales_, out, hidden, stream);
    }
    void embed_batch(const int32_t* tokens, int rows, __nv_bfloat16* out,
                    int hidden, cudaStream_t stream) override {
        launch_embedding_int4_batch(tokens, rows, table_, scales_,
                                    out, hidden, stream);
    }
    void embed_token_device(const int32_t* token, __nv_bfloat16* out,
                           int hidden, cudaStream_t stream) override {
        launch_embedding_int4_device(token, table_, scales_, out, hidden,
                                     stream);
    }

private:
    const uint8_t* table_;
    const float* scales_;
};

class GgufWeightLayout final : public IWeightLayout {
public:
    explicit GgufWeightLayout(GgufLinearSegment segment)
        : segment_(segment) {}

    void embed_token(int32_t token, __nv_bfloat16* out, int hidden,
                     cudaStream_t stream) override {
        if (hidden != segment_.cols) throw std::runtime_error("GGUF embedding width mismatch");
        launch_gguf_embedding(token, segment_, out, stream);
    }
    void embed_batch(const int32_t* tokens, int rows, __nv_bfloat16* out,
                     int hidden, cudaStream_t stream) override {
        if (hidden != segment_.cols) throw std::runtime_error("GGUF embedding width mismatch");
        launch_gguf_embedding_batch(tokens, rows, segment_, out, stream);
    }
    void embed_token_device(const int32_t* token, __nv_bfloat16* out,
                            int hidden, cudaStream_t stream) override {
        if (hidden != segment_.cols) throw std::runtime_error("GGUF embedding width mismatch");
        launch_gguf_embedding_device(token, segment_, out, stream);
    }

private:
    GgufLinearSegment segment_;
};

} // namespace

std::unique_ptr<IWeightLayout> make_weight_layout(
    WeightMode weight_mode,
    const void* table,
    const float* scales) {
    switch (weight_mode) {
        case WeightMode::Int8:
            return std::make_unique<Int8WeightLayout>(
                static_cast<const int8_t*>(table), scales);
        case WeightMode::Int4:
            return std::make_unique<Int4WeightLayout>(
                static_cast<const uint8_t*>(table), scales);
        default:
            return std::make_unique<Bf16WeightLayout>(
                static_cast<const __nv_bfloat16*>(table), scales);
    }
}

std::unique_ptr<IWeightLayout> make_gguf_weight_layout(
    const GgufLinearSegment& segment) {
    return std::make_unique<GgufWeightLayout>(segment);
}

} // namespace celeg
