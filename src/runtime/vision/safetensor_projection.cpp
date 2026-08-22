#include "celeg/runtime/vision/providers.hpp"

#include "celeg/checkpoint/weight_repository.hpp"
#include "celeg/checkpoint/repositories/safetensors.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "celeg/runtime/providers.hpp"
#include "celeg/vision/image.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace celeg {
namespace {


struct Tensor {
    HostTensorView view;

    float value(size_t index) const {
        if (index >= view.bytes / (view.dtype == TensorDType::F32 ? 4 : 2)) {
            throw std::out_of_range("Safetensor vision tensor index out of range");
        }
        if (view.dtype == TensorDType::F32) {
            return reinterpret_cast<const float*>(view.data)[index];
        }
        const uint16_t raw = reinterpret_cast<const uint16_t*>(view.data)[index];
        return view.dtype == TensorDType::BF16
            ? std::bit_cast<float>(static_cast<uint32_t>(raw) << 16)
            : fp16_bits_to_float(raw);
    }
};

std::pair<int, int> choose_grid(const ImageRgbF32& image) {
    const float aspect = static_cast<float>(image.width) / image.height;
    int wide = std::clamp(static_cast<int>(std::lround(std::sqrt(4.0f * std::max(aspect, 1.0f)))), 2, 8);
    int tall = std::clamp(static_cast<int>(std::lround(wide / std::max(aspect, 1.0f))), 2, 8);
    if (aspect < 1.0f) std::swap(wide, tall);
    if (wide & 1) --wide;
    if (tall & 1) --tall;
    return {std::max(2, tall), std::max(2, wide)};
}

void layer_norm(std::vector<float>& values, const Tensor& weight, const Tensor& bias,
                int rows, int width, float eps) {
    for (int row = 0; row < rows; ++row) {
        float* data = values.data() + static_cast<size_t>(row) * width;
        double sum = 0.0;
        for (int i = 0; i < width; ++i) sum += data[i];
        const float mean = static_cast<float>(sum / width);
        double variance = 0.0;
        for (int i = 0; i < width; ++i) {
            const float delta = data[i] - mean;
            variance += static_cast<double>(delta) * delta;
        }
        const float inv = 1.0f / std::sqrt(static_cast<float>(variance / width) + eps);
        for (int i = 0; i < width; ++i) data[i] = (data[i] - mean) * inv * weight.value(i) + bias.value(i);
    }
}

std::vector<float> linear(const Tensor& weight, const Tensor& bias,
                          const std::vector<float>& input, int rows, int in, int out) {
    std::vector<float> result(static_cast<size_t>(rows) * out);
    for (int row = 0; row < rows; ++row) for (int o = 0; o < out; ++o) {
        float sum = bias.view.data ? bias.value(o) : 0.0f;
        for (int i = 0; i < in; ++i) sum += weight.value(static_cast<size_t>(o) * in + i) *
                                                    input[static_cast<size_t>(row) * in + i];
        result[static_cast<size_t>(row) * out + o] = sum;
    }
    return result;
}

class SafetensorProjectionProvider final : public IVisualEmbeddingProvider {
public:
    explicit SafetensorProjectionProvider(std::shared_ptr<const IWeightRepository> repository)
        : repository_(std::move(repository)),
          patch_(tensor("model.visual.patch_embed.proj.weight")),
          patch_bias_(tensor("model.visual.patch_embed.proj.bias")),
          position_(tensor("model.visual.pos_embed.weight")),
          merger_norm_(tensor("model.visual.merger.norm.weight")),
          merger_bias_(tensor("model.visual.merger.norm.bias")),
          merger_fc1_(tensor("model.visual.merger.linear_fc1.weight")),
          merger_fc1_bias_(tensor("model.visual.merger.linear_fc1.bias")),
          merger_fc2_(tensor("model.visual.merger.linear_fc2.weight")),
          merger_fc2_bias_(tensor("model.visual.merger.linear_fc2.bias")) {
        hidden_ = static_cast<int>(patch_.view.shape.at(0));
        if (patch_.view.shape.size() != 5 || patch_.view.shape[1] != 3 ||
            patch_.view.shape[2] != 2 || patch_.view.shape[3] != 16 ||
            patch_.view.shape[4] != 16 || hidden_ != 1152 ||
            position_.view.shape.size() != 2 || position_.view.shape[1] != hidden_) {
            throw std::invalid_argument("invalid safetensor vision patch layout");
        }
        if (merger_fc2_.view.shape.size() != 2) {
            throw std::invalid_argument("invalid safetensor vision merger layout");
        }
        merger_out_ = static_cast<int>(merger_fc2_.view.shape.at(0));
        for (int layer = 0; repository_->contains("model.visual.blocks." +
                                                   std::to_string(layer) + ".norm1.weight"); ++layer) {
            blocks_.push_back(layer);
        }
        if (blocks_.size() != 27) throw std::invalid_argument("safetensor vision expects 27 blocks");
    }

    VisualEmbedding encode(std::string_view data_url) const override {
        const ImageRgbF32 source = decode_image_data_url(data_url);
        const auto [grid_h, grid_w] = choose_grid(source);
        const ImageRgbF32 image = resize_image_bilinear(source, grid_w * 16, grid_h * 16);
        const int rows = grid_h * grid_w;
        std::vector<float> tokens(static_cast<size_t>(rows) * hidden_);
        const size_t patch_size = 3 * 2 * 16 * 16;
        for (int y = 0; y < grid_h; ++y) for (int x = 0; x < grid_w; ++x) {
            const int row = y * grid_w + x;
            for (int o = 0; o < hidden_; ++o) {
                float sum = patch_bias_.value(o);
                for (int c = 0; c < 3; ++c) for (int py = 0; py < 16; ++py)
                    for (int px = 0; px < 16; ++px) {
                        const size_t pixel = (static_cast<size_t>(y * 16 + py) * image.width + x * 16 + px) * 3 + c;
                        for (int temporal = 0; temporal < 2; ++temporal) {
                            const size_t w = static_cast<size_t>(o) * patch_size +
                                static_cast<size_t>(c) * 2 * 16 * 16 +
                                static_cast<size_t>(temporal) * 16 * 16 + py * 16 + px;
                            sum += patch_.value(w) * (2.0f * image.rgb[pixel] - 1.0f);
                        }
                    }
                const float source_y = y * 47.0f / (grid_h - 1);
                const float source_x = x * 47.0f / (grid_w - 1);
                const int y0 = static_cast<int>(source_y);
                const int x0 = static_cast<int>(source_x);
                const int y1 = std::min(y0 + 1, 47);
                const int x1 = std::min(x0 + 1, 47);
                const float fy = source_y - y0;
                const float fx = source_x - x0;
                const auto pos = [&](int py, int px) {
                    return position_.value((static_cast<size_t>(py) * 48 + px) * hidden_ + o);
                };
                const float interpolated =
                    (1.0f - fy) * ((1.0f - fx) * pos(y0, x0) + fx * pos(y0, x1)) +
                    fy * ((1.0f - fx) * pos(y1, x0) + fx * pos(y1, x1));
                tokens[static_cast<size_t>(row) * hidden_ + o] = sum + interpolated;
            }
        }
        for (int layer : blocks_) run_block(tokens, layer, grid_h, grid_w, rows);
        layer_norm(tokens, merger_norm_, merger_bias_, rows, hidden_, 1.0e-6f);
        const int merged_h = grid_h / 2;
        const int merged_w = grid_w / 2;
        const int merged_rows = merged_h * merged_w;
        std::vector<float> merged(static_cast<size_t>(merged_rows) * hidden_ * 4);
        for (int y = 0; y < merged_h; ++y) for (int x = 0; x < merged_w; ++x)
            for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx)
                std::copy_n(tokens.data() + static_cast<size_t>((y * 2 + dy) * grid_w + x * 2 + dx) * hidden_,
                            hidden_, merged.data() + (static_cast<size_t>(y * merged_w + x) * 4 + dy * 2 + dx) * hidden_);
        std::vector<float> projected = linear(merger_fc1_, merger_fc1_bias_, merged,
                                              merged_rows, 4 * hidden_, 4 * hidden_);
        for (float& value : projected) value = 0.5f * value *
            (1.0f + std::tanh(0.7978845608f * (value + 0.044715f * value * value * value)));
        projected = linear(merger_fc2_, merger_fc2_bias_, projected,
                           merged_rows, 4 * hidden_, merger_out_);
        VisualEmbedding result{merger_out_, std::move(projected)};
        result.rope_positions.reserve(static_cast<size_t>(merged_rows));
        for (int y = 0; y < merged_h; ++y) for (int x = 0; x < merged_w; ++x)
            result.rope_positions.push_back({0, y, x});
        return result;
    }

private:
    Tensor tensor(std::string_view name) const {
        if (!repository_->contains(name)) throw std::invalid_argument("missing Safetensor vision tensor: " + std::string(name));
        return Tensor{repository_->tensor(name)};
    }

    void run_block(std::vector<float>& tokens, int layer, int grid_h, int grid_w, int rows) const {
        const std::string p = "model.visual.blocks." + std::to_string(layer) + ".";
        const Tensor n1 = tensor(p + "norm1.weight"), b1 = tensor(p + "norm1.bias");
        const Tensor qkv_w = tensor(p + "attn.qkv.weight"), qkv_b = tensor(p + "attn.qkv.bias");
        const Tensor proj_w = tensor(p + "attn.proj.weight"), proj_b = tensor(p + "attn.proj.bias");
        std::vector<float> residual = tokens;
        layer_norm(tokens, n1, b1, rows, hidden_, 1.0e-6f);
        std::vector<float> qkv = linear(qkv_w, qkv_b, tokens, rows, hidden_, hidden_ * 3);
        std::vector<float> attended(static_cast<size_t>(rows) * hidden_);
        constexpr int heads = 16;
        constexpr int head_dim = 72;
        const int half = head_dim / 2;
        for (int row = 0; row < rows; ++row) for (int head = 0; head < heads; ++head) {
            const int y = row / grid_w;
            const int x = row % grid_w;
            float* q = qkv.data() + static_cast<size_t>(row) * hidden_ * 3 + head * head_dim;
            float* k = qkv.data() + static_cast<size_t>(row) * hidden_ * 3 + hidden_ + head * head_dim;
            for (int i = 0; i < half; ++i) {
                const float angle_x = x * std::pow(10000.0f, -2.0f * i / head_dim);
                const float angle_y = y * std::pow(10000.0f, -2.0f * i / head_dim);
                const float cx = std::cos(angle_x), sx = std::sin(angle_x);
                const float cy = std::cos(angle_y), sy = std::sin(angle_y);
                const float qa = q[i], qb = q[half + i], ka = k[i], kb = k[half + i];
                q[i] = qa * cx - qb * sx; q[half + i] = qb * cx + qa * sx;
                k[i] = ka * cy - kb * sy; k[half + i] = kb * cy + ka * sy;
            }
            float maximum = -INFINITY;
            std::vector<float> scores(static_cast<size_t>(rows));
            for (int key = 0; key < rows; ++key) {
                const float* kval = qkv.data() + static_cast<size_t>(key) * hidden_ * 3 + hidden_ + head * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) score += q[d] * kval[d];
                scores[key] = score / std::sqrt(static_cast<float>(head_dim));
                maximum = std::max(maximum, scores[key]);
            }
            float denominator = 0.0f;
            for (float& score : scores) { score = std::exp(score - maximum); denominator += score; }
            for (int key = 0; key < rows; ++key) {
                const float* v = qkv.data() + static_cast<size_t>(key) * hidden_ * 3 + 2 * hidden_ + head * head_dim;
                for (int d = 0; d < head_dim; ++d)
                    attended[static_cast<size_t>(row) * hidden_ + head * head_dim + d] += scores[key] / denominator * v[d];
            }
        }
        std::vector<float> attention = linear(proj_w, proj_b, attended, rows, hidden_, hidden_);
        for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = residual[i] + attention[i];
        residual = tokens;
        const Tensor n2 = tensor(p + "norm2.weight"), b2 = tensor(p + "norm2.bias");
        layer_norm(tokens, n2, b2, rows, hidden_, 1.0e-6f);
        const Tensor fc1 = tensor(p + "mlp.linear_fc1.weight"), fc1_b = tensor(p + "mlp.linear_fc1.bias");
        const Tensor fc2 = tensor(p + "mlp.linear_fc2.weight"), fc2_b = tensor(p + "mlp.linear_fc2.bias");
        const int intermediate = static_cast<int>(fc1.view.shape[0]);
        std::vector<float> feed_forward = linear(fc1, fc1_b, tokens, rows, hidden_, intermediate);
        for (float& value : feed_forward) value = 0.5f * value *
            (1.0f + std::tanh(0.7978845608f * (value + 0.044715f * value * value * value)));
        feed_forward = linear(fc2, fc2_b, feed_forward, rows, intermediate, hidden_);
        for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = residual[i] + feed_forward[i];
    }

    std::shared_ptr<const IWeightRepository> repository_;
    Tensor patch_, patch_bias_, position_, merger_norm_, merger_bias_, merger_fc1_, merger_fc1_bias_, merger_fc2_, merger_fc2_bias_;
    std::vector<int> blocks_;
    int hidden_ = 0;
    int merger_out_ = 0;
};

class SafetensorProjectionProviderFactory final : public IVisionProviderFactory {
public:
    std::string_view id() const override { return "safetensor-projection"; }
    bool supports(const std::filesystem::path& model_path) const override {
        if (!std::filesystem::is_directory(model_path) ||
            !std::filesystem::is_regular_file(model_path / "model.safetensors.index.json")) {
            return false;
        }
        try {
            const SafeTensorRepository repository(model_path);
            return repository.contains("model.visual.patch_embed.proj.weight");
        } catch (...) {
            return false;
        }
    }
    std::shared_ptr<const IVisualEmbeddingProvider> create(const std::filesystem::path& model_path) const override {
        return make_safetensor_visual_embedding_provider(model_path);
    }
};

}

VisualEmbeddingProvider make_safetensor_visual_embedding_provider(const std::filesystem::path& model_path) {
    auto repository = std::make_shared<SafeTensorRepository>(model_path);
    return make_safetensor_visual_embedding_provider(std::move(repository));
}

VisualEmbeddingProvider make_safetensor_visual_embedding_provider(
    std::shared_ptr<const IWeightRepository> repository) {
    if (!repository) throw std::invalid_argument("Safetensor vision repository is null");
    return std::make_shared<SafetensorProjectionProvider>(std::move(repository));
}

std::unique_ptr<IVisionProviderFactory> make_safetensor_vision_provider_factory() {
    return std::make_unique<SafetensorProjectionProviderFactory>();
}

}
