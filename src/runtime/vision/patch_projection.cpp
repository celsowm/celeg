#include "celeg/runtime/vision/providers.hpp"

#include "celeg/checkpoint/repositories/safetensors.hpp"
#include "celeg/checkpoint/weight_repository.hpp"
#include "celeg/model/weights/quantization.hpp"
#include "celeg/runtime/providers.hpp"
#include "celeg/vision/image.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
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

    std::size_t size() const {
        std::size_t result = 1;
        for (const std::int64_t dimension : view.shape) {
            if (dimension <= 0) throw std::invalid_argument("patch projection vision tensor has invalid shape");
            result *= static_cast<std::size_t>(dimension);
        }
        return result;
    }

    float value(std::size_t index) const {
        if (index >= size()) throw std::out_of_range("patch projection vision tensor index out of range");
        if (view.dtype == TensorDType::F32) return reinterpret_cast<const float*>(view.data)[index];
        const std::uint16_t raw = reinterpret_cast<const std::uint16_t*>(view.data)[index];
        if (view.dtype == TensorDType::BF16) {
            return std::bit_cast<float>(static_cast<std::uint32_t>(raw) << 16);
        }
        if (view.dtype == TensorDType::F16) return fp16_bits_to_float(raw);
        throw std::invalid_argument("patch projection vision weights must be F32, BF16, or F16");
    }
};

void require_shape(const Tensor& tensor, std::initializer_list<std::int64_t> expected,
                   std::string_view name) {
    if (tensor.view.shape.size() != expected.size() ||
        !std::equal(tensor.view.shape.begin(), tensor.view.shape.end(), expected.begin())) {
        throw std::invalid_argument("patch projection vision tensor has unexpected shape: " + std::string(name));
    }
}

std::pair<int, int> smart_grid(int height, int width) {
    constexpr int patch = 14;
    constexpr int merge = 2;
    constexpr int max_tokens = 4096;
    const double patch_height = static_cast<double>(height) / (patch * merge);
    const double patch_width = static_cast<double>(width) / (patch * merge);
    const double ratio = patch_height > 0.0 ? patch_width / patch_height : 1.0;
    double ideal_h = patch_height;
    double ideal_w = patch_width;
    if (ideal_h * ideal_w > max_tokens) {
        ideal_h = std::sqrt(max_tokens / ratio);
        ideal_w = ideal_h * ratio;
    }
    std::pair<int, int> selected{1, 1};
    double best = std::numeric_limits<double>::infinity();
    for (const int h : {std::max(1, static_cast<int>(std::floor(ideal_h))),
                        std::max(1, static_cast<int>(std::ceil(ideal_h)))}) {
        for (const int w : {std::max(1, static_cast<int>(std::floor(ideal_w))),
                            std::max(1, static_cast<int>(std::ceil(ideal_w)))}) {
            if (h * w > max_tokens) continue;
            const double error = std::abs(static_cast<double>(h) / w -
                                          static_cast<double>(height) / width);
            if (error < best) {
                best = error;
                selected = {h, w};
            }
        }
    }
    return {selected.first * merge, selected.second * merge};
}

void layer_norm(std::vector<float>& values, const Tensor& weight, const Tensor& bias,
                int rows, int width, float epsilon) {
    for (int row = 0; row < rows; ++row) {
        float* data = values.data() + static_cast<std::size_t>(row) * width;
        double mean = 0.0;
        for (int i = 0; i < width; ++i) mean += data[i];
        mean /= width;
        double variance = 0.0;
        for (int i = 0; i < width; ++i) {
            const double delta = data[i] - mean;
            variance += delta * delta;
        }
        const float inverse = 1.0f / std::sqrt(static_cast<float>(variance / width) + epsilon);
        for (int i = 0; i < width; ++i) {
            data[i] = (data[i] - static_cast<float>(mean)) * inverse * weight.value(i) + bias.value(i);
        }
    }
}

std::vector<float> linear(const Tensor& weight, const Tensor& bias,
                          const std::vector<float>& input, int rows, int input_width,
                          int output_width) {
    std::vector<float> result(static_cast<std::size_t>(rows) * output_width);
    for (int row = 0; row < rows; ++row) {
        for (int output = 0; output < output_width; ++output) {
            float sum = bias.view.data ? bias.value(output) : 0.0f;
            for (int input_index = 0; input_index < input_width; ++input_index) {
                sum += weight.value(static_cast<std::size_t>(output) * input_width + input_index) *
                       input[static_cast<std::size_t>(row) * input_width + input_index];
            }
            result[static_cast<std::size_t>(row) * output_width + output] = sum;
        }
    }
    return result;
}

float gelu(float value) {
    return 0.5f * value * (1.0f + std::erf(value * 0.7071067811865475f));
}

class PatchProjectionProvider final : public IVisualEmbeddingProvider {
public:
    explicit PatchProjectionProvider(std::shared_ptr<const IWeightRepository> repository)
        : repository_(std::move(repository)),
          patch_(tensor("model.vision_tower.patch_embedder.patch_embedding.weight")),
          position_(tensor("model.vision_tower.patch_embedder.position_embedding_table.weight")),
          ln_pre_(tensor("model.vision_tower.ln_pre.weight")),
          ln_pre_bias_(tensor("model.vision_tower.ln_pre.bias")),
          ln_post_(tensor("model.vision_tower.ln_post.weight")),
          ln_post_bias_(tensor("model.vision_tower.ln_post.bias")),
          adapter_fc1_(tensor("model.vision_adapter.fc1.weight")),
          adapter_fc2_(tensor("model.vision_adapter.fc2.weight")),
          projection_(tensor("model.vision_projection.weight")) {
        require_shape(patch_, {1536, 2 * 3 * 14 * 14}, "patch_embedding");
        require_shape(position_, {32 * 32, 1536}, "position_embedding_table");
        require_shape(ln_pre_, {1536}, "ln_pre.weight");
        require_shape(ln_pre_bias_, {1536}, "ln_pre.bias");
        require_shape(ln_post_, {1536}, "ln_post.weight");
        require_shape(ln_post_bias_, {1536}, "ln_post.bias");
        require_shape(adapter_fc1_, {4096, 6144}, "vision_adapter.fc1");
        require_shape(adapter_fc2_, {4096, 4096}, "vision_adapter.fc2");
        require_shape(projection_, {6656, 4096}, "vision_projection");
        for (int layer = 0; layer < 50; ++layer) {
            const std::string prefix = "model.vision_tower.layers." + std::to_string(layer) + ".";
            blocks_.push_back({
                tensor(prefix + "norm1.weight"), tensor(prefix + "norm1.bias"),
                tensor(prefix + "norm2.weight"), tensor(prefix + "norm2.bias"),
                tensor(prefix + "attn.q_proj.weight"), tensor(prefix + "attn.q_proj.bias"),
                tensor(prefix + "attn.k_proj.weight"), tensor(prefix + "attn.k_proj.bias"),
                tensor(prefix + "attn.v_proj.weight"), tensor(prefix + "attn.v_proj.bias"),
                tensor(prefix + "attn.proj.weight"), tensor(prefix + "attn.proj.bias"),
                tensor(prefix + "mlp.fc1.weight"), tensor(prefix + "mlp.fc1.bias"),
                tensor(prefix + "mlp.fc2.weight"), tensor(prefix + "mlp.fc2.bias")});
        }
    }

    VisualEmbedding encode(std::string_view data_url) const override {
        const ImageRgbF32 source = decode_image_data_url(data_url);
        const auto [grid_h, grid_w] = smart_grid(source.height, source.width);
        const ImageRgbF32 image = resize_image_bilinear(source, grid_w * 14, grid_h * 14);
        const int rows = grid_h * grid_w;
        std::vector<float> hidden(static_cast<std::size_t>(rows) * 1536);
        constexpr int patch_area = 14 * 14;
        for (int y = 0; y < grid_h; ++y) {
            for (int x = 0; x < grid_w; ++x) {
                const int row = y * grid_w + x;
                for (int output = 0; output < 1536; ++output) {
                    float sum = 0.0f;
                    for (int temporal = 0; temporal < 2; ++temporal) {
                        for (int channel = 0; channel < 3; ++channel) {
                            for (int py = 0; py < 14; ++py) {
                                for (int px = 0; px < 14; ++px) {
                                    const std::size_t pixel =
                                        (static_cast<std::size_t>(y * 14 + py) * image.width +
                                         x * 14 + px) * 3 + channel;
                                    const std::size_t weight =
                                        static_cast<std::size_t>(output) * 2 * 3 * patch_area +
                                        static_cast<std::size_t>(temporal) * 3 * patch_area +
                                        static_cast<std::size_t>(channel) * patch_area + py * 14 + px;
                                    sum += patch_.value(weight) * (2.0f * image.rgb[pixel] - 1.0f);
                                }
                            }
                        }
                    }
                    hidden[static_cast<std::size_t>(row) * 1536 + output] = sum;
                }
            }
        }
        add_interpolated_position(hidden, grid_h, grid_w);
        layer_norm(hidden, ln_pre_, ln_pre_bias_, rows, 1536, 1.0e-5f);

        const std::vector<int> window_index = make_window_index(grid_h, grid_w);
        std::vector<float> reordered(hidden.size());
        for (std::size_t index = 0; index < window_index.size(); ++index) {
            std::copy_n(hidden.data() + static_cast<std::size_t>(window_index[index]) * 1536,
                        1536, reordered.data() + index * 1536);
        }
        std::vector<std::pair<int, int>> positions;
        positions.reserve(window_index.size());
        for (const int index : window_index) {
            positions.push_back({index % grid_w + 1, index / grid_w + 1});
        }
        for (std::size_t layer = 0; layer < blocks_.size(); ++layer) {
            run_block(reordered, positions, make_attention_groups(grid_h, grid_w,
                      layer_types_[layer] == WindowAttention), blocks_[layer]);
        }
        std::vector<float> restored(hidden.size());
        for (std::size_t index = 0; index < window_index.size(); ++index) {
            std::copy_n(reordered.data() + index * 1536, 1536,
                        restored.data() + static_cast<std::size_t>(window_index[index]) * 1536);
        }
        layer_norm(restored, ln_post_, ln_post_bias_, rows, 1536, 1.0e-5f);
        std::vector<float> merged(static_cast<std::size_t>(rows / 4) * 6144);
        const int merged_w = grid_w / 2;
        for (int y = 0; y < grid_h / 2; ++y) {
            for (int x = 0; x < grid_w / 2; ++x) {
                const int output_row = y * merged_w + x;
                for (int dy = 0; dy < 2; ++dy) {
                    for (int dx = 0; dx < 2; ++dx) {
                        const int source_row = (y * 2 + dy) * grid_w + x * 2 + dx;
                        std::copy_n(restored.data() + static_cast<std::size_t>(source_row) * 1536,
                                    1536, merged.data() + (static_cast<std::size_t>(output_row) * 4 +
                                    dy * 2 + dx) * 1536);
                    }
                }
            }
        }
        std::vector<float> projected = linear(adapter_fc1_, {}, merged, rows / 4, 6144, 4096);
        for (float& value : projected) value = gelu(value);
        projected = linear(adapter_fc2_, {}, projected, rows / 4, 4096, 4096);
        for (float& value : projected) value = gelu(value);
        projected = linear(projection_, {}, projected, rows / 4, 4096, 6656);
        for (int row = 0; row < rows / 4; ++row) {
            float* data = projected.data() + static_cast<std::size_t>(row) * 6656;
            double sum = 0.0;
            for (int i = 0; i < 6656; ++i) sum += static_cast<double>(data[i]) * data[i];
            const float inverse = 1.0f / std::sqrt(static_cast<float>(sum / 6656.0) + 1.0e-5f);
            for (int i = 0; i < 6656; ++i) data[i] *= inverse;
        }
        return VisualEmbedding{6656, std::move(projected)};
    }

private:
    enum LayerType { WindowAttention, FullAttention };
    struct Block {
        Tensor norm1, norm1_bias, norm2, norm2_bias;
        Tensor q, q_bias, k, k_bias, v, v_bias, output, output_bias;
        Tensor fc1, fc1_bias, fc2, fc2_bias;
    };

    Tensor tensor(std::string_view name) const {
        if (!repository_->contains(name)) {
            throw std::invalid_argument("missing patch projection vision tensor: " + std::string(name));
        }
        return Tensor{repository_->tensor(name)};
    }

    static std::vector<int> make_window_index(int height, int width) {
        constexpr int window = 32;
        std::vector<int> result;
        for (int window_y = 0; window_y < height; window_y += window) {
            for (int window_x = 0; window_x < width; window_x += window) {
                for (int y = window_y; y < std::min(window_y + window, height); ++y) {
                    for (int x = window_x; x < std::min(window_x + window, width); ++x) {
                        result.push_back(y * width + x);
                    }
                }
            }
        }
        return result;
    }

    static std::vector<std::vector<int>> make_attention_groups(int height, int width,
                                                                 bool windowed) {
        if (!windowed) {
            std::vector<int> all(static_cast<std::size_t>(height) * width);
            for (int index = 0; index < static_cast<int>(all.size()); ++index) all[index] = index;
            return {std::move(all)};
        }
        constexpr int window = 32;
        std::vector<std::vector<int>> groups;
        int reordered_offset = 0;
        for (int begin_y = 0; begin_y < height; begin_y += window) {
            for (int begin_x = 0; begin_x < width; begin_x += window) {
                std::vector<int> group;
                for (int y = begin_y; y < std::min(begin_y + window, height); ++y) {
                    for (int x = begin_x; x < std::min(begin_x + window, width); ++x) {
                        group.push_back(reordered_offset++);
                    }
                }
                groups.push_back(std::move(group));
            }
        }
        return groups;
    }

    void add_interpolated_position(std::vector<float>& hidden, int height, int width) const {
        for (int y = 0; y < height; ++y) {
            const float source_y = (static_cast<float>(y) + 0.5f) * 32.0f / height - 0.5f;
            const int y_floor_raw = static_cast<int>(std::floor(source_y));
            const int y_ceil_raw = y_floor_raw + 1;
            const float y_fraction = source_y - y_floor_raw;
            const bool y_floor_valid = y_floor_raw >= 0 && y_floor_raw < 32;
            const bool y_ceil_valid = y_ceil_raw >= 0 && y_ceil_raw < 32;
            const int y_floor = std::clamp(y_floor_raw, 0, 31);
            const int y_ceil = std::clamp(y_ceil_raw, 0, 31);
            for (int x = 0; x < width; ++x) {
                const float source_x = (static_cast<float>(x) + 0.5f) * 32.0f / width - 0.5f;
                const int x_floor_raw = static_cast<int>(std::floor(source_x));
                const int x_ceil_raw = x_floor_raw + 1;
                const float x_fraction = source_x - x_floor_raw;
                const bool x_floor_valid = x_floor_raw >= 0 && x_floor_raw < 32;
                const bool x_ceil_valid = x_ceil_raw >= 0 && x_ceil_raw < 32;
                const int x_floor = std::clamp(x_floor_raw, 0, 31);
                const int x_ceil = std::clamp(x_ceil_raw, 0, 31);
                for (int d = 0; d < 1536; ++d) {
                    const auto sample = [&](int py, int px) {
                        return position_.value(static_cast<std::size_t>(py * 32 + px) * 1536 + d);
                    };
                    const float value =
                        (1.0f - y_fraction) * (1.0f - x_fraction) *
                            (y_floor_valid && x_floor_valid ? sample(y_floor, x_floor) : 0.0f) +
                        (1.0f - y_fraction) * x_fraction *
                            (y_floor_valid && x_ceil_valid ? sample(y_floor, x_ceil) : 0.0f) +
                        y_fraction * (1.0f - x_fraction) *
                            (y_ceil_valid && x_floor_valid ? sample(y_ceil, x_floor) : 0.0f) +
                        y_fraction * x_fraction *
                            (y_ceil_valid && x_ceil_valid ? sample(y_ceil, x_ceil) : 0.0f);
                    hidden[(static_cast<std::size_t>(y * width + x) * 1536) + d] += value;
                }
            }
        }
    }

    void run_block(std::vector<float>& hidden, const std::vector<std::pair<int, int>>& positions,
                   const std::vector<std::vector<int>>& groups, const Block& block) const {
        const int rows = static_cast<int>(positions.size());
        std::vector<float> normalized = hidden;
        layer_norm(normalized, block.norm1, block.norm1_bias, rows, 1536, 1.0e-5f);
        std::vector<float> query = linear(block.q, block.q_bias, normalized, rows, 1536, 1536);
        std::vector<float> key = linear(block.k, block.k_bias, normalized, rows, 1536, 1536);
        std::vector<float> value = linear(block.v, block.v_bias, normalized, rows, 1536, 1536);
        apply_rope(query, key, positions);
        std::vector<float> attended(static_cast<std::size_t>(rows) * 1536, 0.0f);
        constexpr int heads = 16;
        constexpr int head_dim = 96;
        for (const auto& group : groups) {
            for (const int query_row : group) {
                for (int head = 0; head < heads; ++head) {
                    float maximum = -std::numeric_limits<float>::infinity();
                    std::vector<float> scores(group.size());
                    for (std::size_t key_index = 0; key_index < group.size(); ++key_index) {
                        const int key_row = group[key_index];
                        float score = 0.0f;
                        for (int d = 0; d < head_dim; ++d) {
                            score += query[static_cast<std::size_t>(query_row) * 1536 + head * head_dim + d] *
                                     key[static_cast<std::size_t>(key_row) * 1536 + head * head_dim + d];
                        }
                        scores[key_index] = score / std::sqrt(96.0f);
                        maximum = std::max(maximum, scores[key_index]);
                    }
                    float denominator = 0.0f;
                    for (float& score : scores) {
                        score = std::exp(score - maximum);
                        denominator += score;
                    }
                    for (std::size_t key_index = 0; key_index < group.size(); ++key_index) {
                        const float coefficient = scores[key_index] / denominator;
                        const int key_row = group[key_index];
                        for (int d = 0; d < head_dim; ++d) {
                            attended[static_cast<std::size_t>(query_row) * 1536 + head * head_dim + d] +=
                                coefficient * value[static_cast<std::size_t>(key_row) * 1536 + head * head_dim + d];
                        }
                    }
                }
            }
        }
        std::vector<float> attention = linear(block.output, block.output_bias, attended, rows, 1536, 1536);
        for (std::size_t index = 0; index < hidden.size(); ++index) hidden[index] += attention[index];
        normalized = hidden;
        layer_norm(normalized, block.norm2, block.norm2_bias, rows, 1536, 1.0e-5f);
        std::vector<float> feed_forward = linear(block.fc1, block.fc1_bias, normalized, rows, 1536, 8960);
        for (float& value_item : feed_forward) value_item = gelu(value_item);
        feed_forward = linear(block.fc2, block.fc2_bias, feed_forward, rows, 8960, 1536);
        for (std::size_t index = 0; index < hidden.size(); ++index) hidden[index] += feed_forward[index];
    }

    static void apply_rope(std::vector<float>& query, std::vector<float>& key,
                           const std::vector<std::pair<int, int>>& positions) {
        constexpr int heads = 16;
        constexpr int head_dim = 96;
        constexpr int spatial_dim = head_dim / 2;
        for (std::size_t row = 0; row < positions.size(); ++row) {
            const float w = static_cast<float>(positions[row].first);
            const float h = static_cast<float>(positions[row].second);
            for (int head = 0; head < heads; ++head) {
                float cosines[spatial_dim];
                float sines[spatial_dim];
                for (int i = 0; i < spatial_dim / 2; ++i) {
                    const float inverse = 1.0f / std::pow(10000.0f,
                        static_cast<float>(2 * i) / spatial_dim);
                    cosines[i] = std::cos(w * inverse);
                    sines[i] = std::sin(w * inverse);
                    cosines[i + spatial_dim / 4] = std::cos(h * inverse);
                    sines[i + spatial_dim / 4] = std::sin(h * inverse);
                    cosines[i + spatial_dim / 2] = cosines[i];
                    sines[i + spatial_dim / 2] = sines[i];
                    cosines[i + 3 * spatial_dim / 4] = cosines[i + spatial_dim / 4];
                    sines[i + 3 * spatial_dim / 4] = sines[i + spatial_dim / 4];
                }
                for (int d = 0; d < spatial_dim; ++d) {
                    const float cosine = cosines[d];
                    const float sine = sines[d];
                    const std::size_t q_offset = row * 1536 + head * head_dim;
                    const std::size_t k_offset = row * 1536 + head * head_dim;
                    const float q0 = query[q_offset + d], q1 = query[q_offset + d + spatial_dim];
                    const float k0 = key[k_offset + d], k1 = key[k_offset + d + spatial_dim];
                    query[q_offset + d] = q0 * cosine - q1 * sine;
                    query[q_offset + d + spatial_dim] = q1 * cosine + q0 * sine;
                    key[k_offset + d] = k0 * cosine - k1 * sine;
                    key[k_offset + d + spatial_dim] = k1 * cosine + k0 * sine;
                }
            }
        }
    }

    std::shared_ptr<const IWeightRepository> repository_;
    Tensor patch_, position_, ln_pre_, ln_pre_bias_, ln_post_, ln_post_bias_;
    Tensor adapter_fc1_, adapter_fc2_, projection_;
    std::vector<Block> blocks_;
    const std::vector<LayerType> layer_types_ = {
        WindowAttention, WindowAttention, WindowAttention, FullAttention,
    };
};

class PatchProjectionProviderFactory final : public IVisionProviderFactory {
public:
    std::string_view id() const override { return "patch-projection"; }
    bool supports(const std::filesystem::path& model_path) const override {
        if (!std::filesystem::is_directory(model_path) ||
            !std::filesystem::is_regular_file(model_path / "model.safetensors.index.json")) {
            return false;
        }
        try {
            const SafeTensorRepository repository(model_path);
            return repository.contains(
                "model.vision_tower.patch_embedder.patch_embedding.weight");
        } catch (...) {
            return false;
        }
    }
    std::shared_ptr<const IVisualEmbeddingProvider> create(const std::filesystem::path& model_path) const override {
        return make_patch_visual_embedding_provider(model_path);
    }
};

}

VisualEmbeddingProvider make_patch_visual_embedding_provider(const std::filesystem::path& model_path) {
    return make_patch_visual_embedding_provider(
        std::make_shared<SafeTensorRepository>(model_path));
}

VisualEmbeddingProvider make_patch_visual_embedding_provider(
    std::shared_ptr<const IWeightRepository> repository) {
    if (!repository) throw std::invalid_argument("Patch projection vision repository is null");
    return std::make_shared<PatchProjectionProvider>(std::move(repository));
}

std::unique_ptr<IVisionProviderFactory> make_patch_vision_provider_factory() {
    return std::make_unique<PatchProjectionProviderFactory>();
}

}
