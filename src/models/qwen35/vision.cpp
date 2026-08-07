#include "celeg/models/qwen35/vision.hpp"

#include "celeg/checkpoint/repositories/safetensors.hpp"
#include "celeg/runtime/providers.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <objbase.h>
#include <shlwapi.h>
#include <wincodec.h>
#endif

namespace celeg {
namespace {

float half_to_float(uint16_t value) {
    const uint32_t sign = (value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fu;
    const uint32_t fraction = value & 0x3ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (fraction != 0) {
            float result = std::ldexp(static_cast<float>(fraction), -24);
            return (sign ? -result : result);
        }
        bits = sign;
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (fraction << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (fraction << 13);
    }
    return std::bit_cast<float>(bits);
}

struct Tensor {
    HostTensorView view;

    float value(size_t index) const {
        if (index >= view.bytes / (view.dtype == TensorDType::F32 ? 4 : 2)) {
            throw std::out_of_range("Qwen vision tensor index out of range");
        }
        if (view.dtype == TensorDType::F32) {
            return reinterpret_cast<const float*>(view.data)[index];
        }
        const uint16_t raw = reinterpret_cast<const uint16_t*>(view.data)[index];
        return view.dtype == TensorDType::BF16
            ? std::bit_cast<float>(static_cast<uint32_t>(raw) << 16)
            : half_to_float(raw);
    }
};

struct Image {
    int width = 0;
    int height = 0;
    std::vector<float> rgb;
};

std::vector<uint8_t> decode_base64(std::string_view encoded) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uint8_t> result;
    int value = 0;
    int bits = -8;
    for (char byte : encoded) {
        if (byte == '=') break;
        const char* found = std::find(std::begin(alphabet), std::end(alphabet) - 1, byte);
        if (found == std::end(alphabet) - 1) continue;
        value = (value << 6) + static_cast<int>(found - alphabet);
        bits += 6;
        if (bits >= 0) {
            result.push_back(static_cast<uint8_t>((value >> bits) & 0xff));
            bits -= 8;
        }
    }
    if (result.empty()) throw std::invalid_argument("image data URL has no payload");
    return result;
}

Image decode_ppm(std::vector<uint8_t> bytes) {
    size_t cursor = 0;
    auto next = [&]() {
        while (cursor < bytes.size() && bytes[cursor] <= ' ') ++cursor;
        const size_t begin = cursor;
        while (cursor < bytes.size() && bytes[cursor] > ' ') ++cursor;
        if (begin == cursor) throw std::invalid_argument("invalid PPM header");
        return std::string(reinterpret_cast<const char*>(bytes.data() + begin), cursor - begin);
    };
    if (next() != "P6") throw std::invalid_argument("Qwen vision currently accepts P6 PPM input");
    const int width = std::stoi(next());
    const int height = std::stoi(next());
    if (std::stoi(next()) != 255 || width <= 0 || height <= 0) {
        throw std::invalid_argument("invalid PPM dimensions or range");
    }
    while (cursor < bytes.size() && bytes[cursor] <= ' ') ++cursor;
    const size_t count = static_cast<size_t>(width) * height * 3;
    if (cursor + count > bytes.size()) throw std::invalid_argument("truncated PPM pixels");
    Image image{width, height, std::vector<float>(count)};
    for (size_t i = 0; i < count; ++i) image.rgb[i] = bytes[cursor + i] / 255.0f;
    return image;
}

Image resize_grid(const Image& source, int width, int height) {
    Image result{width, height, std::vector<float>(static_cast<size_t>(width) * height * 3)};
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const float source_x = (x + 0.5f) * source.width / width - 0.5f;
        const float source_y = (y + 0.5f) * source.height / height - 0.5f;
        const int x0 = std::clamp(static_cast<int>(std::floor(source_x)), 0, source.width - 1);
        const int y0 = std::clamp(static_cast<int>(std::floor(source_y)), 0, source.height - 1);
        const int x1 = std::min(x0 + 1, source.width - 1);
        const int y1 = std::min(y0 + 1, source.height - 1);
        const float fx = std::clamp(source_x - std::floor(source_x), 0.0f, 1.0f);
        const float fy = std::clamp(source_y - std::floor(source_y), 0.0f, 1.0f);
        for (int c = 0; c < 3; ++c) {
            const float top = (1.0f - fx) * source.rgb[(static_cast<size_t>(y0) * source.width + x0) * 3 + c] +
                              fx * source.rgb[(static_cast<size_t>(y0) * source.width + x1) * 3 + c];
            const float bottom = (1.0f - fx) * source.rgb[(static_cast<size_t>(y1) * source.width + x0) * 3 + c] +
                                 fx * source.rgb[(static_cast<size_t>(y1) * source.width + x1) * 3 + c];
            result.rgb[(static_cast<size_t>(y) * width + x) * 3 + c] =
                (1.0f - fy) * top + fy * bottom;
        }
    }
    return result;
}

#if defined(_WIN32)
Image decode_wic(const std::vector<uint8_t>& bytes) {
    if (bytes.size() > static_cast<size_t>(std::numeric_limits<ULONG>::max())) {
        throw std::invalid_argument("image payload is too large");
    }
    const HRESULT initialized = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const bool uninitialize = SUCCEEDED(initialized);
    IWICImagingFactory* factory = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICFormatConverter* converter = nullptr;
    IStream* stream = SHCreateMemStream(bytes.data(), static_cast<UINT>(bytes.size()));
    auto cleanup = [&]() {
        if (converter) converter->Release();
        if (frame) frame->Release();
        if (decoder) decoder->Release();
        if (factory) factory->Release();
        if (stream) stream->Release();
        if (uninitialize) CoUninitialize();
    };
    if (!stream || FAILED(CoCreateInstance(CLSID_WICImagingFactory, nullptr,
                                            CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory))) ||
        FAILED(factory->CreateDecoderFromStream(stream, nullptr,
                                                 WICDecodeMetadataCacheOnLoad, &decoder)) ||
        FAILED(decoder->GetFrame(0, &frame)) || FAILED(factory->CreateFormatConverter(&converter)) ||
        FAILED(converter->Initialize(frame, GUID_WICPixelFormat24bppRGB,
                                     WICBitmapDitherTypeNone, nullptr, 0.0,
                                     WICBitmapPaletteTypeCustom))) {
        cleanup();
        throw std::invalid_argument("Windows image decoder rejected the image");
    }
    UINT width = 0, height = 0;
    if (FAILED(converter->GetSize(&width, &height)) || width == 0 || height == 0) {
        cleanup();
        throw std::invalid_argument("decoded image has invalid dimensions");
    }
    Image image{static_cast<int>(width), static_cast<int>(height),
                std::vector<float>(static_cast<size_t>(width) * height * 3)};
    std::vector<uint8_t> pixels(image.rgb.size());
    if (FAILED(converter->CopyPixels(nullptr, width * 3, static_cast<UINT>(pixels.size()), pixels.data()))) {
        cleanup();
        throw std::invalid_argument("Windows image decoder failed to read pixels");
    }
    for (size_t i = 0; i < image.rgb.size(); ++i) image.rgb[i] = pixels[i] / 255.0f;
    cleanup();
    return image;
}
#endif

Image decode_image(std::string_view data_url) {
    const size_t comma = data_url.find(',');
    if (!data_url.starts_with("data:") || comma == std::string_view::npos ||
        data_url.substr(0, comma).find(";base64") == std::string_view::npos) {
        throw std::invalid_argument("Qwen vision input must be a base64 data URL");
    }
    const std::vector<uint8_t> bytes = decode_base64(data_url.substr(comma + 1));
    if (bytes.size() >= 2 && bytes[0] == 'P' && bytes[1] == '6') return decode_ppm(bytes);
#if defined(_WIN32)
    return decode_wic(bytes);
#else
    throw std::invalid_argument("PNG and JPEG decoding requires the Windows image codec backend");
#endif
}

std::pair<int, int> choose_grid(const Image& image) {
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

class Qwen35VisionProvider final : public IVisualEmbeddingProvider {
public:
    explicit Qwen35VisionProvider(const std::filesystem::path& path)
        : repository_(std::make_shared<SafeTensorRepository>(path)),
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
            throw std::invalid_argument("invalid Qwen3.5 vision patch layout");
        }
        for (int layer = 0; repository_->contains("model.visual.blocks." +
                                                   std::to_string(layer) + ".norm1.weight"); ++layer) {
            blocks_.push_back(layer);
        }
        if (blocks_.size() != 27) throw std::invalid_argument("Qwen3.5 vision expects 27 blocks");
    }

    VisualEmbedding encode(std::string_view data_url) const override {
        const Image source = decode_image(data_url);
        const auto [grid_h, grid_w] = choose_grid(source);
        const Image image = resize_grid(source, grid_w * 16, grid_h * 16);
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
                           merged_rows, 4 * hidden_, 2048);
        VisualEmbedding result{2048, std::move(projected)};
        result.rope_positions.reserve(static_cast<size_t>(merged_rows));
        for (int y = 0; y < merged_h; ++y) for (int x = 0; x < merged_w; ++x)
            result.rope_positions.push_back({0, y, x});
        return result;
    }

private:
    Tensor tensor(std::string_view name) const {
        if (!repository_->contains(name)) throw std::invalid_argument("missing Qwen vision tensor: " + std::string(name));
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

    std::shared_ptr<SafeTensorRepository> repository_;
    Tensor patch_, patch_bias_, position_, merger_norm_, merger_bias_, merger_fc1_, merger_fc1_bias_, merger_fc2_, merger_fc2_bias_;
    std::vector<int> blocks_;
    int hidden_ = 0;
};

class Qwen35VisionProviderFactory final : public IVisionProviderFactory {
public:
    std::string_view id() const override { return "qwen35"; }
    bool supports(std::string_view architecture_id, const std::filesystem::path& model_path) const override {
        return architecture_id == "qwen35" && std::filesystem::is_directory(model_path) &&
               std::filesystem::is_regular_file(model_path / "model.safetensors.index.json");
    }
    std::shared_ptr<const IVisualEmbeddingProvider> create(const std::filesystem::path& model_path) const override {
        return make_qwen35_visual_embedding_provider(model_path);
    }
};

} // namespace

VisualEmbeddingProvider make_qwen35_visual_embedding_provider(const std::filesystem::path& model_path) {
    return std::make_shared<Qwen35VisionProvider>(model_path);
}

std::unique_ptr<IVisionProviderFactory> make_qwen35_vision_provider_factory() {
    return std::make_unique<Qwen35VisionProviderFactory>();
}

} // namespace celeg
