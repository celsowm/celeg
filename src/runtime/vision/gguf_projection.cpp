#include "celeg/runtime/vision/providers.hpp"
#include "celeg/runtime/providers.hpp"

#include "celeg/checkpoint/formats/gguf.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <iterator>
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

class GgufProjectionProviderFactory final : public IVisionProviderFactory {
public:
    std::string_view id() const override { return "gguf-projection"; }

    bool supports(const std::filesystem::path& model_path) const override {
        return std::filesystem::is_regular_file(model_path / "mmproj-BF16.gguf");
    }

    std::shared_ptr<const IVisualEmbeddingProvider> create(
        const std::filesystem::path& model_path) const override {
        return make_gguf_visual_embedding_provider(model_path / "mmproj-BF16.gguf");
    }
};

}
namespace {

struct Image {
    int width = 0;
    int height = 0;
    std::vector<float> rgb;
};

uint16_t read_u16(const std::vector<uint8_t>& bytes, size_t offset) {
    if (offset + 2 > bytes.size()) throw std::invalid_argument("truncated image");
    return static_cast<uint16_t>(bytes[offset]) |
        static_cast<uint16_t>(bytes[offset + 1]) << 8;
}

uint32_t read_u32(const std::vector<uint8_t>& bytes, size_t offset) {
    if (offset + 4 > bytes.size()) throw std::invalid_argument("truncated image");
    return static_cast<uint32_t>(bytes[offset]) |
        static_cast<uint32_t>(bytes[offset + 1]) << 8 |
        static_cast<uint32_t>(bytes[offset + 2]) << 16 |
        static_cast<uint32_t>(bytes[offset + 3]) << 24;
}

std::vector<uint8_t> decode_base64(std::string_view encoded) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uint8_t> result;
    int value = 0;
    int bits = -8;
    for (const char byte : encoded) {
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

Image decode_ppm(const std::vector<uint8_t>& bytes) {
    size_t cursor = 0;
    auto next = [&]() {
        while (cursor < bytes.size() && (bytes[cursor] == ' ' || bytes[cursor] == '\n' ||
                                         bytes[cursor] == '\r' || bytes[cursor] == '\t')) ++cursor;
        const size_t begin = cursor;
        while (cursor < bytes.size() && bytes[cursor] > ' ') ++cursor;
        if (begin == cursor) throw std::invalid_argument("invalid PPM header");
        return std::string(reinterpret_cast<const char*>(bytes.data() + begin), cursor - begin);
    };
    if (next() != "P6") throw std::invalid_argument("unsupported PPM format");
    const int width = std::stoi(next());
    const int height = std::stoi(next());
    const int max_value = std::stoi(next());
    if (width <= 0 || height <= 0 || max_value != 255) {
        throw std::invalid_argument("invalid PPM dimensions or range");
    }
    while (cursor < bytes.size() && bytes[cursor] <= ' ') ++cursor;
    const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
    if (cursor + count > bytes.size()) throw std::invalid_argument("truncated PPM pixels");
    Image image{width, height, std::vector<float>(count)};
    for (size_t index = 0; index < count; ++index) {
        image.rgb[index] = static_cast<float>(bytes[cursor + index]) / 255.0f;
    }
    return image;
}

Image decode_bmp(const std::vector<uint8_t>& bytes) {
    if (bytes.size() < 54 || bytes[0] != 'B' || bytes[1] != 'M') {
        throw std::invalid_argument("invalid BMP image");
    }
    const uint32_t pixel_offset = read_u32(bytes, 10);
    const int32_t width = static_cast<int32_t>(read_u32(bytes, 18));
    const int32_t signed_height = static_cast<int32_t>(read_u32(bytes, 22));
    const uint16_t planes = read_u16(bytes, 26);
    const uint16_t bits = read_u16(bytes, 28);
    const uint32_t compression = read_u32(bytes, 30);
    if (width <= 0 || signed_height == 0 || planes != 1 || compression != 0 ||
        (bits != 24 && bits != 32)) {
        throw std::invalid_argument("unsupported BMP layout");
    }
    const int height = std::abs(signed_height);
    const size_t source_stride = ((static_cast<size_t>(width) * bits + 31) / 32) * 4;
    if (pixel_offset + source_stride * static_cast<size_t>(height) > bytes.size()) {
        throw std::invalid_argument("truncated BMP pixels");
    }
    Image image{width, height, std::vector<float>(static_cast<size_t>(width) * height * 3)};
    for (int y = 0; y < height; ++y) {
        const int source_y = signed_height > 0 ? height - 1 - y : y;
        const uint8_t* row = bytes.data() + pixel_offset + source_stride * source_y;
        for (int x = 0; x < width; ++x) {
            const size_t source = static_cast<size_t>(x) * (bits / 8);
            const size_t target = (static_cast<size_t>(y) * width + x) * 3;
            image.rgb[target + 0] = row[source + 2] / 255.0f;
            image.rgb[target + 1] = row[source + 1] / 255.0f;
            image.rgb[target + 2] = row[source + 0] / 255.0f;
        }
    }
    return image;
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
                                            CLSCTX_INPROC_SERVER,
                                            IID_PPV_ARGS(&factory))) ||
        FAILED(factory->CreateDecoderFromStream(stream, nullptr,
                                                 WICDecodeMetadataCacheOnLoad, &decoder)) ||
        FAILED(decoder->GetFrame(0, &frame)) ||
        FAILED(factory->CreateFormatConverter(&converter)) ||
        FAILED(converter->Initialize(frame, GUID_WICPixelFormat24bppRGB,
                                     WICBitmapDitherTypeNone, nullptr, 0.0,
                                     WICBitmapPaletteTypeCustom))) {
        cleanup();
        throw std::invalid_argument("Windows image decoder rejected the image");
    }
    UINT width = 0;
    UINT height = 0;
    if (FAILED(converter->GetSize(&width, &height)) || width == 0 || height == 0) {
        cleanup();
        throw std::invalid_argument("decoded image has invalid dimensions");
    }
    Image image{static_cast<int>(width), static_cast<int>(height),
                std::vector<float>(static_cast<size_t>(width) * height * 3)};
    std::vector<uint8_t> pixels(image.rgb.size());
    if (FAILED(converter->CopyPixels(nullptr, width * 3, static_cast<UINT>(pixels.size()),
                                     pixels.data()))) {
        cleanup();
        throw std::invalid_argument("Windows image decoder failed to read pixels");
    }
    for (size_t index = 0; index < image.rgb.size(); index += 3) {
        image.rgb[index + 0] = pixels[index + 0] / 255.0f;
        image.rgb[index + 1] = pixels[index + 1] / 255.0f;
        image.rgb[index + 2] = pixels[index + 2] / 255.0f;
    }
    cleanup();
    return image;
}
#endif

Image decode_image(std::string_view data_url) {
    constexpr std::string_view prefix = "data:";
    if (!data_url.starts_with(prefix)) {
        throw std::invalid_argument("visual input must be a base64 data URL");
    }
    const size_t comma = data_url.find(',');
    if (comma == std::string_view::npos || data_url.substr(0, comma).find(";base64") ==
        std::string_view::npos) {
        throw std::invalid_argument("visual input data URL must use base64");
    }
    const std::vector<uint8_t> bytes = decode_base64(data_url.substr(comma + 1));
    if (bytes.size() >= 2 && bytes[0] == 'B' && bytes[1] == 'M') return decode_bmp(bytes);
    if (bytes.size() >= 2 && bytes[0] == 'P' && bytes[1] == '6') return decode_ppm(bytes);
#if defined(_WIN32)
    return decode_wic(bytes);
#else
    throw std::invalid_argument("PNG and JPEG decoding requires the Windows image codec backend");
#endif
}

Image resize_square(const Image& source, int size) {
    Image result{size, size, std::vector<float>(static_cast<size_t>(size) * size * 3)};
    for (int y = 0; y < size; ++y) {
        const float source_y = (y + 0.5f) * source.height / size - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(source_y)), 0, source.height - 1);
        const int y1 = std::min(y0 + 1, source.height - 1);
        const float fy = std::clamp(source_y - std::floor(source_y), 0.0f, 1.0f);
        for (int x = 0; x < size; ++x) {
            const float source_x = (x + 0.5f) * source.width / size - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(source_x)), 0, source.width - 1);
            const int x1 = std::min(x0 + 1, source.width - 1);
            const float fx = std::clamp(source_x - std::floor(source_x), 0.0f, 1.0f);
            for (int channel = 0; channel < 3; ++channel) {
                const auto at = [&](int px, int py) {
                    return source.rgb[(static_cast<size_t>(py) * source.width + px) * 3 + channel];
                };
                const float top = at(x0, y0) * (1.0f - fx) + at(x1, y0) * fx;
                const float bottom = at(x0, y1) * (1.0f - fx) + at(x1, y1) * fx;
                result.rgb[(static_cast<size_t>(y) * size + x) * 3 + channel] =
                    top * (1.0f - fy) + bottom * fy;
            }
        }
    }
    return result;
}

struct Tensor {
    GgufTensorView view;

    float value(size_t index) const {
        if (view.type == GgmlType::F32) {
            return reinterpret_cast<const float*>(view.data)[index];
        }
        if (view.type == GgmlType::BF16) {
            const uint32_t bits = static_cast<uint32_t>(
                reinterpret_cast<const uint16_t*>(view.data)[index]) << 16;
            float result = 0.0f;
            std::memcpy(&result, &bits, sizeof(result));
            return result;
        }
        throw std::invalid_argument("GGUF visual projector requires F32 or BF16 tensors");
    }

    size_t elements() const { return static_cast<size_t>(view.element_count); }
};

Tensor get_tensor(const GgufFile& file, std::string_view name) {
    return Tensor{file.tensor(name)};
}

void rmsnorm(std::vector<float>& values, const Tensor& weight, float epsilon,
             size_t width) {
    if (weight.elements() != width || values.size() % width != 0) {
        throw std::invalid_argument("GGUF visual norm shape mismatch");
    }
    for (size_t row = 0; row < values.size(); row += width) {
        float sum = 0.0f;
        for (size_t col = 0; col < width; ++col) sum += values[row + col] * values[row + col];
        const float scale = 1.0f / std::sqrt(sum / width + epsilon);
        for (size_t col = 0; col < width; ++col) {
            values[row + col] = values[row + col] * scale * weight.value(col);
        }
    }
}

void rmsnorm_unit(std::vector<float>& values, float epsilon, size_t width) {
    for (size_t row = 0; row < values.size(); row += width) {
        float sum = 0.0f;
        for (size_t col = 0; col < width; ++col) sum += values[row + col] * values[row + col];
        const float scale = 1.0f / std::sqrt(sum / width + epsilon);
        for (size_t col = 0; col < width; ++col) values[row + col] *= scale;
    }
}

void rmsnorm_heads(std::vector<float>& values, const Tensor& weight,
                   size_t rows, int heads, int head_width, float epsilon) {
    if (weight.elements() != static_cast<size_t>(head_width)) {
        throw std::invalid_argument("GGUF visual head norm shape mismatch");
    }
    for (size_t row = 0; row < rows; ++row) {
        for (int head = 0; head < heads; ++head) {
            float sum = 0.0f;
            const size_t base = row * static_cast<size_t>(heads * head_width) +
                static_cast<size_t>(head * head_width);
            for (int col = 0; col < head_width; ++col) sum += values[base + col] * values[base + col];
            const float scale = 1.0f / std::sqrt(sum / head_width + epsilon);
            for (int col = 0; col < head_width; ++col) {
                values[base + col] = values[base + col] * scale * weight.value(col);
            }
        }
    }
}

void rmsnorm_heads_unit(std::vector<float>& values, size_t rows, int heads,
                        int head_width, float epsilon) {
    for (size_t row = 0; row < rows; ++row) {
        for (int head = 0; head < heads; ++head) {
            float sum = 0.0f;
            const size_t base = row * static_cast<size_t>(heads * head_width) +
                static_cast<size_t>(head * head_width);
            for (int col = 0; col < head_width; ++col) sum += values[base + col] * values[base + col];
            const float scale = 1.0f / std::sqrt(sum / head_width + epsilon);
            for (int col = 0; col < head_width; ++col) values[base + col] *= scale;
        }
    }
}

std::vector<float> linear(const Tensor& weight, const std::vector<float>& input,
                          size_t rows, size_t cols, float input_min = 0.0f,
                          float input_max = 0.0f, float output_min = 0.0f,
                          float output_max = 0.0f) {
    if (weight.elements() != rows * cols || input.size() % cols != 0) {
        throw std::invalid_argument("GGUF visual linear shape mismatch");
    }
    std::vector<float> result(input.size() / cols * rows, 0.0f);
    for (size_t sample = 0; sample < input.size() / cols; ++sample) {
        for (size_t row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (size_t col = 0; col < cols; ++col) {
                float value = input[sample * cols + col];
                if (input_min < input_max) value = std::clamp(value, input_min, input_max);
                sum += weight.value(row * cols + col) * value;
            }
            if (output_min < output_max) sum = std::clamp(sum, output_min, output_max);
            result[sample * rows + row] = sum;
        }
    }
    return result;
}

class GgufProjectionProvider final : public IVisualEmbeddingProvider {
public:
    explicit GgufProjectionProvider(const std::filesystem::path& path)
        : file_(std::make_shared<GgufFile>(path.string())),
          patch_(get_tensor(*file_, "v.patch_embd.weight")),
          position_(get_tensor(*file_, "v.position_embd.weight")),
          projection_(get_tensor(*file_, "mm.input_projection.weight")) {
        image_size_ = static_cast<int>(file_->u32("clip.vision.image_size"));
        patch_size_ = static_cast<int>(file_->u32("clip.vision.patch_size"));
        hidden_ = static_cast<int>(file_->u32("clip.vision.embedding_length"));
        heads_ = static_cast<int>(file_->u32("clip.vision.attention.head_count"));
        layers_ = static_cast<int>(file_->u32("clip.vision.block_count"));
        intermediate_ = static_cast<int>(file_->u32("clip.vision.feed_forward_length"));
        epsilon_ = file_->f32_or("clip.vision.attention.layer_norm_epsilon", 1e-6f);
        merge_ = 3;
        if (image_size_ <= 0 || patch_size_ <= 0 || hidden_ <= 0 || heads_ <= 0 ||
            hidden_ % heads_ != 0 || image_size_ % patch_size_ != 0 ||
            projection_.view.shape.size() != 2 || position_.view.shape.size() != 3 ||
            position_.view.shape[0] != 2 || position_.view.shape[2] != hidden_) {
            throw std::invalid_argument("invalid GgufProjection projector metadata");
        }
        position_stride_ = static_cast<int>(position_.view.shape[1]);
        if (position_stride_ < image_size_ / patch_size_ ||
            projection_rows() != static_cast<int>(file_->u32("clip.vision.projection_dim"))) {
            throw std::invalid_argument("GgufProjection projector shape metadata mismatch");
        }
    }

    VisualEmbedding encode(std::string_view data_url) const override {
        const Image image = resize_square(decode_image(data_url), image_size_);
        const int grid = image_size_ / patch_size_;
        const size_t patches = static_cast<size_t>(grid) * grid;
        std::vector<float> tokens(patches * static_cast<size_t>(hidden_));
        const size_t patch_area = static_cast<size_t>(patch_size_) * patch_size_ * 3;
        for (int y = 0; y < grid; ++y) {
            for (int x = 0; x < grid; ++x) {
                const size_t patch_index = static_cast<size_t>(y * grid + x);
                float* output = tokens.data() + patch_index * hidden_;
                for (int out = 0; out < hidden_; ++out) {
                    float sum = 0.0f;
                    for (int channel = 0; channel < 3; ++channel) {
                        for (int py = 0; py < patch_size_; ++py) {
                            for (int px = 0; px < patch_size_; ++px) {
                                const size_t pixel =
                                    (static_cast<size_t>(y * patch_size_ + py) * image_size_ +
                                     x * patch_size_ + px) * 3 + channel;
                                const size_t weight = static_cast<size_t>(out) * patch_area +
                                    static_cast<size_t>(channel * patch_size_ * patch_size_ +
                                                        py * patch_size_ + px);
                                sum += patch_.value(weight) * (2.0f * image.rgb[pixel] - 1.0f);
                            }
                        }
                    }
                    output[out] = sum;
                }
                for (int out = 0; out < hidden_; ++out) {
                    output[out] += position_.value(static_cast<size_t>(x) * hidden_ + out);
                    output[out] += position_.value((static_cast<size_t>(position_stride_) +
                                                     static_cast<size_t>(y)) * hidden_ + out);
                }
            }
        }

        for (int layer = 0; layer < layers_; ++layer) run_block(tokens, layer, grid);

        const int pooled_grid = grid / merge_;
        std::vector<float> pooled(static_cast<size_t>(pooled_grid) * pooled_grid * hidden_);
        for (int y = 0; y < pooled_grid; ++y) {
            for (int x = 0; x < pooled_grid; ++x) {
                float* output = pooled.data() + static_cast<size_t>(y * pooled_grid + x) * hidden_;
                for (int dy = 0; dy < merge_; ++dy) for (int dx = 0; dx < merge_; ++dx) {
                    const float* input = tokens.data() +
                        static_cast<size_t>((y * merge_ + dy) * grid + x * merge_ + dx) * hidden_;
                    for (int col = 0; col < hidden_; ++col) output[col] += input[col] / 9.0f;
                }
                for (int col = 0; col < hidden_; ++col) output[col] *= std::sqrt(static_cast<float>(hidden_));
            }
        }
        if (file_->contains_tensor("v.std_bias") && file_->contains_tensor("v.std_scale")) {
            const Tensor bias = get_tensor(*file_, "v.std_bias");
            const Tensor scale = get_tensor(*file_, "v.std_scale");
            for (size_t index = 0; index < pooled.size(); ++index) {
                pooled[index] = (pooled[index] - bias.value(index % hidden_)) *
                    scale.value(index % hidden_);
            }
        }
        rmsnorm_unit(pooled, epsilon_, hidden_);
        return VisualEmbedding{projection_rows(), project(pooled), {}};
    }

private:
    int projection_rows() const {
        return static_cast<int>(projection_.view.shape.front());
    }

    std::vector<float> project(const std::vector<float>& pooled) const {
        const auto clamp = [&](std::string_view suffix, float fallback) {
            return file_->f32_or(std::string(suffix), fallback);
        };
        return linear(projection_, pooled, projection_rows(), hidden_,
                      clamp("mm.input_projection.input_min", 0.0f),
                      clamp("mm.input_projection.input_max", 0.0f),
                      clamp("mm.input_projection.output_min", 0.0f),
                      clamp("mm.input_projection.output_max", 0.0f));
    }

    void run_block(std::vector<float>& tokens, int layer, int grid) const {
        const std::string prefix = "v.blk." + std::to_string(layer) + ".";
        std::vector<float> residual = tokens;
        rmsnorm(tokens, get_tensor(*file_, prefix + "ln1.weight"), epsilon_, hidden_);
        std::vector<float> q = linear(get_tensor(*file_, prefix + "attn_q.weight"), tokens,
                                      hidden_, hidden_, scalar(prefix + "attn_q.input_min"),
                                      scalar(prefix + "attn_q.input_max"),
                                      scalar(prefix + "attn_q.output_min"),
                                      scalar(prefix + "attn_q.output_max"));
        std::vector<float> k = linear(get_tensor(*file_, prefix + "attn_k.weight"), tokens,
                                      hidden_, hidden_, scalar(prefix + "attn_k.input_min"),
                                      scalar(prefix + "attn_k.input_max"),
                                      scalar(prefix + "attn_k.output_min"),
                                      scalar(prefix + "attn_k.output_max"));
        std::vector<float> v = linear(get_tensor(*file_, prefix + "attn_v.weight"), tokens,
                                      hidden_, hidden_, scalar(prefix + "attn_v.input_min"),
                                      scalar(prefix + "attn_v.input_max"),
                                      scalar(prefix + "attn_v.output_min"),
                                      scalar(prefix + "attn_v.output_max"));
        const int head_width = hidden_ / heads_;
        rmsnorm_heads(q, get_tensor(*file_, prefix + "attn_q_norm.weight"),
                      tokens.size() / hidden_, heads_, head_width, epsilon_);
        rmsnorm_heads(k, get_tensor(*file_, prefix + "attn_k_norm.weight"),
                      tokens.size() / hidden_, heads_, head_width, epsilon_);
        rmsnorm_heads_unit(v, tokens.size() / hidden_, heads_, head_width, epsilon_);
        apply_rope(q, grid, head_width);
        apply_rope(k, grid, head_width);
        std::vector<float> attended(tokens.size(), 0.0f);
        const size_t rows = tokens.size() / hidden_;
        std::vector<float> scores(rows);
        for (size_t row = 0; row < rows; ++row) {
            for (int head = 0; head < heads_; ++head) {
                float maximum = -std::numeric_limits<float>::infinity();
                for (size_t key = 0; key < rows; ++key) {
                    float score = 0.0f;
                    for (int col = 0; col < head_width; ++col) {
                        score += q[row * hidden_ + head * head_width + col] *
                            k[key * hidden_ + head * head_width + col];
                    }
                    scores[key] = score;
                    maximum = std::max(maximum, scores[key]);
                }
                float denominator = 0.0f;
                for (size_t key = 0; key < rows; ++key) {
                    scores[key] = std::exp(scores[key] - maximum);
                    denominator += scores[key];
                }
                for (size_t key = 0; key < rows; ++key) {
                    const float probability = scores[key] / denominator;
                    for (int col = 0; col < head_width; ++col) {
                        attended[row * hidden_ + head * head_width + col] +=
                            probability * v[key * hidden_ + head * head_width + col];
                    }
                }
            }
        }
        std::vector<float> attention = linear(get_tensor(*file_, prefix + "attn_out.weight"),
                                              attended, hidden_, hidden_,
                                              scalar(prefix + "attn_out.input_min"),
                                              scalar(prefix + "attn_out.input_max"),
                                              scalar(prefix + "attn_out.output_min"),
                                              scalar(prefix + "attn_out.output_max"));
        rmsnorm(attention, get_tensor(*file_, prefix + "attn_post_norm.weight"), epsilon_, hidden_);
        for (size_t index = 0; index < tokens.size(); ++index) tokens[index] = residual[index] + attention[index];
        residual = tokens;
        rmsnorm(tokens, get_tensor(*file_, prefix + "ln2.weight"), epsilon_, hidden_);
        std::vector<float> gate = linear(get_tensor(*file_, prefix + "ffn_gate.weight"), tokens,
                                         intermediate_, hidden_, scalar(prefix + "ffn_gate.input_min"),
                                         scalar(prefix + "ffn_gate.input_max"),
                                         scalar(prefix + "ffn_gate.output_min"),
                                         scalar(prefix + "ffn_gate.output_max"));
        std::vector<float> up = linear(get_tensor(*file_, prefix + "ffn_up.weight"), tokens,
                                       intermediate_, hidden_, scalar(prefix + "ffn_up.input_min"),
                                       scalar(prefix + "ffn_up.input_max"),
                                       scalar(prefix + "ffn_up.output_min"),
                                       scalar(prefix + "ffn_up.output_max"));
        for (size_t index = 0; index < gate.size(); ++index) {
            const float value = gate[index];
            gate[index] = 0.5f * value *
                (1.0f + std::tanh(0.7978845608f * (value + 0.044715f * value * value * value))) * up[index];
        }
        std::vector<float> feed_forward = linear(get_tensor(*file_, prefix + "ffn_down.weight"),
                                                   gate, hidden_, intermediate_,
                                                   scalar(prefix + "ffn_down.input_min"),
                                                   scalar(prefix + "ffn_down.input_max"),
                                                   scalar(prefix + "ffn_down.output_min"),
                                                   scalar(prefix + "ffn_down.output_max"));
        rmsnorm(feed_forward, get_tensor(*file_, prefix + "ffn_post_norm.weight"), epsilon_, hidden_);
        for (size_t index = 0; index < tokens.size(); ++index) tokens[index] = residual[index] + feed_forward[index];
    }

    float scalar(const std::string& name) const {
        return file_->has(name) ? file_->f32(name) : 0.0f;
    }
    void apply_rope(std::vector<float>& values, int grid, int head_width) const {
        const size_t rows = values.size() / hidden_;
        const int half = head_width / 2;
        for (size_t row = 0; row < rows; ++row) {
            const int x = static_cast<int>(row % static_cast<size_t>(grid));
            const int y = static_cast<int>(row / static_cast<size_t>(grid));
            for (int head = 0; head < heads_; ++head) {
                float* vector = values.data() + row * hidden_ + head * head_width;
                rotate_half(vector, half, x);
                rotate_half(vector + half, half, y);
            }
        }
    }
    void rotate_half(float* values, int width, int position) const {
        for (int index = 0; index < width; index += 2) {
            const float theta = position * std::pow(100.0f, -static_cast<float>(index) / width);
            const float cosine = std::cos(theta);
            const float sine = std::sin(theta);
            const float first = values[index];
            const float second = values[index + 1];
            values[index] = first * cosine - second * sine;
            values[index + 1] = first * sine + second * cosine;
        }
    }

    std::shared_ptr<GgufFile> file_;
    Tensor patch_;
    Tensor position_;
    Tensor projection_;
    int image_size_ = 0;
    int patch_size_ = 0;
    int hidden_ = 0;
    int heads_ = 0;
    int layers_ = 0;
    int intermediate_ = 0;
    int position_stride_ = 0;
    int merge_ = 3;
    float epsilon_ = 1e-6f;
};

}

VisualEmbeddingProvider make_gguf_visual_embedding_provider(
    const std::filesystem::path& projector_path) {
    return std::make_shared<GgufProjectionProvider>(projector_path);
}

std::unique_ptr<IVisionProviderFactory> make_gguf_vision_provider_factory() {
    return std::make_unique<GgufProjectionProviderFactory>();
}

}
