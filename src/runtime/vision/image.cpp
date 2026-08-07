#include "celeg/vision/image.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

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

std::vector<std::uint8_t> decode_base64(std::string_view encoded) {
    static constexpr char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<std::uint8_t> result;
    int value = 0;
    int bits = -8;
    for (char byte : encoded) {
        if (byte == '=') break;
        const char* found = std::find(std::begin(alphabet), std::end(alphabet) - 1, byte);
        if (found == std::end(alphabet) - 1) continue;
        value = (value << 6) + static_cast<int>(found - alphabet);
        bits += 6;
        if (bits >= 0) {
            result.push_back(static_cast<std::uint8_t>((value >> bits) & 0xff));
            bits -= 8;
        }
    }
    if (result.empty()) throw std::invalid_argument("image data URL has no payload");
    return result;
}

ImageRgbF32 decode_ppm(std::vector<std::uint8_t> bytes) {
    std::size_t cursor = 0;
    auto next = [&]() {
        while (cursor < bytes.size() && bytes[cursor] <= ' ') ++cursor;
        const std::size_t begin = cursor;
        while (cursor < bytes.size() && bytes[cursor] > ' ') ++cursor;
        if (begin == cursor) throw std::invalid_argument("invalid PPM header");
        return std::string(reinterpret_cast<const char*>(bytes.data() + begin), cursor - begin);
    };
    if (next() != "P6") {
        throw std::invalid_argument("image decoder currently accepts P6 PPM input");
    }
    const int width = std::stoi(next());
    const int height = std::stoi(next());
    if (std::stoi(next()) != 255 || width <= 0 || height <= 0) {
        throw std::invalid_argument("invalid PPM dimensions or range");
    }
    while (cursor < bytes.size() && bytes[cursor] <= ' ') ++cursor;
    const std::size_t count = static_cast<std::size_t>(width) * height * 3;
    if (cursor + count > bytes.size()) throw std::invalid_argument("truncated PPM pixels");
    ImageRgbF32 image{width, height, std::vector<float>(count)};
    for (std::size_t index = 0; index < count; ++index) {
        image.rgb[index] = bytes[cursor + index] / 255.0f;
    }
    return image;
}

#if defined(_WIN32)
ImageRgbF32 decode_wic(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() > static_cast<std::size_t>(std::numeric_limits<ULONG>::max())) {
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
    UINT width = 0;
    UINT height = 0;
    if (FAILED(converter->GetSize(&width, &height)) || width == 0 || height == 0) {
        cleanup();
        throw std::invalid_argument("decoded image has invalid dimensions");
    }
    ImageRgbF32 image{static_cast<int>(width), static_cast<int>(height),
                      std::vector<float>(static_cast<std::size_t>(width) * height * 3)};
    std::vector<std::uint8_t> pixels(image.rgb.size());
    if (FAILED(converter->CopyPixels(nullptr, width * 3,
                                     static_cast<UINT>(pixels.size()), pixels.data()))) {
        cleanup();
        throw std::invalid_argument("Windows image decoder failed to read pixels");
    }
    for (std::size_t index = 0; index < image.rgb.size(); ++index) {
        image.rgb[index] = pixels[index] / 255.0f;
    }
    cleanup();
    return image;
}
#endif

} // namespace

ImageRgbF32 decode_image_data_url(std::string_view data_url) {
    const std::size_t comma = data_url.find(',');
    if (!data_url.starts_with("data:") || comma == std::string_view::npos ||
        data_url.substr(0, comma).find(";base64") == std::string_view::npos) {
        throw std::invalid_argument("image input must be a base64 data URL");
    }
    const std::vector<std::uint8_t> bytes = decode_base64(data_url.substr(comma + 1));
    if (bytes.size() >= 2 && bytes[0] == 'P' && bytes[1] == '6') return decode_ppm(bytes);
#if defined(_WIN32)
    return decode_wic(bytes);
#else
    throw std::invalid_argument("PNG and JPEG decoding requires the Windows image codec backend");
#endif
}

ImageRgbF32 resize_image_bilinear(const ImageRgbF32& source, int width, int height) {
    if (source.width <= 0 || source.height <= 0 || source.rgb.size() !=
        static_cast<std::size_t>(source.width) * source.height * 3 || width <= 0 || height <= 0) {
        throw std::invalid_argument("invalid image dimensions for resize");
    }
    ImageRgbF32 result{width, height,
                       std::vector<float>(static_cast<std::size_t>(width) * height * 3)};
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
            const float top = (1.0f - fx) * source.rgb[(static_cast<std::size_t>(y0) * source.width + x0) * 3 + c] +
                              fx * source.rgb[(static_cast<std::size_t>(y0) * source.width + x1) * 3 + c];
            const float bottom = (1.0f - fx) * source.rgb[(static_cast<std::size_t>(y1) * source.width + x0) * 3 + c] +
                                 fx * source.rgb[(static_cast<std::size_t>(y1) * source.width + x1) * 3 + c];
            result.rgb[(static_cast<std::size_t>(y) * width + x) * 3 + c] =
                (1.0f - fy) * top + fy * bottom;
        }
    }
    return result;
}

} // namespace celeg
