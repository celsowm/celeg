#pragma once

#include <cstdint>
#include <string_view>
#include <vector>

namespace celeg {

struct ImageRgbF32 {
    int width = 0;
    int height = 0;
    std::vector<float> rgb;
};

ImageRgbF32 decode_image_data_url(std::string_view data_url);
ImageRgbF32 resize_image_bilinear(const ImageRgbF32& source, int width, int height);

} // namespace celeg
