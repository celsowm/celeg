#include "celeg/vision/image.hpp"
#include "support/assertions.hpp"

#include <cmath>
#include <stdexcept>

int main() {
    const celeg::ImageRgbF32 image = celeg::decode_image_data_url(
        "data:image/x-portable-pixmap;base64,UDYKMSAxCjI1NQr/AIA=");
    CELEG_TEST_CHECK(image.width == 1);
    CELEG_TEST_CHECK(image.height == 1);
    CELEG_TEST_CHECK(image.rgb.size() == 3);
    CELEG_TEST_CHECK(std::abs(image.rgb[0] - 1.0f) < 1.0e-6f);
    CELEG_TEST_CHECK(std::abs(image.rgb[1]) < 1.0e-6f);
    CELEG_TEST_CHECK(std::abs(image.rgb[2] - 128.0f / 255.0f) < 1.0e-6f);

    const celeg::ImageRgbF32 resized = celeg::resize_image_bilinear(image, 2, 2);
    CELEG_TEST_CHECK(resized.width == 2);
    CELEG_TEST_CHECK(resized.height == 2);
    for (float value : resized.rgb) {
        CELEG_TEST_CHECK(value >= 0.0f && value <= 1.0f);
    }

    bool rejected = false;
    try {
        (void)celeg::decode_image_data_url("not-an-image");
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    CELEG_TEST_CHECK(rejected);
    return 0;
}
