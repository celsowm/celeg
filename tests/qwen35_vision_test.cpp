#include "celeg/runtime/vision/providers.hpp"
#include "celeg/checkpoint/weight_repository.hpp"
#include "support/assertions.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

class EmptyVisionRepository final : public celeg::IWeightRepository {
public:
    bool contains(std::string_view) const override { return false; }
    celeg::HostTensorView tensor(std::string_view) const override { return {}; }
    std::vector<std::string> names() const override { return {}; }
};

}

int main() {
    bool rejected_empty_repository = false;
    try {
        (void)celeg::make_safetensor_visual_embedding_provider(
            std::make_shared<EmptyVisionRepository>());
    } catch (const std::invalid_argument&) {
        rejected_empty_repository = true;
    }
    CELEG_TEST_CHECK(rejected_empty_repository);

    const char* path = std::getenv("CELEG_QWEN35_MODEL");
    if (!path || !std::filesystem::is_directory(path)) {
        std::cout << "qwen35_vision_test: skipped (CELEG_QWEN35_MODEL is not set)\n";
        return 0;
    }
    const auto provider = celeg::make_safetensor_visual_embedding_provider(path);
    const celeg::VisualEmbedding embedding = provider->encode(
        "data:image/x-portable-pixmap;base64,UDYKMSAxCjI1NQr/AAA=");
    CELEG_TEST_CHECK(embedding.width == 2048);
    CELEG_TEST_CHECK(embedding.token_count() == 1);
    CELEG_TEST_CHECK(embedding.rope_positions.size() == embedding.token_count());
    for (float value : embedding.values) CELEG_TEST_CHECK(std::isfinite(value));
    const celeg::VisualEmbedding wide = provider->encode(
        "data:image/x-portable-pixmap;base64,UDYKNCAxCjI1NQr/AAAA/wAAAP////8=");
    CELEG_TEST_CHECK(wide.width == 2048);
    CELEG_TEST_CHECK(wide.token_count() == 2);
    CELEG_TEST_CHECK(wide.rope_positions.size() == 2);
    CELEG_TEST_CHECK(wide.rope_positions[1][2] == 1);
    std::cout << "qwen35_vision_test: ok\n";
    return 0;
}
