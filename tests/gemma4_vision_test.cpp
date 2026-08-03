#include "celeg/models/gemma4/vision.hpp"
#include "support/assertions.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>

int main() {
    const char* path = std::getenv("CELEG_GEMMA4_MMPROJ");
    if (!path || !std::filesystem::is_regular_file(path)) {
        std::cout << "gemma4_vision_test: skipped (CELEG_GEMMA4_MMPROJ is not set)\n";
        return 0;
    }

    const celeg::VisualEmbeddingProvider provider =
        celeg::make_gemma4_visual_embedding_provider(path);
    const celeg::VisualEmbedding embedding = provider->encode(
        "data:image/x-portable-pixmap;base64,UDYKMSAxCjI1NQr/AAA=");
    CELEG_TEST_CHECK(embedding.width == 2560);
    CELEG_TEST_CHECK(embedding.token_count() == 16);
    for (const float value : embedding.values) {
        CELEG_TEST_CHECK(std::isfinite(value));
    }

    std::cout << "gemma4_vision_test: ok\n";
    return 0;
}
