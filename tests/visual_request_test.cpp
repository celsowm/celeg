#include "celeg/serve/visual_request.hpp"
#include "support/assertions.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

class FakeVisualProvider final : public celeg::IVisualEmbeddingProvider {
public:
    celeg::VisualEmbedding encode(std::string_view data_url) const override {
        CELEG_TEST_CHECK(data_url == "data:image/test;base64,AA==");
        return {2, {1.0f, 2.0f, 3.0f, 4.0f}};
    }
};

} // namespace

int main() {
    celeg::serve::GenerateRequest request;
    request.prompt_tokens = {7, 42, 9};
    request.image_token_id = 42;
    request.images = {{"data:image/test;base64,AA=="}};

    const celeg::VisualEmbeddingProvider provider =
        std::make_shared<FakeVisualProvider>();
    celeg::serve::materialize_visual_prompt(request, provider);

    CELEG_TEST_CHECK(request.images.empty());
    CELEG_TEST_CHECK(request.prompt_embedding.width == 2);
    CELEG_TEST_CHECK(request.prompt_tokens == std::vector<std::int32_t>({7, 42, 42, 9}));
    CELEG_TEST_CHECK(request.prompt_embedding.positions == std::vector<std::size_t>({1, 2}));
    CELEG_TEST_CHECK(request.prompt_embedding.values == std::vector<float>({1.0f, 2.0f, 3.0f, 4.0f}));

    std::cout << "visual_request_test: ok\n";
    return 0;
}
