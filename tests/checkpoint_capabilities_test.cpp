#include "lfm/checkpoint/formats/safetensors.hpp"
#include "support/assertions.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

class ViewOnlyRepository final : public lfm::IWeightRepository {
public:
    bool contains(std::string_view) const override { return false; }
    lfm::HostTensorView tensor(std::string_view) const override { return {}; }
    std::vector<std::string> names() const override { return {}; }
};

class IndexedRepository final
    : public lfm::IWeightRepository,
      public lfm::ILocatableTensorRepository,
      public lfm::IRandomAccessTensorReader {
public:
    bool contains(std::string_view) const override { return true; }
    lfm::HostTensorView tensor(std::string_view) const override { return {}; }
    std::vector<std::string> names() const override { return {}; }

    lfm::TensorLocator locate(std::string_view) const override {
        return lfm::TensorLocator{.bytes = 1};
    }

    void read(const lfm::TensorLocator& locator,
              std::span<std::byte> destination) const override {
        LFM_TEST_CHECK(locator.bytes == destination.size());
        destination[0] = std::byte{0x5a};
    }
};

} // namespace

int main() {
    ViewOnlyRepository view_only;
    bool rejected = false;
    try {
        (void)lfm::require_locatable_tensor_repository(view_only);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    LFM_TEST_CHECK(rejected);

    IndexedRepository indexed;
    const auto& locator = lfm::require_locatable_tensor_repository(indexed);
    const auto& reader = lfm::require_random_access_tensor_reader(indexed);
    const lfm::TensorLocator location = locator.locate("tensor");
    std::byte destination[1]{};
    reader.read(location, std::span<std::byte>(destination));
    LFM_TEST_CHECK(destination[0] == std::byte{0x5a});
    return 0;
}
