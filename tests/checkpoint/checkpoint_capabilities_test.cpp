#include "celeg/checkpoint/formats/safetensors.hpp"
#include "celeg/checkpoint/metadata.hpp"
#include "celeg/model/inference.hpp"
#include "support/assertions.hpp"

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

class ViewOnlyRepository final : public celeg::IWeightRepository {
public:
    bool contains(std::string_view) const override { return false; }
    celeg::HostTensorView tensor(std::string_view) const override { return {}; }
    std::vector<std::string> names() const override { return {}; }
};

class IndexedRepository final
    : public celeg::IWeightRepository,
      public celeg::ILocatableTensorRepository,
      public celeg::IRandomAccessTensorReader {
public:
    bool contains(std::string_view) const override { return true; }
    celeg::HostTensorView tensor(std::string_view) const override { return {}; }
    std::vector<std::string> names() const override { return {}; }

    celeg::TensorLocator locate(std::string_view) const override {
        return celeg::TensorLocator{.bytes = 1};
    }

    void read(const celeg::TensorLocator& locator,
              std::span<std::byte> destination) const override {
        CELEG_TEST_CHECK(locator.bytes == destination.size());
        destination[0] = std::byte{0x5a};
    }
};

}

int main() {
    ViewOnlyRepository view_only;
    bool rejected = false;
    try {
        (void)celeg::require_locatable_tensor_repository(view_only);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    CELEG_TEST_CHECK(rejected);

    IndexedRepository indexed;
    const auto& locator = celeg::require_locatable_tensor_repository(indexed);
    const auto& reader = celeg::require_random_access_tensor_reader(indexed);
    const celeg::TensorLocator location = locator.locate("tensor");
    std::byte destination[1]{};
    reader.read(location, std::span<std::byte>(destination));
    CELEG_TEST_CHECK(destination[0] == std::byte{0x5a});

    const auto metadata = celeg::CheckpointMetadata::from_json(
        celeg::Json::parse(R"({"rope_layer_flags":[true,false,true]})"));
    const auto flags = metadata.integers("rope_layer_flags");
    CELEG_TEST_CHECK(flags.size() == 3);
    CELEG_TEST_CHECK(flags[0] == 1);
    CELEG_TEST_CHECK(flags[1] == 0);
    CELEG_TEST_CHECK(flags[2] == 1);

    const auto rms_qk_metadata = celeg::CheckpointMetadata::from_json(
        celeg::Json::parse(
            R"({"rope_theta":10000,"use_qk_norm":true,"qk_norm_type":"rmsnorm"})"));
    const auto rms_qk = celeg::normalize_model_metadata(rms_qk_metadata);
    CELEG_TEST_CHECK(rms_qk.attention.query_key_norm.value_or(false));

    bool rejected_qk_math = false;
    try {
        const auto unsupported_qk_metadata = celeg::CheckpointMetadata::from_json(
            celeg::Json::parse(
                R"({"rope_theta":10000,"use_qk_norm":true,"qk_norm_type":"layernorm"})"));
        (void)celeg::normalize_model_metadata(unsupported_qk_metadata);
    } catch (const celeg::ResolutionError& error) {
        rejected_qk_math =
            error.kind() == celeg::ResolutionFailureKind::UnsupportedSemanticFeature;
    }
    CELEG_TEST_CHECK(rejected_qk_math);
    return 0;
}
