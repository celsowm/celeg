#include "celeg/checkpoint/catalog.hpp"
#include "support/assertions.hpp"

#include <memory>
#include <stdexcept>

namespace {

class TestFormat final : public celeg::ICheckpointFormat {
public:
    explicit TestFormat(const char* name, bool selected) : name_(name), selected_(selected) {}

    std::string_view id() const override { return name_; }
    bool matches(const std::filesystem::path&) const override { return selected_; }
    celeg::CheckpointView open(const std::filesystem::path& path) const override {
        celeg::CheckpointView result;
        result.path = path;
        return result;
    }

private:
    const char* name_;
    bool selected_;
};

} // namespace

int main() {
    celeg::CheckpointFormatCatalog catalog;
    catalog.add(std::make_unique<TestFormat>("selected", true));
    catalog.add(std::make_unique<TestFormat>("unused", false));
    catalog.freeze();

    CELEG_TEST_CHECK(catalog.select("fixture").id() == "selected");
    CELEG_TEST_CHECK(catalog.open("fixture").path == "fixture");

    bool rejected = false;
    try {
        catalog.add(std::make_unique<TestFormat>("late", false));
    } catch (const std::logic_error&) {
        rejected = true;
    }
    CELEG_TEST_CHECK(rejected);

    celeg::CheckpointFormatCatalog ambiguous;
    ambiguous.add(std::make_unique<TestFormat>("one", true));
    ambiguous.add(std::make_unique<TestFormat>("two", true));
    bool ambiguous_rejected = false;
    try {
        (void)ambiguous.select("fixture");
    } catch (const std::runtime_error&) {
        ambiguous_rejected = true;
    }
    CELEG_TEST_CHECK(ambiguous_rejected);
    return 0;
}
