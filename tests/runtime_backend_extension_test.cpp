#include "celeg/runtime/context.hpp"
#include "celeg/serve/inference_service.hpp"
#include "support/assertions.hpp"

#include <memory>

namespace {

class FakeBackend final : public celeg::IBackendFactory {
public:
    std::string_view id() const override { return "test-backend"; }

    bool supports(celeg::BackendId backend) const override {
        return backend == id();
    }

    std::unique_ptr<celeg::serve::ServiceBundle> create(
        const celeg::BackendCreateRequest&) const override {
        return {};
    }
};

class FakeBackendModule final : public celeg::IRuntimeModule {
public:
    std::string_view id() const override { return "test-backend-module"; }

    void register_into(celeg::RuntimeBuilder& builder) const override {
        builder.add_backend_factory(std::make_unique<FakeBackend>());
    }
};

}

int main() {
    const auto runtime = celeg::RuntimeBuilder{}
        .add_module(std::make_unique<FakeBackendModule>())
        .build_shared();
    const celeg::IBackendFactory& selected = runtime->backends().select_best_if(
        [](const celeg::IBackendFactory& factory) {
            return factory.supports("test-backend");
        });
    CELEG_TEST_CHECK(selected.id() == "test-backend");
    CELEG_TEST_CHECK(runtime->backends().ids().size() == 1);
    return 0;
}
