#include "celeg/runtime/context.hpp"
#include "celeg/text/tokenizer.hpp"
#include "support/assertions.hpp"

#include <memory>
#include <stdexcept>

namespace {

class ExtensionArchitecture final : public celeg::IArchitecture {
public:
    std::string_view id() const override { return "test-extension"; }
    celeg::ProbeResult probe(const celeg::CheckpointMetadata&) const override {
        return {false, 0, "test-only"};
    }
    celeg::ResolvedModel resolve(const celeg::CheckpointView&) const override {
        return {};
    }
};

class ExtensionFormat final : public celeg::ICheckpointFormat {
public:
    std::string_view id() const override { return "test-format"; }
    bool matches(const std::filesystem::path&) const override { return false; }
    celeg::CheckpointView open(const std::filesystem::path& path) const override {
        celeg::CheckpointView result;
        result.path = path;
        return result;
    }
};

class ExtensionTokenizerProvider final : public celeg::ITokenizerProvider {
public:
    std::string_view id() const override { return "test-tokenizer"; }
    bool supports(const celeg::CheckpointView&,
                  const std::filesystem::path&) const override { return false; }
    std::unique_ptr<celeg::BpeTokenizer> create(
        const celeg::CheckpointView&, const std::filesystem::path&) const override {
        throw std::logic_error("test tokenizer provider is not selected");
    }
};

class ExtensionBackendProvider final : public celeg::IBackendFactory {
public:
    explicit ExtensionBackendProvider(const char* id) : id_(id) {}
    std::string_view id() const override { return id_; }
    bool supports(celeg::BackendId) const override { return false; }
    std::unique_ptr<celeg::serve::ServiceBundle> create(
        const celeg::BackendCreateRequest&) const override {
        throw std::logic_error("test backend cannot create services");
    }
private:
    const char* id_;
};

class ExtensionVisionProvider final : public celeg::IVisionProviderFactory {
public:
    explicit ExtensionVisionProvider(const char* id) : id_(id) {}
    std::string_view id() const override { return id_; }
    bool supports(std::string_view,
                  const std::filesystem::path&) const override { return false; }
    std::shared_ptr<const celeg::IVisualEmbeddingProvider> create(
        const std::filesystem::path&) const override {
        throw std::logic_error("test provider cannot create vision embeddings");
    }
private:
    const char* id_;
};

class ExtensionModule final : public celeg::IRuntimeModule {
public:
    std::string_view id() const override { return "test-module"; }
    void register_into(celeg::RuntimeBuilder& builder) const override {
        builder.add_architecture(std::make_unique<ExtensionArchitecture>());
    }
};

} // namespace

int main() {
    celeg::RuntimeContext runtime = celeg::RuntimeBuilder{}
        .add_builtins()
        .add_module(std::make_unique<ExtensionModule>())
        .add_checkpoint_format(std::make_unique<ExtensionFormat>())
        .add_tokenizer_provider(
            std::make_unique<ExtensionTokenizerProvider>())
        .add_backend_factory(
            std::make_unique<ExtensionBackendProvider>("test-backend"))
        .add_vision_provider(
            std::make_unique<ExtensionVisionProvider>("test-vision"))
        .build();

    CELEG_TEST_CHECK(runtime.architectures().find("test-extension") != nullptr);
    CELEG_TEST_CHECK(runtime.checkpoint_formats().select("model.gguf").id() == "gguf");
    CELEG_TEST_CHECK(runtime.chat_profiles().find("lfm2-instruct").format({}, false).size() > 0);
    CELEG_TEST_CHECK(runtime.tokenizer_providers().find("test-tokenizer").id() == "test-tokenizer");
    celeg::CheckpointView tokenizer_checkpoint;
    auto tokenizer_data = std::make_shared<celeg::TokenizerData>();
    tokenizer_data->tokens = {"h", "i", "hi"};
    tokenizer_data->merges = {"h i"};
    tokenizer_checkpoint.tokenizer = std::move(tokenizer_data);
    const auto& tokenizer_provider = celeg::select_tokenizer_provider(
        runtime, tokenizer_checkpoint, "model.gguf");
    CELEG_TEST_CHECK(tokenizer_provider.id() == "bpe");
    const auto tokenizer = tokenizer_provider.create(tokenizer_checkpoint, "model.gguf");
    CELEG_TEST_CHECK(tokenizer->encode("hi", false).size() == 1);
    CELEG_TEST_CHECK(runtime.backends().find("test-backend").id() == "test-backend");
    CELEG_TEST_CHECK(runtime.vision_providers().find("test-vision").id() == "test-vision");

    bool rejected = false;
    try {
        (void)runtime.architectures();
        // The frozen catalog must reject additions through its public API.
        const_cast<celeg::ArchitectureCatalog&>(runtime.architectures())
            .add(std::make_unique<ExtensionArchitecture>());
    } catch (const std::logic_error&) {
        rejected = true;
    }
    CELEG_TEST_CHECK(rejected);

    bool duplicate_module_rejected = false;
    try {
        celeg::RuntimeBuilder duplicate_builder;
        duplicate_builder.add_module(std::make_unique<ExtensionModule>());
        duplicate_builder.add_module(std::make_unique<ExtensionModule>());
    } catch (const std::invalid_argument&) {
        duplicate_module_rejected = true;
    }
    CELEG_TEST_CHECK(duplicate_module_rejected);
    return 0;
}
