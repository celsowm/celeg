#include "celeg/runtime/context.hpp"
#include "celeg/text/tokenizer.hpp"
#include "support/assertions.hpp"

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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
    class FakeTokenizer final : public celeg::ITokenizer {
    public:
        std::vector<int32_t> encode(std::string_view text, bool add_bos) const override {
            std::vector<int32_t> result;
            if (add_bos) result.push_back(bos_id());
            for (const unsigned char byte : text) result.push_back(byte);
            return result;
        }
        std::string decode(const std::vector<int32_t>& ids, bool skip_special) const override {
            std::string result;
            for (const int32_t id : ids) {
                if (skip_special && id == bos_id()) continue;
                result.push_back(static_cast<char>(id));
            }
            return result;
        }
        std::string decode_token(int32_t id, bool skip_special) const override {
            return skip_special && id == bos_id() ? std::string{} :
                std::string(1, static_cast<char>(id));
        }
        std::optional<int32_t> token_id(std::string_view text) const override {
            return text.size() == 1 ? std::optional<int32_t>{
                static_cast<unsigned char>(text.front())} : std::nullopt;
        }
        int32_t bos_id() const override { return 1; }
        int32_t eos_id() const override { return 2; }
        int32_t pad_id() const override { return 0; }
    };

    std::string_view id() const override { return "test-tokenizer"; }
    bool supports(const celeg::CheckpointView&,
                  const std::filesystem::path&) const override { return false; }
    std::unique_ptr<celeg::ITokenizer> create(
        const celeg::CheckpointView&, const std::filesystem::path&) const override {
        return std::make_unique<FakeTokenizer>();
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
    bool supports(const std::filesystem::path&) const override { return false; }
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

}

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
    ExtensionTokenizerProvider extension_tokenizer;
    const auto fake_tokenizer = extension_tokenizer.create(tokenizer_checkpoint, "model.gguf");
    CELEG_TEST_CHECK(fake_tokenizer->decode(fake_tokenizer->encode("x")) == "x");
    CELEG_TEST_CHECK(runtime.backends().find("test-backend").id() == "test-backend");
    CELEG_TEST_CHECK(runtime.vision_providers().find("test-vision").id() == "test-vision");

    bool rejected = false;
    try {
        (void)runtime.architectures();
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
