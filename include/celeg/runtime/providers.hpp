#pragma once

#include "celeg/checkpoint/view.hpp"
#include "celeg/text/tokenizer.hpp"
#include "celeg/text/tokenizer_definition.hpp"

#include <filesystem>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {

class RuntimeBuilder;
class RuntimeContext;

namespace serve {
class ServiceBundle;
}

class IVisualEmbeddingProvider;

using BackendId = std::string_view;

class IBackendOptions {
public:
    virtual ~IBackendOptions() = default;
    virtual BackendId backend_id() const noexcept = 0;

    template <typename T>
    const T* as() const noexcept {
        return type_tag() == type_tag_for<T>()
            ? static_cast<const T*>(data()) : nullptr;
    }

protected:
    template <typename T>
    static const void* type_tag_for() noexcept {
        static const int tag = 0;
        return &tag;
    }

    virtual const void* type_tag() const noexcept = 0;
    virtual const void* data() const noexcept = 0;
};

struct BackendCreateRequest {
    std::string model_path;
    int max_context = 0;
    BackendId backend_id;
    std::shared_ptr<const RuntimeContext> runtime;
    std::shared_ptr<const IBackendOptions> options;
};

class ITokenizerProvider {
public:
    virtual ~ITokenizerProvider() = default;
    virtual std::string_view id() const = 0;
    virtual bool supports(const CheckpointView& checkpoint,
                          const std::filesystem::path& model_path) const = 0;
    virtual std::unique_ptr<ITokenizer> create(
        const CheckpointView& checkpoint,
        const std::filesystem::path& model_path) const = 0;
};

class IRuntimeModule {
public:
    virtual ~IRuntimeModule() = default;
    virtual std::string_view id() const = 0;
    virtual void register_into(RuntimeBuilder& builder) const = 0;
};

class IBackendFactory {
public:
    virtual ~IBackendFactory() = default;
    virtual std::string_view id() const = 0;
    virtual int priority() const noexcept { return 0; }
    virtual bool supports(BackendId backend) const = 0;
    virtual std::unique_ptr<serve::ServiceBundle> create(
        const BackendCreateRequest& request) const = 0;
};

class IAbiBackendFactory : public IBackendFactory {
public:
    ~IAbiBackendFactory() override = default;
    virtual std::shared_ptr<const IBackendOptions> decode_options(
        std::span<const std::byte> bytes) const = 0;
};

class IVisionProviderFactory {
public:
    virtual ~IVisionProviderFactory() = default;
    virtual std::string_view id() const = 0;
    virtual bool supports(const std::filesystem::path& model_path) const = 0;
    virtual std::shared_ptr<const IVisualEmbeddingProvider> create(
        const std::filesystem::path& model_path) const = 0;
};

template <typename Provider>
class ProviderCatalog {
public:
    void add(std::unique_ptr<Provider> provider) {
        if (frozen_) throw std::logic_error("provider catalog is frozen");
        if (!provider || provider->id().empty()) {
            throw std::invalid_argument("provider must have a non-empty id");
        }
        for (const auto& existing : providers_) {
            if (existing->id() == provider->id()) {
                throw std::invalid_argument(
                    "duplicate provider id: " + std::string(provider->id()));
            }
        }
        providers_.push_back(std::move(provider));
    }

    void freeze() { frozen_ = true; }

    const Provider& find(std::string_view id) const {
        for (const auto& provider : providers_) {
            if (provider->id() == id) return *provider;
        }
        throw std::invalid_argument("unknown provider: " + std::string(id));
    }

    std::vector<std::string_view> ids() const {
        std::vector<std::string_view> result;
        result.reserve(providers_.size());
        for (const auto& provider : providers_) result.push_back(provider->id());
        return result;
    }

    template <typename Predicate>
    const Provider& select_if(Predicate&& predicate) const {
        for (const auto& provider : providers_) {
            if (predicate(*provider)) return *provider;
        }
        throw std::invalid_argument("no provider supports the requested input");
    }

    template <typename Predicate>
    const Provider& select_best_if(Predicate&& predicate) const {
        const Provider* selected = nullptr;
        int selected_priority = 0;
        for (const auto& provider : providers_) {
            if (!predicate(*provider)) continue;
            const int priority = provider->priority();
            if (selected == nullptr || priority > selected_priority) {
                selected = provider.get();
                selected_priority = priority;
            } else if (priority == selected_priority) {
                throw std::invalid_argument(
                    "multiple providers support the requested input at the same priority");
            }
        }
        if (selected == nullptr) {
            throw std::invalid_argument("no provider supports the requested input");
        }
        return *selected;
    }

private:
    bool frozen_ = false;
    std::vector<std::unique_ptr<Provider>> providers_;
};

using TokenizerProviderCatalog = ProviderCatalog<ITokenizerProvider>;
using BackendFactoryCatalog = ProviderCatalog<IBackendFactory>;
using VisionProviderCatalog = ProviderCatalog<IVisionProviderFactory>;

std::unique_ptr<ITokenizerProvider> make_builtin_tokenizer_provider(
    std::vector<TokenizerPreTokenizerRule> rules = {});

}
