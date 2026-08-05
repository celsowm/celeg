#pragma once

#include "celeg/checkpoint/view.hpp"

#include <filesystem>
#include <memory>
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

class BpeTokenizer;
class IVisualEmbeddingProvider;

enum class BackendKind { Cpu, Cuda };

class IBackendOptions {
public:
    virtual ~IBackendOptions() = default;
    virtual BackendKind backend() const noexcept = 0;
};

struct BackendCreateRequest {
    std::string model_path;
    int max_context = 0;
    std::shared_ptr<const RuntimeContext> runtime;
    std::shared_ptr<const IBackendOptions> options;
};

class ITokenizerProvider {
public:
    virtual ~ITokenizerProvider() = default;
    virtual std::string_view id() const = 0;
    virtual bool supports(const CheckpointView& checkpoint,
                          const std::filesystem::path& model_path) const = 0;
    virtual std::unique_ptr<BpeTokenizer> create(
        const CheckpointView& checkpoint,
        const std::filesystem::path& model_path) const = 0;
};

// A family module is the only unit allowed to contribute a coherent set of
// family-owned providers. RuntimeBuilder owns the catalogs and freezes them
// after all modules have registered.
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
    virtual bool supports(BackendKind backend) const = 0;
    virtual std::unique_ptr<serve::ServiceBundle> create(
        const BackendCreateRequest& request) const = 0;
};

class IVisionProviderFactory {
public:
    virtual ~IVisionProviderFactory() = default;
    virtual std::string_view id() const = 0;
    virtual bool supports(std::string_view architecture_id,
                          const std::filesystem::path& projector_path) const = 0;
    virtual std::shared_ptr<const IVisualEmbeddingProvider> create(
        const std::filesystem::path& projector_path) const = 0;
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

private:
    bool frozen_ = false;
    std::vector<std::unique_ptr<Provider>> providers_;
};

using TokenizerProviderCatalog = ProviderCatalog<ITokenizerProvider>;
using BackendFactoryCatalog = ProviderCatalog<IBackendFactory>;
using VisionProviderCatalog = ProviderCatalog<IVisionProviderFactory>;

std::unique_ptr<ITokenizerProvider> make_builtin_tokenizer_provider();

} // namespace celeg
