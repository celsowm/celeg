#pragma once

#include "lfm/model/config/config.hpp"
#include "lfm/model/definition.hpp"
#include "lfm/model/weights/roles.hpp"

#include <memory>
#include <string_view>
#include <vector>

namespace lfm {

// Architecture providers own checkpoint interpretation. This contract is
// intentionally limited to metadata and common definitions; CUDA operators,
// allocation, and execution programs remain backend responsibilities.
class IArchitectureProvider {
public:
    virtual ~IArchitectureProvider() = default;
    virtual std::string_view id() const = 0;
    virtual bool supports(const ModelConfig& config) const = 0;
    virtual ModelDefinition inspect(const ModelConfig& config) const = 0;
    virtual const ITensorNamingPolicy& tensor_naming() const = 0;
};

class ArchitectureRegistry {
public:
    static ArchitectureRegistry& instance();

    void register_provider(std::unique_ptr<IArchitectureProvider> provider);
    const IArchitectureProvider& select(const ModelConfig& config) const;
    const IArchitectureProvider* find(std::string_view id) const;
    std::vector<std::string_view> ids() const;

private:
    ArchitectureRegistry() = default;
    std::vector<std::unique_ptr<IArchitectureProvider>> providers_;
};

// The LFM provider is the first concrete implementation of the generic
// metadata seam. Its architecture-specific layer schedule and MoE topology
// remain in the LFM config/variant layer.
class LfmArchitectureProvider final : public IArchitectureProvider {
public:
    std::string_view id() const override;
    bool supports(const ModelConfig& config) const override;
    ModelDefinition inspect(const ModelConfig& config) const override;
    const ITensorNamingPolicy& tensor_naming() const override;
};

class GraniteArchitectureProvider final : public IArchitectureProvider {
public:
    std::string_view id() const override;
    bool supports(const ModelConfig& config) const override;
    ModelDefinition inspect(const ModelConfig& config) const override;
    const ITensorNamingPolicy& tensor_naming() const override;
};

void register_builtin_architecture_providers();

} // namespace lfm
