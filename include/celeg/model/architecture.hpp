#pragma once

#include "celeg/checkpoint/view.hpp"
#include "celeg/model/resolved.hpp"

#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

namespace celeg {

class IWeightRepository;

struct ProbeResult {
    bool supported = false;
    int specificity = 0;
    std::string reason;
};

// An architecture owns checkpoint interpretation and produces a fully
// resolved, backend-neutral model. Backends never inspect architecture IDs.
class IArchitecture {
public:
    virtual ~IArchitecture() = default;
    virtual std::string_view id() const = 0;
    virtual ProbeResult probe(const CheckpointMetadata& metadata) const = 0;
    virtual ResolvedModel resolve(const CheckpointView& checkpoint) const = 0;
};

class ArchitectureCatalog {
public:
    void add(std::unique_ptr<IArchitecture> architecture);
    void freeze();
    const IArchitecture& select(const CheckpointMetadata& metadata) const;
    const IArchitecture* find(std::string_view id) const;
    std::vector<std::string_view> ids() const;

private:
    bool frozen_ = false;
    std::vector<std::unique_ptr<IArchitecture>> architectures_;
};

std::shared_ptr<const ArchitectureCatalog> create_builtin_architecture_catalog();

} // namespace celeg
