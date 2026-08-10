#pragma once

#include "celeg/checkpoint/view.hpp"
#include "celeg/model/resolved.hpp"

#include <filesystem>
#include <functional>
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

// Family resolution is an ordered composition of neutral stages. Each stage
// has one responsibility and receives only the checkpoint/model data it
// needs; a family may still implement the policies in its own directory.
struct ArchitectureResolutionStages {
    std::function<RuntimeTopology(const CheckpointView&)> topology;
    std::function<void(ResolvedModel&, const CheckpointView&)> graph;
    std::function<void(ResolvedModel&, const CheckpointView&)> weights;
    ModelCapabilities capabilities;
    ModelProvenance provenance;
};

ResolvedModel resolve_architecture_stages(
    const CheckpointView& checkpoint, ArchitectureResolutionStages stages);

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

// Registers the evidence-driven checkpoint provider. It is deliberately a
// fallback provider: explicit descriptors/importers outrank it when they
// supply semantics that cannot be inferred from the checkpoint itself.
std::unique_ptr<IArchitecture> make_automatic_architecture();

} // namespace celeg
