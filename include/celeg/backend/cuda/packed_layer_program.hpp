#pragma once

#include "celeg/model/resolved.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace celeg {

enum class PackedLayerKind { Attention, ShortConvolution };

struct PackedLayerBinding {
    PackedLayerKind kind;
};

// Immutable operator schedule compiled from the resolved neutral topology.
// Runtime execution validates concrete layer bindings against this schedule;
// it does not rediscover architecture behavior from checkpoint identity.
class PackedLayerProgram {
public:
    static PackedLayerProgram compile(const RuntimeTopology& shape);

    size_t size() const { return layers_.size(); }
    PackedLayerKind kind(size_t layer) const { return layers_.at(layer).kind; }

private:
    explicit PackedLayerProgram(std::vector<PackedLayerBinding> layers)
        : layers_(std::move(layers)) {}

    std::vector<PackedLayerBinding> layers_;
};

} // namespace celeg
