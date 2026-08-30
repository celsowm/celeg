#pragma once

#include "celeg/model/program.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace celeg {

enum class PackedLayerKind { Attention, ShortConvolution, GatedDeltaNet, Mamba2, MlpOnly };

struct PackedLayerBinding {
    PackedLayerKind kind;
};

class PackedLayerProgram {
public:
    static PackedLayerProgram compile(const CompiledModelProgram& program);

    size_t size() const { return layers_.size(); }
    PackedLayerKind kind(size_t layer) const { return layers_.at(layer).kind; }

private:
    explicit PackedLayerProgram(std::vector<PackedLayerBinding> layers)
        : layers_(std::move(layers)) {}

    std::vector<PackedLayerBinding> layers_;
};

}
