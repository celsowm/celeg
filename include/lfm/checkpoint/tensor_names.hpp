#pragma once

// Generates canonical checkpoint tensor names for a given layer index.
// Used by the CPU weight loader, the CUDA weight loader, and the expert
// loader to avoid string-formatting duplication across backends.

#include <string>

namespace lfm {

inline std::string layer_name(int index, const std::string& suffix) {
    return "model.layers." + std::to_string(index) + "." + suffix;
}

} // namespace lfm
