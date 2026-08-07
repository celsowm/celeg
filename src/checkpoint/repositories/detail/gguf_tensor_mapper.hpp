#pragma once

#include <string>
#include <string_view>

namespace celeg {

struct GgufTensorReference {
    std::string native_name;
    int expert = -1;
    bool rows_rope_permuted = false;

    bool is_expert_slice() const noexcept { return expert >= 0; }
};

// Resolves canonical model tensor names into the native GGUF tensor name.
// This is a format-boundary mapping; it does not interpret model semantics.
class GgufTensorNameMapper final {
public:
    static GgufTensorReference resolve(std::string_view canonical_name);
};

} // namespace celeg
