#pragma once

#include <string>
#include <string_view>

namespace celeg {

struct GgufTensorReference {
    std::string native_name;
    int expert = -1;

    bool is_expert_slice() const noexcept { return expert >= 0; }
};

class GgufTensorNameMapper final {
public:
    static GgufTensorReference resolve(std::string_view canonical_name);
};

}
