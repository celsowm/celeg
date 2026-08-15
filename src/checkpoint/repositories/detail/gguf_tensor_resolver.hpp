#pragma once

#include "celeg/checkpoint/formats/gguf.hpp"
#include "celeg/checkpoint/tensor.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace celeg {

class GgufTensorResolver final {
public:
    explicit GgufTensorResolver(std::shared_ptr<GgufFile> file);

    bool contains(std::string_view canonical_name) const;
    HostTensorView tensor(std::string_view canonical_name) const;
    std::vector<std::string> names() const;

private:
    std::shared_ptr<GgufFile> file_;
};

}
