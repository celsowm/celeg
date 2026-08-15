#include "celeg/checkpoint/repositories/gguf.hpp"

#include "detail/gguf_tensor_resolver.hpp"

#include <stdexcept>

namespace celeg {

GgufRepository::GgufRepository(std::shared_ptr<GgufFile> gguf)
{
    if (!gguf) throw std::invalid_argument("GgufRepository requires a GgufFile");
    tensors_ = std::make_unique<GgufTensorResolver>(std::move(gguf));
}

GgufRepository::~GgufRepository() = default;

bool GgufRepository::contains(std::string_view name) const {
    return tensors_->contains(name);
}

HostTensorView GgufRepository::tensor(std::string_view name) const {
    return tensors_->tensor(name);
}

std::vector<std::string> GgufRepository::names() const {
    return tensors_->names();
}

}
