#include "gguf_tensor_resolver.hpp"

#include "gguf_tensor_adapter.hpp"
#include "gguf_tensor_mapper.hpp"

#include <stdexcept>

namespace celeg {

GgufTensorResolver::GgufTensorResolver(std::shared_ptr<GgufFile> file)
    : file_(std::move(file)) {
    if (!file_) throw std::invalid_argument("GGUF tensor resolver requires a file");
}

bool GgufTensorResolver::contains(std::string_view canonical_name) const {
    const GgufTensorReference reference =
        GgufTensorNameMapper::resolve(canonical_name);
    return !reference.native_name.empty() &&
        file_->contains_tensor(reference.native_name);
}

HostTensorView GgufTensorResolver::tensor(std::string_view canonical_name) const {
    const GgufTensorReference reference =
        GgufTensorNameMapper::resolve(canonical_name);
    if (reference.native_name.empty()) {
        throw std::out_of_range("gguf: no mapping for tensor " +
                                std::string(canonical_name));
    }
    const GgufTensorView view = file_->tensor(reference.native_name);
    if (reference.is_expert_slice()) {
        return GgufTensorViewAdapter::adapt_expert(view, reference);
    }
    return GgufTensorViewAdapter::adapt(view, reference.rows_rope_permuted);
}

std::vector<std::string> GgufTensorResolver::names() const {
    return file_->tensor_names();
}

} // namespace celeg
