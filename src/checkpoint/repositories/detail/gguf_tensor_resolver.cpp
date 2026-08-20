#include "gguf_tensor_resolver.hpp"

#include "gguf_tensor_adapter.hpp"
#include "gguf_tensor_mapper.hpp"

#include <stdexcept>

namespace celeg {

namespace {

HostTensorView adapt_native_tensor(const GgufTensorView& view, std::string_view name) {
    HostTensorView result = GgufTensorViewAdapter::adapt(view);
    if (name.ends_with(".shortconv.conv.weight") && result.shape.size() == 2) {
        result.shape.insert(result.shape.begin() + 1, 1);
    }
    if ((name.ends_with(".ssm_a") || name.ends_with(".ssm_d")) &&
        result.shape.size() == 2 && result.shape[1] == 1) {
        result.shape.resize(1);
    } else if (name.ends_with(".ssm_norm.weight") && result.shape.size() == 2) {
        result.shape = {static_cast<int64_t>(view.element_count)};
    } else if (name.ends_with(".ssm_conv1d.weight") && result.shape.size() == 2) {
        result.shape.insert(result.shape.begin() + 1, 1);
    }
    return result;
}

}

GgufTensorResolver::GgufTensorResolver(std::shared_ptr<GgufFile> file)
    : file_(std::move(file)) {
    if (!file_) throw std::invalid_argument("GGUF tensor resolver requires a file");
}

bool GgufTensorResolver::contains(std::string_view name) const {
    if (file_->contains_tensor(name)) return true;

    const GgufTensorReference reference =
        GgufTensorNameMapper::resolve(name);
    return !reference.native_name.empty() &&
        file_->contains_tensor(reference.native_name);
}

HostTensorView GgufTensorResolver::tensor(std::string_view name) const {
    if (file_->contains_tensor(name)) {
        return adapt_native_tensor(file_->tensor(name), name);
    }

    const GgufTensorReference reference =
        GgufTensorNameMapper::resolve(name);
    if (reference.native_name.empty()) {
        throw std::out_of_range("gguf: no mapping for tensor " +
                                std::string(name));
    }
    const GgufTensorView view = file_->tensor(reference.native_name);
    if (reference.is_expert_slice()) {
        return GgufTensorViewAdapter::adapt_expert(view, reference);
    }
    return adapt_native_tensor(view, reference.native_name);
}

std::vector<std::string> GgufTensorResolver::names() const {
    return file_->tensor_names();
}

}
