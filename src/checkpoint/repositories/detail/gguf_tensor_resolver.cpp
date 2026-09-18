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
    } else if (name.ends_with(".ffn_gate_inp_shexp.weight") &&
               result.shape.size() == 1) {
        /// llama.cpp flattens the shared-expert gate row ([1, hidden] in HF
        /// checkpoints) to a vector; restore the logical row shape so the
        /// gate binds and loads like its HF counterpart. Element count and
        /// data layout are unchanged.
        result.shape.insert(result.shape.begin(), 1);
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
    std::vector<std::string> out = file_->tensor_names();
    /// Canonical aliases for GGUF MoE tensors so canonical-addressed binding
    /// (`bind_moe`) works off tensor inventory: stacked `ffn_*_exps` publish
    /// per-expert `feed_forward.experts.{E}.w{1,3,2}` names (slices served by
    /// `tensor()` via the mapper), and router/shared-expert tensors publish
    /// their `mlp.*` canonical spellings. Only emitted when the native
    /// tensor exists. Dialect-generic (llama.cpp layout), no per-arch names.
    const auto publish = [&](std::string_view native_suffix,
                             std::string_view canonical_suffix) {
        for (const std::string& name : file_->tensor_names()) {
            /// Exact `blk.<digits>.<native_suffix>` shape; the layer digits
            /// are validated so `blk.foo.<suffix>` cannot sneak through.
            constexpr std::string_view block_prefix = "blk.";
            if (!name.starts_with(block_prefix) ||
                !name.ends_with(native_suffix)) {
                continue;
            }
            const size_t dot = name.find('.', block_prefix.size());
            if (dot == std::string::npos ||
                name.size() != dot + 1 + native_suffix.size()) {
                continue;
            }
            bool digits = dot > block_prefix.size();
            for (size_t i = block_prefix.size(); digits && i < dot; ++i) {
                digits = name[i] >= '0' && name[i] <= '9';
            }
            if (!digits) continue;
            out.push_back("model.layers." +
                          name.substr(block_prefix.size(),
                                      dot - block_prefix.size()) +
                          "." + std::string(canonical_suffix));
        }
    };
    publish("ffn_gate_inp.weight", "mlp.gate.weight");
    publish("ffn_expert_bias.weight", "feed_forward.expert_bias.weight");
    publish("ffn_gate_inp_shexp.weight", "mlp.shared_expert_gate.weight");
    publish("ffn_gate_shexp.weight", "mlp.shared_experts.gate_proj.weight");
    publish("ffn_up_shexp.weight", "mlp.shared_experts.up_proj.weight");
    publish("ffn_down_shexp.weight", "mlp.shared_experts.down_proj.weight");
    for (const std::string& name : file_->tensor_names()) {
        /// `blk.<L>.ffn_<gate|up|down>_exps.weight`, rank 3, expert dim last.
        constexpr std::string_view block_prefix = "blk.";
        constexpr std::string_view proj_marker = ".ffn_";
        constexpr std::string_view stacked_suffix = "_exps.weight";
        if (!name.starts_with(block_prefix) ||
            !name.ends_with(stacked_suffix)) {
            continue;
        }
        const size_t dot = name.find('.', block_prefix.size());
        if (dot == std::string::npos ||
            name.compare(dot + 1, proj_marker.size() - 1, "ffn_") != 0) {
            continue;
        }
        const std::string layer =
            name.substr(block_prefix.size(), dot - block_prefix.size());
        const size_t middle_begin = dot + 1 + (proj_marker.size() - 1);
        const size_t middle_end = name.size() - stacked_suffix.size();
        if (middle_end <= middle_begin) continue;
        const std::string middle =
            name.substr(middle_begin, middle_end - middle_begin);
        std::string_view canonical;
        if (middle == "gate") canonical = "w1.weight";
        else if (middle == "up") canonical = "w3.weight";
        else if (middle == "down") canonical = "w2.weight";
        else continue;
        const GgufTensorInfo& info = file_->tensor_info(name);
        /// File order keeps the expert dim last; `hf_shape()` reverses it
        /// first, which is what the slice adapter consumes.
        if (info.dims.size() != 3 || info.dims.back() == 0) continue;
        for (uint64_t expert = 0; expert < info.dims.back(); ++expert) {
            out.push_back("model.layers." + layer + ".feed_forward.experts." +
                          std::to_string(expert) + "." + std::string(canonical));
        }
    }
    return out;
}

}
