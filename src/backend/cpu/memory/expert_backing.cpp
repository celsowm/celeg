#include "../detail/model_internal.hpp"

#include "celeg/checkpoint/tensor_names.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>

namespace celeg {

void configure_cpu_expert_backing(CpuCompiledModel::Shared& shared) {
    if (shared.options.expert_backing != CpuExpertBacking::DiskCached ||
        shared.shape.num_experts <= 0) {
        return;
    }

    std::lock_guard lock(shared.expert_pack_mutex);
    if (shared.expert_cache) return;

    // Native GGUF matrices already point into the memory-mapped checkpoint and
    // let the OS page cache provide SSD-backed demand paging without copying
    // every expert into owned RAM.
    if (shared.native_checkpoint) return;
    if (!shared.options.use_pack_cache || shared.pack_file.empty()) {
        throw std::invalid_argument(
            "CPU disk-backed experts require the CPU pack cache");
    }
    if (!std::filesystem::exists(shared.pack_file)) {
        throw std::runtime_error(
            "CPU expert pack is missing after weight preparation: " +
            shared.pack_file.string());
    }

    shared.expert_pack_reader =
        std::make_unique<CpuPackReader>(shared.pack_file);
    shared.expert_cache =
        std::make_unique<CpuExpertCache>(shared.options.expert_cache_bytes);

    for (std::size_t index = 0; index < shared.weight_store.layers.size(); ++index) {
        auto* moe = std::get_if<CpuCompiledModel::MoeWeights>(
            &shared.weight_store.layers[index]);
        if (!moe) continue;
        moe->layer_index = static_cast<int>(index);
        moe->disk_cached = true;
        moe->expert_w13.clear();
        moe->expert_w13.shrink_to_fit();
        moe->expert_w2.clear();
        moe->expert_w2.shrink_to_fit();
    }
}

std::shared_ptr<const CpuExpertWeights>
CpuCompiledModel::Shared::acquire_expert(int layer, int expert) {
    if (!expert_cache || !expert_pack_reader) {
        throw std::logic_error("CPU disk-backed expert cache is not configured");
    }
    if (layer < 0 || layer >= shape.num_hidden_layers ||
        expert < 0 || expert >= shape.num_experts) {
        throw std::out_of_range("CPU expert cache request is out of range");
    }

    return expert_cache->acquire(layer, expert, [this, layer, expert]() {
        const std::string prefix =
            "feed_forward.experts." + std::to_string(expert);
        const std::string w13_name =
            layer_name(layer, prefix + ".w13.weight");
        const std::string w2_name =
            layer_name(layer, prefix + ".w2.weight");

        std::lock_guard lock(expert_pack_mutex);
        CpuExpertWeights weights;
        weights.w13 = CpuLinearWeight::from_q4(
            expert_pack_reader->read_q4_matrix(w13_name));
        weights.w2 = CpuLinearWeight::from_q4(
            expert_pack_reader->read_q4_matrix(w2_name));
        return weights;
    });
}

} // namespace celeg
