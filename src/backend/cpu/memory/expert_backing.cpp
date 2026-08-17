#include "../detail/model_internal.hpp"

#include "celeg/checkpoint/tensor_names.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace celeg {

CpuExpertBackingStore::CpuExpertBackingStore(
    const std::filesystem::path& pack_path, std::size_t budget_bytes,
    int hidden_size, int intermediate_size)
    : reader_(std::make_unique<CpuPackReader>(pack_path)),
      cache_(std::make_unique<CpuExpertCache>(budget_bytes)),
      hidden_size_(hidden_size), intermediate_size_(intermediate_size) {
    if (hidden_size_ <= 0 || intermediate_size_ <= 0) {
        throw std::invalid_argument("CPU expert backing has invalid dimensions");
    }
}

std::shared_ptr<const CpuExpertWeights>
CpuExpertBackingStore::acquire(ExpertKey key) {
    key.validate();
    return cache_->acquire(key.layer, key.expert, [this, key]() {
        const std::string prefix =
            "feed_forward.experts." + std::to_string(key.expert);
        const std::string w13_name =
            layer_name(key.layer, prefix + ".w13.weight");
        const std::string w2_name =
            layer_name(key.layer, prefix + ".w2.weight");
        CpuExpertWeights weights;
        weights.w13 = CpuLinearWeight::from_q4(reader_->read_q4_matrix(w13_name));
        weights.w2 = CpuLinearWeight::from_q4(reader_->read_q4_matrix(w2_name));
        if (weights.w13.rows != static_cast<uint32_t>(2 * intermediate_size_) ||
            weights.w13.cols != static_cast<uint32_t>(hidden_size_) ||
            weights.w2.rows != static_cast<uint32_t>(hidden_size_) ||
            weights.w2.cols != static_cast<uint32_t>(intermediate_size_)) {
            throw std::runtime_error("CPU expert pack entry has unexpected dimensions");
        }
        return weights;
    });
}

CpuExpertBackingMetrics CpuExpertBackingStore::metrics() const {
    return {cache_->resident_bytes(), cache_->hits(), cache_->misses(),
            cache_->coalesced_waits(), cache_->evictions()};
}

void configure_cpu_expert_backing(CpuCompiledModel::Shared& shared) {
    if (shared.options.expert_backing != CpuExpertBacking::DiskCached ||
        !shared.program.has_moe()) {
        return;
    }

    std::lock_guard lock(shared.expert_pack_mutex);
    if (shared.expert_backing_store) return;

    if (shared.checkpoint.native_checkpoint) return;
    if (!shared.options.use_pack_cache || shared.checkpoint.pack_file.empty()) {
        throw std::invalid_argument(
            "CPU disk-backed experts require the CPU pack cache");
    }
    if (!std::filesystem::exists(shared.checkpoint.pack_file)) {
        throw std::runtime_error(
            "CPU expert pack is missing after weight preparation: " +
            shared.checkpoint.pack_file.string());
    }

    const auto moe_program = std::find_if(shared.program.layers.begin(),
                                           shared.program.layers.end(),
        [](const CompiledLayerProgram& layer) {
            return std::holds_alternative<MoeLayerProgram>(layer.feed_forward);
        });
    if (moe_program == shared.program.layers.end()) {
        throw std::runtime_error("cannot create expert backing without MoE semantics");
    }
    const int intermediate =
        std::get<MoeLayerProgram>(moe_program->feed_forward).routed.mlp.intermediate_size;
    shared.expert_backing_store = std::make_unique<CpuExpertBackingStore>(
        shared.checkpoint.pack_file, shared.options.expert_cache_bytes,
        shared.program.hidden, intermediate);

    for (std::size_t index = 0; index < shared.weight_store.layers.size(); ++index) {
        auto* moe = std::get_if<CpuCompiledModel::MoeWeights>(
            &shared.weight_store.layers[index].feed_forward);
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
    if (!expert_backing_store) {
        throw std::logic_error("CPU disk-backed expert cache is not configured");
    }
    if (layer < 0 || layer >= static_cast<int>(program.layers.size())) {
        throw std::out_of_range("CPU expert cache request is out of range");
    }
    const auto* moe = std::get_if<MoeLayerProgram>(
        &program.layers[static_cast<size_t>(layer)].feed_forward);
    if (!moe || expert < 0 || expert >= moe->router.expert_count) {
        throw std::out_of_range("CPU expert cache request is out of range");
    }

    return expert_backing_store->acquire(ExpertKey{layer, expert});
}

}
