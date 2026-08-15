#include "celeg/backend/cuda/moe/offload.hpp"
#include "celeg/detail/binary_codec.hpp"

#include <algorithm>
#include <array>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <fstream>
#include <filesystem>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fileapi.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace celeg {

namespace {

// BF16 element size. Kept as a plain constant so this host translation unit
// does not depend on the CUDA toolkit headers (__nv_bfloat16).
constexpr std::size_t kBf16Bytes = 2;

std::string format_bytes(std::size_t bytes) {
    static constexpr const char* units[] = {"B", "KiB", "MiB", "GiB"};
    double value = static_cast<double>(bytes);
    int unit = 0;
    while (value >= 1024.0 && unit < 3) {
        value /= 1024.0;
        ++unit;
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(unit == 0 ? 0 : 2)
        << value << ' ' << units[unit];
    return out.str();
}

const char* host_mode_name(ExpertHostMode mode) {
    switch (mode) {
        case ExpertHostMode::Mapped: return "mapped";
        case ExpertHostMode::PinnedCopy: return "pinned-copy";
        case ExpertHostMode::Staged: return "staged";
    }
    return "unknown";
}

const char* policy_name(ExpertCachePolicy policy) {
    switch (policy) {
        case ExpertCachePolicy::Static: return "static";
        case ExpertCachePolicy::Lru: return "lru";
        case ExpertCachePolicy::LayerLocalLfuLru: return "lfu-lru";
    }
    return "unknown";
}

} // namespace

std::size_t bytes_per_expert_bf16(const CompiledModelProgram& program) {
    int moe_intermediate = 0;
    for (const CompiledLayerProgram& layer : program.layers) {
        if (const auto* moe = std::get_if<MoeLayerProgram>(&layer.feed_forward)) {
            moe_intermediate = std::max(moe_intermediate,
                                        moe->routed.mlp.intermediate_size);
        }
    }
    const std::size_t gate_up = static_cast<std::size_t>(2) *
        static_cast<std::size_t>(moe_intermediate) *
        static_cast<std::size_t>(program.hidden);
    const std::size_t down = static_cast<std::size_t>(program.hidden) *
        static_cast<std::size_t>(moe_intermediate);
    return (gate_up + down) * kBf16Bytes;
}

int moe_layer_count(const CompiledModelProgram& program) {
    int count = 0;
    for (const CompiledLayerProgram& layer : program.layers) {
        if (std::holds_alternative<MoeLayerProgram>(layer.feed_forward)) ++count;
    }
    return count;
}

std::size_t kv_cache_bytes(const CompiledModelProgram& program, int context_tokens) {
    if (context_tokens <= 0) return 0;
    std::size_t bytes = 0;
    for (const CompiledLayerProgram& layer : program.layers) {
        const auto* compiled = std::get_if<CompiledAttentionProgram>(&layer.mixer);
        if (!compiled) continue;
        const AttentionSpec& layout = compiled->semantics;
        if (layout.kv_sharing.shared() && !layout.kv_sharing.publishes) continue;
        bytes += static_cast<std::size_t>(2) *
            static_cast<std::size_t>(layout.key_value_width()) * kBf16Bytes *
            static_cast<std::size_t>(context_tokens);
    }
    return bytes;
}

ExpertOffloadPlan plan_expert_offload(const ExpertOffloadPlanInputs& inputs) {
    const CompiledModelProgram& program = inputs.program;
    const ExpertOffloadOptions& options = inputs.options;

    int max_experts = 0;
    int max_experts_per_token = 0;
    for (const CompiledLayerProgram& layer : program.layers) {
        const auto* moe = std::get_if<MoeLayerProgram>(&layer.feed_forward);
        if (!moe) continue;
        max_experts = std::max(max_experts, moe->router.expert_count);
        max_experts_per_token = std::max(max_experts_per_token,
                                         moe->router.experts_per_token);
    }

    ExpertOffloadPlan plan;
    plan.moe_layers = moe_layer_count(program) + std::max(0, inputs.extra_moe_layers);
    plan.bytes_per_expert = bytes_per_expert_bf16(program);
    plan.gpu_free_bytes = inputs.gpu_free_bytes;
    plan.non_expert_weight_bytes = inputs.non_expert_weight_bytes;
    plan.kv_reservation_bytes = kv_cache_bytes(program, inputs.context_tokens) +
        inputs.extra_kv_reservation_bytes;
    plan.workspace_bytes = inputs.workspace_bytes;
    plan.reserve_bytes = options.gpu_memory_reserve_bytes;
    plan.host_mode = options.host_mode;
    plan.policy = options.policy;

    if (!options.enabled() || plan.moe_layers == 0 || max_experts <= 0) {
        plan.enabled = false;
        plan.experts_per_layer = max_experts;
        plan.host_experts_per_layer = 0;
        plan.gpu_expert_cache_bytes =
            static_cast<std::size_t>(max_experts) *
            static_cast<std::size_t>(plan.moe_layers) * plan.bytes_per_expert;
        return plan;
    }

    plan.enabled = true;

    int slots_per_layer = 0;
    if (options.experts_per_layer > 0) {
        slots_per_layer = options.experts_per_layer;
    } else if (options.gpu_expert_cache_bytes > 0) {
        slots_per_layer = static_cast<int>(
            options.gpu_expert_cache_bytes /
            (static_cast<std::size_t>(plan.moe_layers) * plan.bytes_per_expert));
    } else {
        // Auto: derive the cache budget from free VRAM.
        const std::size_t fixed = plan.non_expert_weight_bytes +
            plan.kv_reservation_bytes + plan.workspace_bytes + plan.reserve_bytes;
        std::size_t available_for_experts = 0;
        if (plan.gpu_free_bytes > fixed) {
            available_for_experts = plan.gpu_free_bytes - fixed;
        }
        slots_per_layer = static_cast<int>(
            available_for_experts /
            (static_cast<std::size_t>(plan.moe_layers) * plan.bytes_per_expert));
    }

    // Every routed token may select K distinct experts.  A disk-backed cache
    // smaller than K cannot execute even a single token without a resident
    // pointer for every routed expert, so make the topology's routing width a
    // hard lower bound for the planner.
    const int min_experts = std::max(
        0, std::min(std::max(options.minimum_experts_per_layer,
                             max_experts_per_token),
                    max_experts));
    if (slots_per_layer < min_experts) {
        std::ostringstream msg;
        msg << "expert offload cannot fit the minimum "
            << min_experts << " experts/layer on the GPU (room for "
            << slots_per_layer << ")";
        throw std::runtime_error(msg.str());
    }
    slots_per_layer = std::min(slots_per_layer, max_experts);

    // Respect the host pinned budget: every non-resident expert may need host
    // storage. If it would overflow, force more experts onto the GPU.
    if (options.host_mode == ExpertHostMode::PinnedCopy &&
        options.maximum_pinned_host_bytes > 0) {
        const std::size_t per_layer_host_bytes =
            static_cast<std::size_t>(plan.moe_layers) * plan.bytes_per_expert;
        const int max_host_per_layer = static_cast<int>(
            options.maximum_pinned_host_bytes / per_layer_host_bytes);
        const int min_gpu_for_host = max_experts - max_host_per_layer;
        slots_per_layer = std::max(slots_per_layer, min_gpu_for_host);
        slots_per_layer = std::min(slots_per_layer, max_experts);
    }

    plan.experts_per_layer = slots_per_layer;
    plan.host_experts_per_layer = max_experts - slots_per_layer;
    // Prefetch depth cannot exceed the experts that are actually host-resident
    // (no point prefetching what is already cached).
    if (options.backing == ExpertBackingMode::DiskCached) {
        plan.prefetch_experts = 0;
    } else {
        plan.prefetch_experts = std::max(
            0, std::min(options.prefetch_experts, plan.host_experts_per_layer));
    }
    plan.gpu_expert_cache_bytes =
        static_cast<std::size_t>(slots_per_layer) *
        static_cast<std::size_t>(plan.moe_layers) * plan.bytes_per_expert;
    plan.host_expert_bytes =
        static_cast<std::size_t>(plan.host_experts_per_layer) *
        static_cast<std::size_t>(plan.moe_layers) * plan.bytes_per_expert;
    return plan;
}

std::string ExpertOffloadPlan::report() const {
    std::ostringstream out;
    out << "MoE offload plan:\n";
    if (!enabled) {
        out << "  enabled:                  no (all experts GPU-resident)\n"
            << "  experts per layer:        " << experts_per_layer << " / "
            << experts_per_layer << '\n';
        return out.str();
    }
    out << "  GPU memory free:          " << format_bytes(gpu_free_bytes) << '\n'
        << "  non-expert weights:       " << format_bytes(non_expert_weight_bytes) << '\n'
        << "  KV reservation:           " << format_bytes(kv_reservation_bytes) << '\n'
        << "  workspace/headroom:       "
        << format_bytes(workspace_bytes + reserve_bytes) << '\n'
        << "  GPU expert cache:         " << format_bytes(gpu_expert_cache_bytes) << '\n'
        << "  experts per layer:        " << experts_per_layer << " / "
        << (experts_per_layer + host_experts_per_layer) << '\n'
        << "  host expert storage:      " << format_bytes(host_expert_bytes) << '\n'
        << "  bytes per expert:         " << format_bytes(bytes_per_expert) << '\n'
        << "  host mode:                " << host_mode_name(host_mode) << '\n'
        << "  cache policy:             " << policy_name(policy) << '\n'
        << "  prefetch depth:           " << prefetch_experts << '\n';
    return out.str();
}

// ---------------------------------------------------------------------------
// ExpertSidecar Implementation
// ---------------------------------------------------------------------------

struct SidecarHeader {
    char magic[8];
    std::uint32_t num_layers;
    std::uint32_t num_experts;
    std::uint64_t moe_intermediate;
    std::uint64_t hidden;
    std::uint64_t reserved[4];
};

constexpr size_t kSidecarHeaderBytes = 64;
constexpr size_t kSidecarIndexBytes = 32;

SidecarHeader decode_header(std::span<const std::byte> bytes) {
    if (bytes.size() != kSidecarHeaderBytes) {
        throw std::invalid_argument("invalid sidecar header size");
    }
    SidecarHeader header{};
    std::memcpy(header.magic, bytes.data(), sizeof(header.magic));
    size_t offset = sizeof(header.magic);
    header.num_layers = binary::read_le<uint32_t>(bytes, offset);
    header.num_experts = binary::read_le<uint32_t>(bytes, offset);
    header.moe_intermediate = binary::read_le<uint64_t>(bytes, offset);
    header.hidden = binary::read_le<uint64_t>(bytes, offset);
    for (uint64_t& value : header.reserved) value = binary::read_le<uint64_t>(bytes, offset);
    return header;
}

SidecarExpertIndex decode_index(std::span<const std::byte> bytes) {
    size_t offset = 0;
    SidecarExpertIndex index;
    index.gate_up_offset = binary::read_le<uint64_t>(bytes, offset);
    index.gate_up_bytes = binary::read_le<uint64_t>(bytes, offset);
    index.down_offset = binary::read_le<uint64_t>(bytes, offset);
    index.down_bytes = binary::read_le<uint64_t>(bytes, offset);
    return index;
}

ExpertSidecar::~ExpertSidecar() {
#if defined(_WIN32)
    if (file_handle_) {
        ::CloseHandle(file_handle_);
        file_handle_ = nullptr;
    }
#else
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
}

bool ExpertSidecar::load(const std::string& path, int expected_layers, int expected_experts,
                         std::uint64_t expected_inter, std::uint64_t expected_hidden) {
#if defined(_WIN32)
    file_handle_ = ::CreateFileA(
        path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_handle_ == INVALID_HANDLE_VALUE) {
        file_handle_ = nullptr;
        return false;
    }
    LARGE_INTEGER size{};
    if (::GetFileSizeEx(file_handle_, &size) == 0) {
        ::CloseHandle(file_handle_);
        file_handle_ = nullptr;
        return false;
    }
    file_size_ = static_cast<std::uint64_t>(size.QuadPart);
#else
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
        return false;
    }
    struct stat st {};
    if (::fstat(fd_, &st) != 0) {
        ::close(fd_);
        fd_ = -1;
        return false;
    }
    file_size_ = static_cast<std::uint64_t>(st.st_size);
#endif

    if (file_size_ < kSidecarHeaderBytes) {
        return false;
    }

    // Read header using positioned read
    std::array<std::byte, kSidecarHeaderBytes> header_bytes{};
#if defined(_WIN32)
    OVERLAPPED overlapped = {};
    DWORD bytes_read = 0;
    if (!::ReadFile(file_handle_, header_bytes.data(), static_cast<DWORD>(header_bytes.size()), &bytes_read, &overlapped) || bytes_read != header_bytes.size()) {
        return false;
    }
#else
    if (::pread(fd_, header_bytes.data(), header_bytes.size(), 0) != header_bytes.size()) {
        return false;
    }
#endif
    const SidecarHeader header = decode_header(header_bytes);

    // Validate magic, layers, experts, and shapes
    if (std::memcmp(header.magic, "LFMSIDE2", 8) != 0) {
        return false;
    }
    if (header.num_layers != static_cast<std::uint32_t>(expected_layers) ||
        header.num_experts != static_cast<std::uint32_t>(expected_experts) ||
        header.moe_intermediate != expected_inter ||
        header.hidden != expected_hidden) {
        return false;
    }

    // Read index
    index_.resize(expected_layers, std::vector<SidecarExpertIndex>(expected_experts));
    std::uint64_t index_offset = kSidecarHeaderBytes;
    for (int l = 0; l < expected_layers; ++l) {
        for (int expert = 0; expert < expected_experts; ++expert) {
            std::array<std::byte, kSidecarIndexBytes> index_bytes{};
#if defined(_WIN32)
            overlapped.Offset = static_cast<DWORD>(index_offset & 0xFFFFFFFF);
            overlapped.OffsetHigh = static_cast<DWORD>((index_offset >> 32) & 0xFFFFFFFF);
            if (!::ReadFile(file_handle_, index_bytes.data(), static_cast<DWORD>(index_bytes.size()), &bytes_read, &overlapped) || bytes_read != index_bytes.size()) return false;
#else
            if (::pread(fd_, index_bytes.data(), index_bytes.size(), index_offset) != index_bytes.size()) return false;
#endif
            index_[l][expert] = decode_index(index_bytes);
            index_offset += kSidecarIndexBytes;
        }
    }

    return true;
}

void ExpertSidecar::read_expert(int layer_idx, int expert_id, std::span<std::byte> gu_dest, std::span<std::byte> dn_dest) const {
    if (!valid()) {
        throw std::runtime_error("Sidecar is not loaded");
    }
    if (layer_idx < 0 || layer_idx >= static_cast<int>(index_.size()) ||
        expert_id < 0 || expert_id >= static_cast<int>(index_[layer_idx].size())) {
        throw std::out_of_range("Sidecar: layer or expert index out of range");
    }

    const SidecarExpertIndex& idx = index_[layer_idx][expert_id];
    if (gu_dest.size() != idx.gate_up_bytes || dn_dest.size() != idx.down_bytes) {
        throw std::invalid_argument("Sidecar read: destination sizes do not match packed sizes");
    }

#if defined(_WIN32)
    // Read gate_up
    OVERLAPPED overlapped = {};
    overlapped.Offset = static_cast<DWORD>(idx.gate_up_offset & 0xFFFFFFFF);
    overlapped.OffsetHigh = static_cast<DWORD>((idx.gate_up_offset >> 32) & 0xFFFFFFFF);
    DWORD bytes_read = 0;
    if (!::ReadFile(file_handle_, gu_dest.data(), static_cast<DWORD>(idx.gate_up_bytes), &bytes_read, &overlapped) ||
        bytes_read != idx.gate_up_bytes) {
        throw std::runtime_error("Sidecar ReadFile failed for gate_up");
    }

    // Read down
    overlapped.Offset = static_cast<DWORD>(idx.down_offset & 0xFFFFFFFF);
    overlapped.OffsetHigh = static_cast<DWORD>((idx.down_offset >> 32) & 0xFFFFFFFF);
    if (!::ReadFile(file_handle_, dn_dest.data(), static_cast<DWORD>(idx.down_bytes), &bytes_read, &overlapped) ||
        bytes_read != idx.down_bytes) {
        throw std::runtime_error("Sidecar ReadFile failed for down");
    }
#else
    // Read gate_up
    if (static_cast<std::size_t>(::pread(fd_, gu_dest.data(), idx.gate_up_bytes, idx.gate_up_offset)) != idx.gate_up_bytes) {
        throw std::runtime_error("Sidecar pread failed for gate_up");
    }
    // Read down
    if (static_cast<std::size_t>(::pread(fd_, dn_dest.data(), idx.down_bytes, idx.down_offset)) != idx.down_bytes) {
        throw std::runtime_error("Sidecar pread failed for down");
    }
#endif
}

// ---------------------------------------------------------------------------
// ModelUsageStats Implementation
// ---------------------------------------------------------------------------

bool ModelUsageStats::load(const std::string& path, int expected_layers, int expected_experts) {
    std::ifstream in(path);
    if (!in) {
        return false;
    }
    int num_layers = 0;
    int num_experts = 0;
    if (!(in >> num_layers >> num_experts)) {
        return false;
    }
    if (num_layers != expected_layers || num_experts != expected_experts) {
        return false;
    }

    layers.assign(static_cast<size_t>(num_layers), std::vector<ExpertUsageEntry>(static_cast<size_t>(num_experts)));
    int l = 0, e = 0;
    while (in >> l >> e) {
        if (l < 0 || l >= num_layers || e < 0 || e >= num_experts) {
            return false;
        }
        auto& entry = layers[static_cast<size_t>(l)][static_cast<size_t>(e)];
        if (!(in >> entry.selection_count >> entry.recent_heat >> entry.last_used_sequence
                 >> entry.ram_cache_hits >> entry.gpu_cache_hits >> entry.ssd_misses)) {
            return false;
        }
    }
    return true;
}

void ModelUsageStats::save(const std::string& path) const {
    if (layers.empty()) return;
    std::string tmp_path = path + ".tmp";
    std::ofstream out(tmp_path);
    if (!out) {
        return;
    }

    out << layers.size() << " " << layers[0].size() << "\n";
    for (size_t l = 0; l < layers.size(); ++l) {
        for (size_t e = 0; e < layers[l].size(); ++e) {
            const auto& entry = layers[l][e];
            out << l << " " << e << " "
                << entry.selection_count << " "
                << entry.recent_heat << " "
                << entry.last_used_sequence << " "
                << entry.ram_cache_hits << " "
                << entry.gpu_cache_hits << " "
                << entry.ssd_misses << "\n";
        }
    }
    out.close();

    // Atomic rename
    std::error_code ec;
    std::filesystem::rename(tmp_path, path, ec);
}

} // namespace celeg