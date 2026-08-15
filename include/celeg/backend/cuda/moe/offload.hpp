#pragma once

#include "celeg/model/resolved.hpp"
#include "celeg/model/program.hpp"
#include "celeg/checkpoint/formats/safetensors.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <sstream>
#include <vector>

namespace celeg {

enum class ExpertBackingMode : uint8_t {
    HostResident,
    DiskCached
};

enum class ExpertIoBackend : uint8_t {
    Auto,
    ThreadPool,
    IoUring,
    WindowsOverlapped
};

struct ExpertLocation {
    TensorLocator w1;
    TensorLocator w2;
    TensorLocator w3;
};

struct ExpertUsageEntry {
    std::uint64_t selection_count = 0;
    double recent_heat = 0.0;
    std::uint64_t last_used_sequence = 0;
    std::uint64_t ram_cache_hits = 0;
    std::uint64_t gpu_cache_hits = 0;
    std::uint64_t ssd_misses = 0;
};

struct ModelUsageStats {
    std::vector<std::vector<ExpertUsageEntry>> layers;

    bool load(const std::string& path, int expected_layers, int expected_experts);
    void save(const std::string& path) const;
};

struct SidecarExpertIndex {
    std::uint64_t gate_up_offset = 0;
    std::uint64_t gate_up_bytes = 0;
    std::uint64_t down_offset = 0;
    std::uint64_t down_bytes = 0;
};

class ExpertSidecar {
public:
    ExpertSidecar() = default;
    ~ExpertSidecar();

    ExpertSidecar(const ExpertSidecar&) = delete;
    ExpertSidecar& operator=(const ExpertSidecar&) = delete;

    bool load(const std::string& path, int expected_layers, int expected_experts,
              std::uint64_t expected_inter, std::uint64_t expected_hidden);

    void read_expert(int layer_idx, int expert_id, std::span<std::byte> gu_dest, std::span<std::byte> dn_dest) const;
    bool valid() const { return fd_ >= 0 || file_handle_ != nullptr; }

private:
    int fd_ = -1;
    void* file_handle_ = nullptr;
    std::uint64_t file_size_ = 0;
    std::vector<std::vector<SidecarExpertIndex>> index_;
};


enum class ExpertResidence : uint8_t {
    DeviceCache,
    HostMapped,
};

enum class ExpertHostMode : uint8_t {
    Mapped,
    PinnedCopy,
    Staged,
};

enum class ExpertCachePolicy : uint8_t {
    Static,
    Lru,
    LayerLocalLfuLru,
    Score,
};

enum class ExpertOffloadMode : uint8_t {
    None,
    Auto,
    Host,
};

inline constexpr std::size_t operator"" _MiB(unsigned long long v) {
    return static_cast<std::size_t>(v) * 1024ull * 1024ull;
}
inline constexpr std::size_t operator"" _GiB(unsigned long long v) {
    return static_cast<std::size_t>(v) * 1024ull * 1024ull * 1024ull;
}

struct ExpertOffloadOptions {
    ExpertOffloadMode mode = ExpertOffloadMode::None;

    // Headroom kept free on the GPU for the driver, display, CUDA context,
    // cuBLAS workspaces and allocator fragmentation.
    std::size_t gpu_memory_reserve_bytes = 768_MiB;

    // Explicit GPU expert-cache budget in bytes; 0 means "auto" (derive from
    std::size_t gpu_expert_cache_bytes = 0;

    std::size_t maximum_pinned_host_bytes = 9_GiB;

    int experts_per_layer = 0;

    int prefill_chunk_tokens = 256;

    int minimum_experts_per_layer = 4;

    int prefetch_experts = 0;

    ExpertHostMode host_mode = ExpertHostMode::Mapped;
    ExpertCachePolicy policy = ExpertCachePolicy::LayerLocalLfuLru;

    ExpertBackingMode backing = ExpertBackingMode::HostResident;
    std::size_t host_expert_cache_bytes = 4_GiB;
    ExpertIoBackend io_backend = ExpertIoBackend::Auto;
    int io_queue_depth = 16;
    int io_workers = 4;
    bool direct_io = false;
    std::string expert_sidecar_path;
    std::string mirror_path;
    std::string usage_profile_path;

    bool enabled() const { return mode != ExpertOffloadMode::None; }

    std::string fingerprint() const {
        std::ostringstream out;
        out << static_cast<int>(mode) << ':' << gpu_memory_reserve_bytes << ':'
            << gpu_expert_cache_bytes << ':' << maximum_pinned_host_bytes << ':'
            << experts_per_layer << ':' << prefill_chunk_tokens << ':'
            << minimum_experts_per_layer << ':' << prefetch_experts << ':'
            << static_cast<int>(host_mode) << ':' << static_cast<int>(policy) << ':'
            << static_cast<int>(backing) << ':' << host_expert_cache_bytes << ':'
            << static_cast<int>(io_backend) << ':' << io_queue_depth << ':'
            << io_workers << ':' << direct_io << ':' << expert_sidecar_path << ':'
            << mirror_path << ':' << usage_profile_path;
        return out.str();
    }
};

std::size_t bytes_per_expert_bf16(const CompiledModelProgram& program);

int moe_layer_count(const CompiledModelProgram& program);

std::size_t kv_cache_bytes(const CompiledModelProgram& program, int context_tokens);

struct ExpertOffloadPlan {
    bool enabled = false;

    int experts_per_layer = 0;
    int host_experts_per_layer = 0;
    int moe_layers = 0;

    std::size_t gpu_free_bytes = 0;
    std::size_t non_expert_weight_bytes = 0;
    std::size_t kv_reservation_bytes = 0;
    std::size_t workspace_bytes = 0;
    std::size_t reserve_bytes = 0;

    std::size_t bytes_per_expert = 0;
    std::size_t gpu_expert_cache_bytes = 0;
    std::size_t host_expert_bytes = 0;

    int prefetch_experts = 0;

    ExpertHostMode host_mode = ExpertHostMode::Mapped;
    ExpertCachePolicy policy = ExpertCachePolicy::LayerLocalLfuLru;

    std::string report() const;
};

struct ExpertOffloadPlanInputs {
    explicit ExpertOffloadPlanInputs(const CompiledModelProgram& program_value)
        : program(program_value) {}
    CompiledModelProgram program;
    ExpertOffloadOptions options;
    std::size_t gpu_free_bytes = 0;
    std::size_t non_expert_weight_bytes = 0;
    std::size_t workspace_bytes = 0;
    int context_tokens = 0;
    int extra_moe_layers = 0;
    std::size_t extra_kv_reservation_bytes = 0;
};

ExpertOffloadPlan plan_expert_offload(const ExpertOffloadPlanInputs& inputs);

}
