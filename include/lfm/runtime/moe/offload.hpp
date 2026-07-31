#pragma once

#include "lfm/model/config/shape.hpp"
#include "lfm/checkpoint/formats/safetensors.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <sstream>
#include <vector>

namespace lfm {

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

// ---------------------------------------------------------------------------
// MoE expert offload configuration and memory planning.
//
// The LFM2.5-8B-A1B checkpoint stores >90% of its weights in sparse MoE
// experts (704 experts for the reference topology). The natural offload unit
// is therefore a single expert, not a whole transformer layer: non-expert
// weights (attention, convolution, dense FFNs, router, norms, embeddings) run
// for every token and stay GPU-resident, while cold experts live in host RAM
// and are streamed / mapped on demand.
//
// This header defines the configuration surface and the *pure* memory planner.
// The planner takes GPU/host byte budgets as inputs (the caller supplies
// cudaMemGetInfo results) so it can be unit-tested without a GPU.
// ---------------------------------------------------------------------------

// Where a given expert currently lives.
enum class ExpertResidence : uint8_t {
    // Resident in a GPU expert-cache slot; run with device pointers.
    DeviceCache,
    // Backed by pinned/mapped host memory; run via the host-mapped GEMV
    // fallback or promoted into a cache slot before use.
    HostMapped,
};

// Selects how offloaded experts are made accessible from the device.
enum class ExpertHostMode : uint8_t {
    // cudaHostRegister page-aligned ranges of the mmap'd safetensors file and
    // obtain device-visible pointers through CUDA UVA. Preferred: no copy into
    // a separate arena.
    Mapped,
    // cudaHostAlloc a pinned arena and copy offloaded experts into it. Fallback
    // for platforms where mapped registration is unavailable/unreliable.
    PinnedCopy,
    // Stage a whole expert layer into a reusable device arena for prefill,
    // decode uses the cache. Used only during layer-major prefill.
    Staged,
};

// Eviction / promotion policy for the per-layer GPU expert cache.
enum class ExpertCachePolicy : uint8_t {
    // Fixed set chosen once (e.g. from the prefill routing profile).
    Static,
    // Least-recently-used.
    Lru,
    // Combined least-frequently-used + least-recently-used score.
    LayerLocalLfuLru,
    // Router-likelihood-driven: evict the resident with the lowest last-known
    // routing score (sentinel -1e30 for seeded/unseen experts). The resident set
    // converges to the highest-likelihood experts -- but only beats LRU when
    // router scores have strong cross-token correlation, which LFM2-8B-A1B on
    // RTX 3060 does NOT (measured 3.67 tok/s vs LRU 5.10 tok/s).
    Score,
};

// Top-level toggle for the offload feature.
enum class ExpertOffloadMode : uint8_t {
    // No offload: experts are loaded eagerly onto the GPU (current behaviour).
    None,
    // Compute a plan from available memory and enable offload automatically.
    Auto,
    // Force host-backed experts even when they would fit on the GPU.
    Host,
};

// Byte-size helpers (mirror the packed expert layout in expert_loader.cpp).
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
    // free VRAM).
    std::size_t gpu_expert_cache_bytes = 0;

    // Upper bound on pinned host memory used for offloaded experts.
    std::size_t maximum_pinned_host_bytes = 9_GiB;

    // Explicit resident experts per MoE layer; <= 0 means "auto".
    int experts_per_layer = 0;

    // Chunk size (in tokens) for layer-major streamed prefill.
    int prefill_chunk_tokens = 256;

    // Never keep fewer than this many experts per layer on the GPU.
    int minimum_experts_per_layer = 4;

    // During decode, after promoting the experts a token actually routes to,
    // speculatively prefetch this many additional not-yet-resident experts
    // (ranked by router likelihood) so they are GPU-resident before the next
    // token's FFN. Hides PCIe promotion latency behind the current token's
    // compute. Default 0: prefetch is gated by score-aware eviction, and
    // experiments on LFM2-8B-A1B (RTX 3060 12GB) show on-demand promotion is
    // already fast enough that any speculative promotion displaces a
    // genuinely-useful resident, regressing decode tok/s; raise only if working
    // set has clear predictive locality beyond the routed top-K.
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

// Per-expert BF16 byte size for the given topology:
//   gate_up : 2 * moe_intermediate * hidden
//   down    :     hidden * moe_intermediate
// times sizeof(__nv_bfloat16) == 2.
std::size_t bytes_per_expert_bf16(const ModelShape& shape);

// Number of MoE layers in the topology (layers at or beyond num_dense_layers).
int moe_layer_count(const ModelShape& shape);

// BF16 KV-cache bytes required for `context_tokens` tokens across all
// attention layers.
std::size_t kv_cache_bytes(const ModelShape& shape, int context_tokens);

// The resolved offload layout. All byte fields are exact multiples of the
// per-expert / per-layer sizes so the caller can allocate directly.
struct ExpertOffloadPlan {
    bool enabled = false;

    int experts_per_layer = 0;      // GPU-resident experts per MoE layer
    int host_experts_per_layer = 0; // remaining experts (in host RAM)
    int moe_layers = 0;

    std::size_t gpu_free_bytes = 0;
    std::size_t non_expert_weight_bytes = 0;
    std::size_t kv_reservation_bytes = 0;
    std::size_t workspace_bytes = 0;
    std::size_t reserve_bytes = 0;

    std::size_t bytes_per_expert = 0;
    std::size_t gpu_expert_cache_bytes = 0;
    std::size_t host_expert_bytes = 0;

    int prefetch_experts = 0;       // speculative prefetch depth per decode step

    ExpertHostMode host_mode = ExpertHostMode::Mapped;
    ExpertCachePolicy policy = ExpertCachePolicy::LayerLocalLfuLru;

    // Human-readable multi-line "MoE offload plan:" report.
    std::string report() const;
};

// Inputs to the planner. `gpu_free_bytes` / `non_expert_weight_bytes` /
// `workspace_bytes` are supplied by the caller (from cudaMemGetInfo and the
// weight arena) so the planner itself is pure and testable.
struct ExpertOffloadPlanInputs {
    ModelShape shape;
    ExpertOffloadOptions options;
    std::size_t gpu_free_bytes = 0;
    std::size_t non_expert_weight_bytes = 0;
    std::size_t workspace_bytes = 0;
    int context_tokens = 0;
};

// Computes the offload layout. When offload is disabled the returned plan has
// enabled == false and experts_per_layer == shape.num_experts (all resident).
// Throws std::runtime_error if even minimum_experts_per_layer cannot fit.
ExpertOffloadPlan plan_expert_offload(const ExpertOffloadPlanInputs& inputs);

} // namespace lfm
