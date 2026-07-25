#pragma once

#include "lfm/cpu_isa.hpp"
#include "lfm/cpu_numa.hpp"
#include "lfm/cpu_runtime_types.hpp"
#include "lfm/cpu_topology.hpp"
#include "lfm/runtime_types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace lfm {

struct CpuModelOptions {
    CpuIsa isa = CpuIsa::Auto;
    CpuWeightFormat weight_format = CpuWeightFormat::Q4Group32;
    CpuKvCacheMode kv_cache_mode = CpuKvCacheMode::Bf16;
    size_t threads = 0;
    CpuAffinityPolicy affinity = CpuAffinityPolicy::None;
    bool use_pack_cache = true;
    std::filesystem::path pack_cache_directory;
    size_t kv_page_tokens = 32;
    size_t prefill_chunk_tokens = 256;
    size_t prefill_chunk_threshold = 16;
    size_t attention_parallel_threshold = 256;
    size_t attention_page_tile = 4;
    CpuNumaMode numa_mode = CpuNumaMode::Disabled;
};

struct CpuModelMemoryStats {
    size_t weights = 0;
    size_t kv_cache = 0;
    size_t conv_state = 0;
    size_t activations = 0;
    size_t kv_pages_used = 0;
    size_t kv_pages_total = 0;
    size_t total() const { return weights + kv_cache + conv_state + activations; }
};

class CpuPackedExecutor;
class CpuConcurrentEngine;
class CpuKvPagePool;
struct CpuPrefixSnapshot;
class CpuModel;

class CpuInferenceSession {
public:
    void reset();
    void prefill(const std::vector<int32_t>& tokens);
    int32_t decode();
    void set_generation_config(GenerationConfig generation);
    int position() const;
    bool ready_for_decode() const;

private:
    friend class CpuModel;
    explicit CpuInferenceSession(CpuModel& owner) : owner_(&owner) {}
    CpuModel* owner_;
};

class CpuDiagnostics {
public:
    std::vector<float> copy_logits() const;
    RuntimeMetrics runtime_metrics() const;
    void clear_runtime_metrics();
    CpuModelMemoryStats memory_stats() const;
    int vocab_size() const;
    CpuIsa isa() const;
    CpuKvCacheMode kv_cache_mode() const;
    std::string backend_description() const;
    const std::filesystem::path& pack_path() const;
    bool loaded_from_pack() const;
    uint64_t attention_parallel_calls() const;

private:
    friend class CpuModel;
    explicit CpuDiagnostics(CpuModel& owner) : owner_(&owner) {}
    CpuModel* owner_;
};

class CpuPersistence {
public:
    CpuPrefixSnapshot export_prefix_snapshot() const;
    void restore_prefix_snapshot(CpuPrefixSnapshot snapshot,
                                 bool ready_for_decode);

private:
    friend class CpuModel;
    explicit CpuPersistence(CpuModel& owner) : owner_(&owner) {}
    CpuModel* owner_;
};

class CpuModel {
    friend class CpuInferenceSession;
    friend class CpuDiagnostics;
    friend class CpuPersistence;
public:
    CpuModel(const std::string& safetensors_path,
             int max_context = 4096,
             CpuModelOptions options = {},
             GenerationConfig generation = {});
    ~CpuModel();

    CpuModel(const CpuModel&) = delete;
    CpuModel& operator=(const CpuModel&) = delete;
    CpuModel(CpuModel&&) noexcept;
    CpuModel& operator=(CpuModel&&) noexcept;

    CpuInferenceSession& session() { return session_view_; }
    const CpuInferenceSession& session() const { return session_view_; }
    CpuDiagnostics& diagnostics() { return diagnostics_view_; }
    const CpuDiagnostics& diagnostics() const { return diagnostics_view_; }
    CpuPersistence& persistence() { return persistence_view_; }
    const CpuPersistence& persistence() const { return persistence_view_; }

    // Creates a new mutable inference session that reuses the same immutable
    // packed weights and the same persistent CPU thread pool.
    std::unique_ptr<CpuModel> clone_session() const;
    std::unique_ptr<CpuModel> clone_session_on_node(int numa_node) const;

    std::vector<std::shared_ptr<CpuKvPagePool>> shared_kv_pools() const;

private:
    struct Impl;
    explicit CpuModel(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    CpuInferenceSession session_view_;
    CpuDiagnostics diagnostics_view_;
    CpuPersistence persistence_view_;

    friend class CpuPackedExecutor;
    friend class CpuConcurrentEngine;
};

const char* cpu_kv_cache_mode_name(CpuKvCacheMode mode);
CpuKvCacheMode parse_cpu_kv_cache_mode(const std::string& text);

} // namespace lfm
