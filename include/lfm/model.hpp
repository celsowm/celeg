#pragma once

#include "lfm/runtime_types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lfm {

struct PackedDecodeExecutorImpl;
class PhysicalPagedKvCache;
class LfmModel;
class IPackedSession;

// Narrow interface for token-processing operations. New C++ callers should
// depend on this view instead of the compatibility surface on LfmModel.
class LfmInferenceSession {
public:
    void reset(bool allocate_local_kv = true);
    void prefill(const std::vector<int32_t>& tokens);
    void prefill_chunk(const std::vector<int32_t>& tokens,
                       bool begin, bool finalize);
    void prefill_chunk_paged(const std::vector<int32_t>& tokens,
                             bool begin, bool finalize,
                             PhysicalPagedKvCache& paged_kv,
                             const std::vector<uint32_t>& page_table);
    int32_t decode();
    void decode_async_begin();
    int32_t decode_async_finish();
    void set_generation_config(GenerationConfig generation);
    void release_local_kv_cache();

    bool local_kv_cache_available() const;
    SessionPhase phase() const;
    bool ready_for_decode() const;
    bool decode_pending() const;
    int position() const;

private:
    friend class LfmModel;
    explicit LfmInferenceSession(LfmModel& owner) : owner_(&owner) {}
    LfmModel* owner_;
};

// Read-only operational and benchmarking surface.
class LfmDiagnostics {
public:
    std::vector<float> copy_logits() const;
    DecodeBenchmark benchmark_decode(int warmup_steps, int measured_steps);
    ModelMemoryStats memory_stats() const;
    RuntimeMetrics runtime_metrics() const;
    void clear_runtime_metrics();
    bool cuda_graph_ready() const;

    // MoE expert-offload residency stats (aggregate across all MoE layers):
    // GPU-resident experts per layer, host-resident experts per layer, and the
    // decode-time cache hit rate (hits / (hits + misses)).
    struct ExpertOffloadStats {
        int experts_per_layer = 0;
        int host_experts_per_layer = 0;
        uint64_t hits = 0;
        uint64_t misses = 0;
        double hit_rate = 0.0;  // in [0,1]; -1 if offload disabled
    };
    ExpertOffloadStats expert_offload_stats() const;

private:
    friend class LfmModel;
    explicit LfmDiagnostics(LfmModel& owner) : owner_(&owner) {}
    LfmModel* owner_;
};

// Persistence and deterministic prefix-state boundary.
class LfmPersistence {
public:
    void save_session(const std::string& path) const;
    void load_session(const std::string& path);
    PrefixState export_prefix_state() const;
    void restore_prefix_state(const PrefixState& state);

private:
    friend class LfmModel;
    explicit LfmPersistence(LfmModel& owner) : owner_(&owner) {}
    LfmModel* owner_;
};

// Thin compatibility facade. The implementation, CUDA resources and model
// topology live in Impl; focused clients can use session(), diagnostics() and
// persistence() to avoid depending on the complete legacy surface.
class LfmModel {
    friend class LfmInferenceSession;
    friend class LfmDiagnostics;
    friend class LfmPersistence;
public:
    LfmModel(const std::string& safetensors_path,
             int max_context = 4096,
             ModelOptions options = {},
             GenerationConfig generation = {});
    ~LfmModel();

    LfmModel(const LfmModel&) = delete;
    LfmModel& operator=(const LfmModel&) = delete;
    LfmModel(LfmModel&&) = delete;
    LfmModel& operator=(LfmModel&&) = delete;

    LfmInferenceSession& session() { return session_view_; }
    const LfmInferenceSession& session() const { return session_view_; }
    LfmDiagnostics& diagnostics() { return diagnostics_view_; }
    const LfmDiagnostics& diagnostics() const { return diagnostics_view_; }
    LfmPersistence& persistence() { return persistence_view_; }
    const LfmPersistence& persistence() const { return persistence_view_; }

    // Returns the narrow IPackedSession view. PackedDecodeExecutor depends
    // only on this interface rather than on LfmModel's full surface
    // (Interface Segregation Principle); the friend declaration for
    // PackedDecodeExecutorImpl is no longer required.
    IPackedSession& packed_session();
    const IPackedSession& packed_session() const;

    // Compatibility forwarding API retained for existing C++ and C adapters.
    void reset(bool allocate_local_kv = true);
    void prefill(const std::vector<int32_t>& tokens);
    void prefill_chunk(const std::vector<int32_t>& tokens, bool begin, bool finalize);
    void prefill_chunk_paged(const std::vector<int32_t>& tokens, bool begin,
                             bool finalize, PhysicalPagedKvCache& paged_kv,
                             const std::vector<uint32_t>& page_table);
    int32_t decode();
    void decode_async_begin();
    int32_t decode_async_finish();
    void set_generation_config(GenerationConfig generation);
    std::vector<float> copy_logits();
    DecodeBenchmark benchmark_decode(int warmup_steps, int measured_steps);
    ModelMemoryStats memory_stats() const;
    LfmDiagnostics::ExpertOffloadStats expert_offload_stats() const;
    RuntimeMetrics runtime_metrics() const;
    void clear_runtime_metrics();
    void save_session(const std::string& path);
    void load_session(const std::string& path);
    PrefixState export_prefix_state() const;
    void restore_prefix_state(const PrefixState& state);
    void release_local_kv_cache();
    bool local_kv_cache_available() const;
    SessionPhase phase() const;
    bool ready_for_decode() const;
    bool decode_pending() const;
    int position() const;
    int vocab_size() const;
    bool cuda_graph_ready() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    LfmInferenceSession session_view_;
    LfmDiagnostics diagnostics_view_;
    LfmPersistence persistence_view_;
};

} // namespace lfm
