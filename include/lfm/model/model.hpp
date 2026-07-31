#pragma once

#include "lfm/model/execution/runtime_types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace lfm {

struct PackedDecodeExecutorImpl;
class PhysicalPagedKvCache;
class Model;
struct PackedSessionContext;

// Narrow interface for token-processing operations. New C++ callers should
// depend on this view instead of the complete Model surface.
class InferenceSession {
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
    friend class Model;
    explicit InferenceSession(Model& owner) : owner_(&owner) {}
    Model* owner_;
};

// Read-only operational and benchmarking surface.
class ModelDiagnostics {
public:
    std::vector<float> copy_logits() const;
    DecodeBenchmark benchmark_decode(int warmup_steps, int measured_steps);
    ModelMemoryStats memory_stats() const;
    RuntimeMetrics runtime_metrics() const;
    void clear_runtime_metrics();
    bool cuda_graph_ready() const;
    int vocab_size() const;

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
    friend class Model;
    explicit ModelDiagnostics(Model& owner) : owner_(&owner) {}
    Model* owner_;
};

// Persistence and deterministic prefix-state boundary.
class SessionPersistence {
public:
    void save_session(const std::string& path) const;
    void load_session(const std::string& path);
    PrefixState export_prefix_state() const;
    void restore_prefix_state(const PrefixState& state);

private:
    friend class Model;
    explicit SessionPersistence(Model& owner) : owner_(&owner) {}
    Model* owner_;
};

// Thin runtime facade. The implementation, CUDA resources and model
// topology live in Impl; focused clients can use session(), diagnostics() and
// persistence() to avoid depending on the complete model implementation.
class Model {
    friend class InferenceSession;
    friend class ModelDiagnostics;
    friend class SessionPersistence;
    friend PackedSessionContext packed_session_context(Model& model);
public:
    Model(const std::string& model_path,
             int max_context = 4096,
             ModelOptions options = {},
             GenerationConfig generation = {});
    ~Model();

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = delete;
    Model& operator=(Model&&) = delete;

    InferenceSession& session() { return session_view_; }
    const InferenceSession& session() const { return session_view_; }
    ModelDiagnostics& diagnostics() { return diagnostics_view_; }
    const ModelDiagnostics& diagnostics() const { return diagnostics_view_; }
    SessionPersistence& persistence() { return persistence_view_; }
    const SessionPersistence& persistence() const { return persistence_view_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    InferenceSession session_view_;
    ModelDiagnostics diagnostics_view_;
    SessionPersistence persistence_view_;
};

} // namespace lfm
