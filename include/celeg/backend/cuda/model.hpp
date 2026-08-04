#pragma once

#include "celeg/model/runtime_types.hpp"
#include "celeg/backend/cuda/runtime_types.hpp"
#include "celeg/runtime/context.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace celeg {

struct PackedDecodeExecutorImpl;
class PhysicalPagedKvCache;
class CudaModel;
struct CudaCompiledModel;
struct PackedSessionContext;

// Narrow interface for token-processing operations. New C++ callers should
// depend on this view instead of the complete CudaModel surface.
class CudaInferenceSession {
public:
    void reset(bool allocate_local_kv = true);
    void prefill(const std::vector<int32_t>& tokens);
    void prefill(const std::vector<int32_t>& tokens,
                 const PromptEmbedding& embeddings);
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
    friend class CudaModel;
    explicit CudaInferenceSession(CudaModel& owner) : owner_(&owner) {}
    CudaModel* owner_;
};

// Read-only operational and benchmarking surface.
class CudaModelDiagnostics {
public:
    std::vector<float> copy_logits() const;
    DecodeBenchmark benchmark_decode(int warmup_steps, int measured_steps);
    ModelMemoryStats memory_stats() const;
    RuntimeMetrics runtime_metrics() const;
    void clear_runtime_metrics();
    std::string execution_plan_description() const;
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
    friend class CudaModel;
    explicit CudaModelDiagnostics(CudaModel& owner) : owner_(&owner) {}
    CudaModel* owner_;
};

// Persistence and deterministic prefix-state boundary.
class SessionPersistence {
public:
    void save_session(const std::string& path) const;
    void load_session(const std::string& path);
    PrefixState export_prefix_state() const;
    void restore_prefix_state(const PrefixState& state);

private:
    friend class CudaModel;
    explicit SessionPersistence(CudaModel& owner) : owner_(&owner) {}
    CudaModel* owner_;
};

// Thin runtime facade. The implementation, CUDA resources and model
// topology live in CudaCompiledModel; focused clients can use session(), diagnostics() and
// persistence() to avoid depending on the complete model implementation.
class CudaModel {
    friend class CudaInferenceSession;
    friend class CudaModelDiagnostics;
    friend class SessionPersistence;
    friend PackedSessionContext packed_session_context(CudaModel& model);
public:
    CudaModel(const std::string& model_path,
             int max_context = 4096,
             CudaModelOptions options = {},
             GenerationConfig generation = {},
             std::shared_ptr<const RuntimeContext> runtime = nullptr);
    ~CudaModel();

    CudaModel(const CudaModel&) = delete;
    CudaModel& operator=(const CudaModel&) = delete;
    CudaModel(CudaModel&&) = delete;
    CudaModel& operator=(CudaModel&&) = delete;

    CudaInferenceSession& session() { return session_view_; }
    const CudaInferenceSession& session() const { return session_view_; }
    CudaModelDiagnostics& diagnostics() { return diagnostics_view_; }
    const CudaModelDiagnostics& diagnostics() const { return diagnostics_view_; }
    SessionPersistence& persistence() { return persistence_view_; }
    const SessionPersistence& persistence() const { return persistence_view_; }

private:
    std::unique_ptr<CudaCompiledModel> state_;
    CudaInferenceSession session_view_;
    CudaModelDiagnostics diagnostics_view_;
    SessionPersistence persistence_view_;
};

} // namespace celeg
