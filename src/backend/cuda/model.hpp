#pragma once

#include "celeg/model/runtime_types.hpp"
#include "runtime_types.hpp"
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

    struct ExpertOffloadStats {
        int experts_per_layer = 0;
        int host_experts_per_layer = 0;
        uint64_t hits = 0;
        uint64_t misses = 0;
        double hit_rate = 0.0;
    };
    ExpertOffloadStats expert_offload_stats() const;

private:
    friend class CudaModel;
    explicit CudaModelDiagnostics(CudaModel& owner) : owner_(&owner) {}
    CudaModel* owner_;
};

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
              std::shared_ptr<const RuntimeContext> runtime = nullptr,
              int tokenizer_vocab_size = 0);
    ~CudaModel();

    CudaModel(const CudaModel&) = delete;
    CudaModel& operator=(const CudaModel&) = delete;
    CudaModel(CudaModel&&) = delete;
    CudaModel& operator=(CudaModel&&) = delete;

    CudaInferenceSession session() { return CudaInferenceSession(*this); }
    CudaInferenceSession session() const { return CudaInferenceSession(const_cast<CudaModel&>(*this)); }
    CudaModelDiagnostics diagnostics() { return CudaModelDiagnostics(*this); }
    CudaModelDiagnostics diagnostics() const { return CudaModelDiagnostics(const_cast<CudaModel&>(*this)); }
    SessionPersistence persistence() { return SessionPersistence(*this); }
    SessionPersistence persistence() const { return SessionPersistence(const_cast<CudaModel&>(*this)); }

private:
    std::unique_ptr<CudaCompiledModel> state_;
};

}
