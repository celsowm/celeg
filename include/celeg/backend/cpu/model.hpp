#pragma once

#include "celeg/backend/cpu/isa.hpp"
#include "celeg/backend/cpu/numa.hpp"
#include "celeg/backend/cpu/runtime_types.hpp"
#include "celeg/backend/cpu/topology.hpp"
#include "celeg/model/runtime_types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace celeg {

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

class CpuConcurrentEngine;
class CpuKvPagePool;
struct CpuPrefixSnapshot;
class CpuModel;
struct CpuCompiledModel;

struct CpuPrefillItem {
    CpuModel* session = nullptr;
    int32_t token = 0;
    bool final_token = false;
};

struct CpuPrefillProfile {
    double linear_ms = 0.0;
    double attention_ms = 0.0;
    double shortconv_ms = 0.0;
    double moe_router_ms = 0.0;
    double moe_expert_ms = 0.0;
    double total_ms = 0.0;
};

struct CpuBatchMetrics {
    size_t batch_size = 0;
    double elapsed_ms = 0.0;
};

class CpuInferenceSession {
public:
    void reset();
    void prefill(const std::vector<int32_t>& tokens);
    int32_t decode();
    // Advances the session with a caller-supplied token without sampling.
    // Intended for direct-evaluation benchmarks and deterministic replay.
    void eval_token(int32_t token);
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
    CpuPrefillProfile prefill_profile() const;
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
public:
    CpuModel(const std::string& model_path,
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

    // Batch operations are the only packed-execution boundary. They keep
    // weights and mutable session state opaque to callers.
    static CpuBatchMetrics prefill_batch(std::span<const CpuPrefillItem> items);
    static CpuBatchMetrics prefill_chunk(CpuModel& session,
                                         std::span<const int32_t> tokens,
                                         bool final_chunk);
    static std::pair<std::vector<int32_t>, CpuBatchMetrics>
    decode_batch(std::span<CpuModel* const> sessions);

private:
    friend class CpuInferenceSession;
    friend class CpuDiagnostics;
    friend class CpuPersistence;

    void reset_session();
    void prefill_session(const std::vector<int32_t>& tokens);
    int32_t decode_session();
    void eval_token_session(int32_t token);
    void set_session_generation(GenerationConfig generation);
    int session_position() const;
    bool session_ready_for_decode() const;
    std::vector<float> session_logits() const;
    RuntimeMetrics session_metrics() const;
    CpuPrefillProfile session_prefill_profile() const;
    void clear_session_metrics();
    CpuModelMemoryStats session_memory_stats() const;
    int session_vocab_size() const;
    CpuIsa session_isa() const;
    CpuKvCacheMode session_kv_cache_mode() const;
    std::string session_backend_description() const;
    const std::filesystem::path& session_pack_path() const;
    bool session_loaded_from_pack() const;
    uint64_t session_attention_parallel_calls() const;
    CpuPrefixSnapshot export_session_prefix() const;
    void restore_session_prefix(CpuPrefixSnapshot snapshot, bool ready_for_decode);

    explicit CpuModel(std::unique_ptr<CpuCompiledModel> compiled_model);
    std::unique_ptr<CpuCompiledModel> state_;
    CpuInferenceSession session_view_;
    CpuDiagnostics diagnostics_view_;
    CpuPersistence persistence_view_;

};

const char* cpu_kv_cache_mode_name(CpuKvCacheMode mode);
CpuKvCacheMode parse_cpu_kv_cache_mode(const std::string& text);

} // namespace celeg
