#pragma once

#include "celeg/backend/metal/runtime_types.hpp"
#include "celeg/model/runtime_types.hpp"
#include "celeg/runtime/context.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace celeg {

class MetalModel;

struct MetalSessionSnapshot {
    int position = 0;
    bool ready = false;
    GenerationConfig generation;
    uint64_t rng_state = 1;
    RuntimeMetrics metrics;
    std::vector<uint8_t> seen;
    std::vector<float> hidden;
    std::vector<float> residual;
    std::vector<float> logits;
    std::vector<std::vector<float>> key_state;
    std::vector<std::vector<float>> value_state;
    std::vector<std::vector<float>> mixer_state;
    std::vector<std::vector<float>> recurrent_state;
};

class MetalInferenceSession {
public:
    void reset();
    void prefill(const std::vector<int32_t>& tokens);
    int32_t decode();
    void eval_token(int32_t token);
    void set_generation_config(GenerationConfig generation);
    std::vector<float> copy_logits() const;
    int position() const;
    bool ready_for_decode() const;

private:
    friend class MetalModel;
    explicit MetalInferenceSession(MetalModel& owner) : owner_(&owner) {}
    MetalModel* owner_;
};

class MetalModel {
public:
    MetalModel(const std::string& model_path,
               int max_context = 4096,
               MetalModelOptions options = {},
               GenerationConfig generation = {},
               std::shared_ptr<const RuntimeContext> runtime = nullptr);
    ~MetalModel();

    MetalModel(const MetalModel&) = delete;
    MetalModel& operator=(const MetalModel&) = delete;
    MetalModel(MetalModel&&) noexcept;
    MetalModel& operator=(MetalModel&&) noexcept;

    MetalInferenceSession session() { return MetalInferenceSession(*this); }

    int vocab_size() const;
    const std::string& model_identity() const;
    std::string backend_description() const;
    RuntimeMetrics metrics() const;
    MetalExecutionMetrics execution_metrics() const;
    MetalSessionSnapshot export_session_snapshot() const;
    void restore_session_snapshot(MetalSessionSnapshot snapshot);

private:
    friend class MetalInferenceSession;
    void reset_session();
    void prefill_session(const std::vector<int32_t>& tokens);
    int32_t decode_session();
    void eval_token_session(int32_t token);
    void set_session_generation(GenerationConfig generation);
    std::vector<float> session_logits() const;
    int session_position() const;
    bool session_ready_for_decode() const;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
