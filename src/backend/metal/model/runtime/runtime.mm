#include "detail.hpp"

#include "celeg/runtime/sampler.hpp"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace celeg {

void MetalModel::Impl::apply_logits_transforms() {
    float* values = static_cast<float*>(logits.contents);
    const size_t count = static_cast<size_t>(model.topology.dims.vocab_size);
    if (model.graph.logits_multiplier != 1.0f) {
        for (size_t index = 0; index < count; ++index) values[index] *= model.graph.logits_multiplier;
    }
    if (model.graph.logits_divisor != 1.0f) {
        for (size_t index = 0; index < count; ++index) values[index] /= model.graph.logits_divisor;
    }
    if (model.graph.final_logit_softcap > 0.0f) {
        for (size_t index = 0; index < count; ++index) {
            values[index] = std::tanh(values[index] / model.graph.final_logit_softcap) *
                model.graph.final_logit_softcap;
        }
    }
}

MetalModel::~MetalModel() = default;
MetalModel::MetalModel(MetalModel&&) noexcept = default;
MetalModel& MetalModel::operator=(MetalModel&&) noexcept = default;

void MetalModel::reset_session() {
    (*impl_).reset();
    (*impl_).next_rope_position = {0, 0, 0};
}

void MetalModel::prefill_session(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) throw std::invalid_argument("Metal prefill needs at least one token");
    if (tokens.size() > static_cast<size_t>((*impl_).max_context)) {
        throw std::invalid_argument("Metal prefill exceeds context");
    }
    (*impl_).reset();
    (*impl_).next_rope_position = {0, 0, 0};
    (*impl_).execution_metrics.command_encoding_ms = 0.0;
    (*impl_).execution_metrics.command_wait_ms = 0.0;
    (*impl_).execution_metrics.gpu_execution_ms = 0.0;
    (*impl_).execution_metrics.command_buffers = 0;
    (*impl_).execution_metrics.dispatches = 0;
    const auto started = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    (*impl_).begin_commands(command_buffer, encoder);
    for (const int32_t token : tokens) {
        if (token < 0 || token >= (*impl_).model.topology.dims.vocab_size) {
            throw std::invalid_argument("Metal token out of range");
        }
        (*impl_).seen[static_cast<size_t>(token)] = 1;
    }
    if ((*impl_).supports_prefill_batch()) {
        (*impl_).encode_prefill_batch(encoder, tokens);
    } else {
        for (const int32_t token : tokens) {
            (*impl_).encode_token(command_buffer, encoder, token);
            for (int32_t& value : (*impl_).next_rope_position) ++value;
        }
    }
    (*impl_).finish_commands(command_buffer, encoder);
    (*impl_).apply_logits_transforms();
    (*impl_).metrics.last_prefill_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
    (*impl_).metrics.prefill_tokens = tokens.size();
    (*impl_).ready = true;
}

void MetalModel::prefill_session(const std::vector<int32_t>& tokens,
                                 const PromptEmbedding& embeddings) {
    if (!embeddings.has_rope_positions) {
        prefill_session(tokens);
        return;
    }
    if (tokens.empty()) throw std::invalid_argument("Metal prefill needs at least one token");
    if (tokens.size() > static_cast<size_t>((*impl_).max_context)) {
        throw std::invalid_argument("Metal prefill exceeds context");
    }
    if (embeddings.rope_positions.size() != tokens.size()) {
        throw std::invalid_argument("Metal prompt M-RoPE positions must cover every token");
    }
    if (!embeddings.empty() || !embeddings.values.empty()) {
        throw std::invalid_argument(
            "Metal M-RoPE prefill currently accepts position metadata without raw prompt embeddings");
    }
    (*impl_).reset();
    (*impl_).next_rope_position = {0, 0, 0};
    (*impl_).execution_metrics.command_encoding_ms = 0.0;
    (*impl_).execution_metrics.command_wait_ms = 0.0;
    (*impl_).execution_metrics.gpu_execution_ms = 0.0;
    (*impl_).execution_metrics.command_buffers = 0;
    (*impl_).execution_metrics.dispatches = 0;
    const auto started = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    (*impl_).begin_commands(command_buffer, encoder);
    for (const int32_t token : tokens) {
        if (token < 0 || token >= (*impl_).model.topology.dims.vocab_size) {
            throw std::invalid_argument("Metal token out of range");
        }
        (*impl_).seen[static_cast<size_t>(token)] = 1;
    }
    if ((*impl_).supports_prefill_batch()) {
        (*impl_).encode_prefill_batch(encoder, tokens, embeddings.rope_positions);
    } else {
        for (size_t index = 0; index < tokens.size(); ++index) {
            (*impl_).encode_token(command_buffer, encoder, tokens[index],
                                  &embeddings.rope_positions[index]);
        }
    }
    (*impl_).next_rope_position = embeddings.next_rope_position;
    (*impl_).finish_commands(command_buffer, encoder);
    (*impl_).apply_logits_transforms();
    (*impl_).metrics.last_prefill_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
    (*impl_).metrics.prefill_tokens = tokens.size();
    (*impl_).ready = true;
}

int32_t MetalModel::decode_session() {
    if (!(*impl_).ready) throw std::runtime_error("Metal model is not ready for decode");
    const auto started = std::chrono::steady_clock::now();
    std::vector<float> values = session_logits();
    const int32_t token = Sampler::sample(values, vocab_size(), (*impl_).generation,
                                             (*impl_).seen, (*impl_).rng_state);
    (*impl_).seen[static_cast<size_t>(token)] = 1;
    (*impl_).run_token(token);
    (*impl_).metrics.cumulative_decode_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    ++(*impl_).metrics.decoded_tokens;
    return token;
}

void MetalModel::eval_token_session(int32_t token) {
    if (!(*impl_).ready) throw std::runtime_error("Metal model is not ready for token evaluation");
    if (token < 0 || token >= vocab_size()) throw std::invalid_argument("Metal token out of range");
    (*impl_).seen[static_cast<size_t>(token)] = 1;
    (*impl_).run_token(token);
    ++(*impl_).metrics.decoded_tokens;
}

void MetalModel::eval_token_session(int32_t token,
                                    const std::array<int32_t, 3>& rope_position) {
    if (!(*impl_).ready) throw std::runtime_error("Metal model is not ready for token evaluation");
    if (token < 0 || token >= vocab_size()) throw std::invalid_argument("Metal token out of range");
    (*impl_).seen[static_cast<size_t>(token)] = 1;
    (*impl_).run_token(token, &rope_position);
    ++(*impl_).metrics.decoded_tokens;
}

void MetalModel::set_session_generation(GenerationConfig generation) {
    generation.validate();
    (*impl_).generation = std::move(generation);
    (*impl_).rng_state = (*impl_).generation.seed;
}

std::vector<float> MetalModel::session_logits() const {
    std::vector<float> values(static_cast<size_t>(vocab_size()));
    std::memcpy(values.data(), (*impl_).logits.contents,
                values.size() * sizeof(float));
    return values;
}

int MetalModel::vocab_size() const { return (*impl_).model.topology.dims.vocab_size; }
const std::string& MetalModel::model_identity() const { return (*impl_).model.provenance.identity; }
std::string MetalModel::backend_description() const {
    return std::string("metal-native device=") + (*impl_).device.name.UTF8String +
        " tensor_f16=" + ((*impl_).pipeline_cache.tensor_matmul_f16 ? "yes" : "no") +
        " tensor_bf16=" + ((*impl_).pipeline_cache.tensor_matmul_bf16 ? "yes" : "no") +
        " tensor_q4_0=" + ((*impl_).pipeline_cache.tensor_matmul_q4_0 ? "yes" : "no") +
        " tensor_q4k=" + ((*impl_).pipeline_cache.tensor_matmul_q4k ? "yes" : "no") +
        " tensor_q5k=" + ((*impl_).pipeline_cache.tensor_matmul_q5k ? "yes" : "no") +
        " tensor_q6k=" + ((*impl_).pipeline_cache.tensor_matmul_q6k ? "yes" : "no") +
        " tensor_q8_0=" + ((*impl_).pipeline_cache.tensor_matmul_q8_0 ? "yes" : "no") +
        " tensor_pair_q5k=" + ((*impl_).pipeline_cache.tensor_matmul_pair_q5k ? "yes" : "no") +
        " tensor_pair_q6k=" + ((*impl_).pipeline_cache.tensor_matmul_pair_q6k ? "yes" : "no") +
        ((*impl_).pipeline_cache.tensor_compile_error.empty()
            ? std::string{} : " tensor_error=" + (*impl_).pipeline_cache.tensor_compile_error);
}
RuntimeMetrics MetalModel::metrics() const { return (*impl_).metrics; }
MetalExecutionMetrics MetalModel::execution_metrics() const { return (*impl_).execution_metrics; }
MetalSessionSnapshot MetalModel::export_session_snapshot() const {
    MetalSessionSnapshot snapshot = (*impl_).export_snapshot();
    snapshot.next_rope_position = (*impl_).next_rope_position;
    return snapshot;
}
void MetalModel::restore_session_snapshot(MetalSessionSnapshot snapshot) {
    const auto next_rope_position = snapshot.next_rope_position;
    (*impl_).restore_snapshot(std::move(snapshot));
    (*impl_).next_rope_position = next_rope_position;
}

int MetalModel::session_position() const { return (*impl_).position; }
bool MetalModel::session_ready_for_decode() const { return (*impl_).ready; }

void MetalInferenceSession::reset() { owner_->reset_session(); }
void MetalInferenceSession::prefill(const std::vector<int32_t>& tokens) { owner_->prefill_session(tokens); }
void MetalInferenceSession::prefill(const std::vector<int32_t>& tokens,
                                    const PromptEmbedding& embeddings) {
    owner_->prefill_session(tokens, embeddings);
}
int32_t MetalInferenceSession::decode() { return owner_->decode_session(); }
void MetalInferenceSession::eval_token(int32_t token) { owner_->eval_token_session(token); }
void MetalInferenceSession::eval_token(
    int32_t token, const std::array<int32_t, 3>& rope_position) {
    owner_->eval_token_session(token, rope_position);
}
void MetalInferenceSession::set_generation_config(GenerationConfig generation) {
    owner_->set_session_generation(std::move(generation));
}
std::vector<float> MetalInferenceSession::copy_logits() const { return owner_->session_logits(); }
int MetalInferenceSession::position() const { return owner_->session_position(); }
bool MetalInferenceSession::ready_for_decode() const { return owner_->session_ready_for_decode(); }

}
