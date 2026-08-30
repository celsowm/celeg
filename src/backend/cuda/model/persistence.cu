#include "backend/cuda/session_store.hpp"
#include "detail/compiled_model.hpp"

#include <fstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>

namespace celeg {

namespace {

uint64_t fnv1a_hash(std::string_view text) {
    uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char byte : text) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

}  // namespace

SessionStore::SessionState CudaCompiledModel::make_session_state() {
    SessionStore::SessionState state{
        .shape = resources_.shape_, .program = resources_.program_,
        .dims = resources_.dims_,
        .max_context = max_context_, .position = session_.position_,
        .kv_cache_mode = resources_.options_.kv_cache_mode, .model_identity = resources_.model_identity_,
        .stream = stream_.get(), .seen_tokens = &sampling_.seen_tokens,
        .logits = &workspace_.logits_, .rng_state = &sampling_.rng_state};
    state.layer_buffers.reserve(resources_.layers_.size());
    for (Layer& layer : resources_.layers_) {
        SessionStore::SessionState::LayerBuffers buffers{};
        visit_layer(layer,
          [&](AttentionLayer* attention) {
            buffers.is_attention = true;
            buffers.owns_kv_cache = attention->key_cache_bf16() != nullptr ||
                attention->key_cache_int8_ptr() != nullptr;
            buffers.key_cache_bf16 = attention->key_cache_bf16();
            buffers.value_cache_bf16 = attention->value_cache_bf16();
            buffers.key_cache_int8 = attention->key_cache_int8_ptr();
            buffers.value_cache_int8 = attention->value_cache_int8_ptr();
            buffers.key_cache_scales = attention->key_cache_scales_ptr();
            buffers.value_cache_scales = attention->value_cache_scales_ptr();
          },
          [&](ConvolutionLayer* convolution) {
            buffers.conv_state = convolution->conv_state.data();
            buffers.conv_state_elements = convolution->conv_state.size();
          },
          [&](Mamba2Layer* mamba) {
            buffers.is_mamba = true;
            buffers.conv_state = mamba->conv_state.data();
            buffers.conv_state_elements = mamba->conv_state.size();
            buffers.ssm_state = mamba->ssm_state.data();
            buffers.ssm_state_elements = mamba->ssm_state.size();
          },
          [&](GatedDeltaNetLayer* gated_delta) {
            buffers.is_gated_delta = true;
            buffers.conv_state = gated_delta->conv_state.data();
            buffers.conv_state_elements = gated_delta->conv_state.size();
            buffers.recurrent_state = gated_delta->recurrent_state.data();
            buffers.recurrent_state_elements = gated_delta->recurrent_state.size();
          },
          [](MlpOnlyLayer*) {});
        state.layer_buffers.push_back(buffers);
    }
    return state;
}

void CudaCompiledModel::save_session(const std::string& path) {
    if (!local_kv_cache_available_)
        throw std::runtime_error("save_session requires a model with a local contiguous KV cache");
    if (session_.phase_ != SessionPhase::Ready)
        throw std::runtime_error("cannot save a session before prefill or load_session");
    auto state = make_session_state();
    SessionStore::save(path, state);
}

void CudaCompiledModel::load_session(const std::string& path) {
    reset();
    auto state = make_session_state();
    SessionStore::load(path, state);
    session_.position_ = state.position;
    CELEG_CUDA(cudaMemcpy(position_device_.data(), &session_.position_, sizeof(session_.position_), cudaMemcpyHostToDevice));
    session_.phase_ = SessionPhase::Ready;
    session_.active_segmented_attention_ = use_segmented_attention(session_.position_);
    session_.metrics_ = {};
}

PrefixState CudaCompiledModel::export_prefix_state() const {
    if (session_.phase_ != SessionPhase::Ready)
        throw std::runtime_error("cannot export prefix state before prefill");
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
    auto state = const_cast<CudaCompiledModel*>(this)->make_session_state();
    auto snapshot = SessionStore::export_prefix(state);
    PrefixState out;
    out.position = snapshot.position;
    out.seen_tokens = std::move(snapshot.seen_tokens);
    out.logits_bf16 = std::move(snapshot.logits_bf16);
    out.conv_state_bf16 = std::move(snapshot.conv_state_bf16);
    out.mamba_state_bf16 = std::move(snapshot.mamba_state_bf16);
    out.gated_delta_state_bf16 = std::move(snapshot.gated_delta_state_bf16);
    return out;
}

void CudaCompiledModel::restore_prefix_state(const PrefixState& state) {
    SessionStore::PrefixSnapshot snapshot;
    snapshot.position = state.position;
    snapshot.seen_tokens = state.seen_tokens;
    snapshot.logits_bf16 = state.logits_bf16;
    snapshot.conv_state_bf16 = state.conv_state_bf16;
    snapshot.mamba_state_bf16 = state.mamba_state_bf16;
    snapshot.gated_delta_state_bf16 = state.gated_delta_state_bf16;
    auto session = make_session_state();
    SessionStore::restore_prefix(snapshot, session, session_.generation_.seed);
    session_.phase_ = SessionPhase::Prefilling;
    session_.position_ = session.position;
    CELEG_CUDA(cudaMemcpyAsync(position_device_.data(), &session_.position_, sizeof(session_.position_),
                             cudaMemcpyHostToDevice, stream_.get()));
    session_.phase_ = SessionPhase::Ready;
    session_.active_segmented_attention_ = use_segmented_attention(session_.position_);
    session_.metrics_ = {};
}

void SessionPersistence::save_session(const std::string& path) const {
    owner_->state_->save_session(path);
}
void SessionPersistence::load_session(const std::string& path) {
    owner_->state_->load_session(path);
}
PrefixState SessionPersistence::export_prefix_state() const {
    return owner_->state_->export_prefix_state();
}
void SessionPersistence::restore_prefix_state(const PrefixState& state) {
    owner_->state_->restore_prefix_state(state);
}

namespace {

template <typename T>
void write_scalar(std::ostream& out, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("failed writing session");
}

template <typename T>
void read_scalar(std::istream& in, T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("truncated session file");
}

void write_device(std::ostream& out, const void* device, size_t bytes,
                  cudaStream_t stream) {
    if (bytes == 0) return;
    std::vector<std::byte> host(bytes);
    CELEG_CUDA(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
    out.write(reinterpret_cast<const char*>(host.data()),
              static_cast<std::streamsize>(bytes));
    if (!out) throw std::runtime_error("failed writing session payload");
}

void read_device(std::istream& in, void* device, size_t bytes) {
    if (bytes == 0) return;
    std::vector<std::byte> host(bytes);
    in.read(reinterpret_cast<char*>(host.data()),
            static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("truncated session payload");
    CELEG_CUDA(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice));
}

}

void SessionStore::encode(std::ostream& out, SessionState& state) {
    if (state.stream != nullptr) {
        CELEG_CUDA(cudaStreamSynchronize(state.stream));
    }
    Header header;
    header.kv_cache_mode =
        state.kv_cache_mode == KvCacheMode::Int8 ? 1U : 0U;
    header.position = state.position;
    header.max_context = state.max_context;
    header.layers = state.shape.num_hidden_layers;
    header.vocab = state.dims.vocab_size;
    header.attention_layers = state.shape.attention_layer_count;
    if (state.rng_state != nullptr) {
        CELEG_CUDA(cudaMemcpy(&header.rng_state, state.rng_state->data(),
                            sizeof(header.rng_state), cudaMemcpyDeviceToHost));
    }
    if (!state.model_identity.empty()) {
        header.model_identity_hash = fnv1a_hash(state.model_identity);
    }

    write_scalar(out, header);

    if (state.seen_tokens != nullptr) {
        write_device(out, state.seen_tokens->data(), state.seen_tokens->bytes(),
                     state.stream);
    }
    if (state.logits != nullptr) {
        write_device(out, state.logits->data(), state.logits->bytes(),
                     state.stream);
    }
    for (size_t layer_index = 0; layer_index < state.layer_buffers.size(); ++layer_index) {
        const auto& layer = state.layer_buffers[layer_index];
        if (layer.is_attention && layer.owns_kv_cache) {
            const AttentionSpec& layout = std::get<CompiledAttentionProgram>(
                state.program.layers.at(layer_index).mixer).semantics;
            const size_t cache_elements = static_cast<size_t>(state.position) *
                static_cast<size_t>(layout.key_value_width());
            const size_t scale_elements = static_cast<size_t>(state.position) *
                static_cast<size_t>(layout.key_value_heads);
            if (state.kv_cache_mode == KvCacheMode::Int8) {
                write_device(out, layer.key_cache_int8,
                             cache_elements * sizeof(int8_t), state.stream);
                write_device(out, layer.value_cache_int8,
                             cache_elements * sizeof(int8_t), state.stream);
                write_device(out, layer.key_cache_scales,
                             scale_elements * sizeof(float), state.stream);
                write_device(out, layer.value_cache_scales,
                             scale_elements * sizeof(float), state.stream);
            } else {
                write_device(out, layer.key_cache_bf16,
                             cache_elements * sizeof(__nv_bfloat16),
                             state.stream);
                write_device(out, layer.value_cache_bf16,
                             cache_elements * sizeof(__nv_bfloat16),
                             state.stream);
            }
        } else {
            write_device(out, layer.conv_state,
                         layer.conv_state_elements * sizeof(__nv_bfloat16),
                         state.stream);
            if (layer.is_mamba) {
                write_device(out, layer.ssm_state,
                             layer.ssm_state_elements * sizeof(__nv_bfloat16),
                             state.stream);
            } else if (layer.is_gated_delta) {
                write_device(out, layer.recurrent_state,
                             layer.recurrent_state_elements * sizeof(__nv_bfloat16),
                             state.stream);
            }
        }
    }
}

void SessionStore::save(const std::string& path, SessionState& state) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create session file: " + path);
    encode(out, state);
}

void SessionStore::decode(std::istream& in, SessionState& state) {
    Header header;
    read_scalar(in, header);
    if (header.magic != kMagic || header.version != kVersion) {
        throw std::runtime_error(
            "unsupported session format; this build requires session v5 files");
    }
    const uint32_t expected_kv =
        state.kv_cache_mode == KvCacheMode::Int8 ? 1U : 0U;
    if (header.kv_cache_mode != expected_kv) {
        throw std::runtime_error("session KV cache mode differs from model options");
    }
    if (header.position <= 0 || header.position > state.max_context ||
        header.layers != state.shape.num_hidden_layers ||
        header.vocab != state.dims.vocab_size ||
        header.attention_layers != state.shape.attention_layer_count) {
        throw std::runtime_error("session dimensions are incompatible with this model");
    }
    if (!state.model_identity.empty() && header.model_identity_hash != 0) {
        if (fnv1a_hash(state.model_identity) != header.model_identity_hash) {
            throw std::runtime_error(
                "session was written for a different resolved model");
        }
    }

    if (state.seen_tokens != nullptr) {
        read_device(in, state.seen_tokens->data(), state.seen_tokens->bytes());
    }
    if (state.logits != nullptr) {
        read_device(in, state.logits->data(), state.logits->bytes());
    }
    for (size_t layer_index = 0; layer_index < state.layer_buffers.size(); ++layer_index) {
        const auto& layer = state.layer_buffers[layer_index];
        if (layer.is_attention && layer.owns_kv_cache) {
            const AttentionSpec& layout = std::get<CompiledAttentionProgram>(
                state.program.layers.at(layer_index).mixer).semantics;
            const size_t cache_elements = static_cast<size_t>(header.position) *
                static_cast<size_t>(layout.key_value_width());
            const size_t scale_elements = static_cast<size_t>(header.position) *
                static_cast<size_t>(layout.key_value_heads);
            if (state.kv_cache_mode == KvCacheMode::Int8) {
                read_device(in, layer.key_cache_int8,
                            cache_elements * sizeof(int8_t));
                read_device(in, layer.value_cache_int8,
                            cache_elements * sizeof(int8_t));
                read_device(in, layer.key_cache_scales,
                            scale_elements * sizeof(float));
                read_device(in, layer.value_cache_scales,
                            scale_elements * sizeof(float));
            } else {
                read_device(in, layer.key_cache_bf16,
                            cache_elements * sizeof(__nv_bfloat16));
                read_device(in, layer.value_cache_bf16,
                            cache_elements * sizeof(__nv_bfloat16));
            }
        } else {
            read_device(in, layer.conv_state,
                        layer.conv_state_elements * sizeof(__nv_bfloat16));
            if (layer.is_mamba) {
                read_device(in, layer.ssm_state,
                            layer.ssm_state_elements * sizeof(__nv_bfloat16));
            } else if (layer.is_gated_delta) {
                read_device(in, layer.recurrent_state,
                            layer.recurrent_state_elements * sizeof(__nv_bfloat16));
            }
        }
    }
    char trailing = 0;
    if (in.read(&trailing, 1)) {
        throw std::runtime_error("session file has trailing data");
    }
    if (!in.eof()) throw std::runtime_error("failed reading session file");

    state.position = header.position;
}

void SessionStore::load(const std::string& path, SessionState& state) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open session file: " + path);
    decode(in, state);
}

SessionStore::PrefixSnapshot SessionStore::export_prefix(const SessionState& state) {
    PrefixSnapshot snapshot;
    snapshot.position = state.position;
    if (state.seen_tokens != nullptr) {
        snapshot.seen_tokens.resize(state.seen_tokens->size());
        CELEG_CUDA(cudaMemcpy(snapshot.seen_tokens.data(),
                            state.seen_tokens->data(),
                            state.seen_tokens->bytes(),
                            cudaMemcpyDeviceToHost));
    }
    if (state.logits != nullptr) {
        snapshot.logits_bf16.resize(state.logits->size());
        CELEG_CUDA(cudaMemcpy(snapshot.logits_bf16.data(),
                            state.logits->data(), state.logits->bytes(),
                            cudaMemcpyDeviceToHost));
    }
    size_t conv_elements = 0;
    for (const auto& layer : state.layer_buffers) {
        if (!layer.is_attention && !layer.is_mamba && !layer.is_gated_delta) {
            conv_elements += layer.conv_state_elements;
        }
        if (layer.is_mamba) {
            snapshot.mamba_state_bf16.resize(snapshot.mamba_state_bf16.size() +
                layer.conv_state_elements + layer.ssm_state_elements);
        }
        if (layer.is_gated_delta) {
            snapshot.gated_delta_state_bf16.resize(snapshot.gated_delta_state_bf16.size() +
                layer.conv_state_elements + layer.recurrent_state_elements);
        }
    }
    snapshot.conv_state_bf16.resize(conv_elements);
    size_t offset = 0;
    for (const auto& layer : state.layer_buffers) {
        if (layer.is_attention || layer.is_mamba || layer.is_gated_delta) continue;
        const size_t count = layer.conv_state_elements;
        if (count == 0) continue;
        CELEG_CUDA(cudaMemcpy(snapshot.conv_state_bf16.data() + offset,
                            layer.conv_state,
                            count * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToHost));
        offset += count;
    }
    size_t mamba_offset = 0;
    for (const auto& layer : state.layer_buffers) {
        if (!layer.is_mamba) continue;
        const size_t conv_count = layer.conv_state_elements;
        const size_t ssm_count = layer.ssm_state_elements;
        CELEG_CUDA(cudaMemcpy(snapshot.mamba_state_bf16.data() + mamba_offset,
                              layer.conv_state, conv_count * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        mamba_offset += conv_count;
        CELEG_CUDA(cudaMemcpy(snapshot.mamba_state_bf16.data() + mamba_offset,
                              layer.ssm_state, ssm_count * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        mamba_offset += ssm_count;
    }
    size_t gated_delta_offset = 0;
    for (const auto& layer : state.layer_buffers) {
        if (!layer.is_gated_delta) continue;
        const size_t conv_count = layer.conv_state_elements;
        const size_t recurrent_count = layer.recurrent_state_elements;
        CELEG_CUDA(cudaMemcpy(snapshot.gated_delta_state_bf16.data() + gated_delta_offset,
                              layer.conv_state, conv_count * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        gated_delta_offset += conv_count;
        CELEG_CUDA(cudaMemcpy(snapshot.gated_delta_state_bf16.data() + gated_delta_offset,
                              layer.recurrent_state, recurrent_count * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));
        gated_delta_offset += recurrent_count;
    }
    return snapshot;
}

void SessionStore::restore_prefix(const PrefixSnapshot& snapshot,
                                  SessionState& state,
                                  uint64_t request_seed) {
    if (snapshot.position <= 0 || snapshot.position > state.max_context) {
        throw std::invalid_argument("prefix state position is invalid");
    }
    if (state.seen_tokens != nullptr &&
        snapshot.seen_tokens.size() != state.seen_tokens->size()) {
        throw std::invalid_argument("prefix state sampling dimensions differ");
    }
    if (state.logits != nullptr &&
        snapshot.logits_bf16.size() != state.logits->size()) {
        throw std::invalid_argument("prefix state sampling dimensions differ");
    }
    size_t expected_conv = 0;
    size_t expected_mamba = 0;
    size_t expected_gated_delta = 0;
    for (const auto& layer : state.layer_buffers) {
        if (!layer.is_attention && !layer.is_mamba && !layer.is_gated_delta) {
            expected_conv += layer.conv_state_elements;
        }
        if (layer.is_mamba) expected_mamba += layer.conv_state_elements + layer.ssm_state_elements;
        if (layer.is_gated_delta) {
            expected_gated_delta += layer.conv_state_elements + layer.recurrent_state_elements;
        }
    }
    if (snapshot.conv_state_bf16.size() != expected_conv) {
        throw std::invalid_argument("prefix state convolution dimensions differ");
    }
    if (snapshot.mamba_state_bf16.size() != expected_mamba) {
        throw std::invalid_argument("prefix state Mamba dimensions differ");
    }
    if (snapshot.gated_delta_state_bf16.size() != expected_gated_delta) {
        throw std::invalid_argument("prefix state GatedDeltaNet dimensions differ");
    }

    state.position = snapshot.position;
    if (state.stream != nullptr && state.seen_tokens != nullptr) {
        CELEG_CUDA(cudaMemcpyAsync(state.seen_tokens->data(),
                                 snapshot.seen_tokens.data(),
                                 state.seen_tokens->bytes(),
                                 cudaMemcpyHostToDevice, state.stream));
    }
    if (state.stream != nullptr && state.logits != nullptr) {
        CELEG_CUDA(cudaMemcpyAsync(state.logits->data(),
                                 snapshot.logits_bf16.data(),
                                 state.logits->bytes(),
                                 cudaMemcpyHostToDevice, state.stream));
    }
    if (state.stream != nullptr && state.rng_state != nullptr) {
        CELEG_CUDA(cudaMemcpyAsync(state.rng_state->data(), &request_seed,
                                 sizeof(request_seed), cudaMemcpyHostToDevice,
                                 state.stream));
    }
    size_t offset = 0;
    for (const auto& layer : state.layer_buffers) {
        if (layer.is_attention) continue;
        if (layer.is_mamba) continue;
        if (layer.is_gated_delta) continue;
        const size_t count = layer.conv_state_elements;
        if (count == 0) continue;
        CELEG_CUDA(cudaMemcpyAsync(layer.conv_state,
                                 snapshot.conv_state_bf16.data() + offset,
                                 count * sizeof(__nv_bfloat16),
                                 cudaMemcpyHostToDevice, state.stream));
        offset += count;
    }
    size_t mamba_offset = 0;
    for (const auto& layer : state.layer_buffers) {
        if (!layer.is_mamba) continue;
        const size_t conv_count = layer.conv_state_elements;
        const size_t ssm_count = layer.ssm_state_elements;
        CELEG_CUDA(cudaMemcpyAsync(layer.conv_state,
                                   snapshot.mamba_state_bf16.data() + mamba_offset,
                                   conv_count * sizeof(__nv_bfloat16),
                                   cudaMemcpyHostToDevice, state.stream));
        mamba_offset += conv_count;
        CELEG_CUDA(cudaMemcpyAsync(layer.ssm_state,
                                   snapshot.mamba_state_bf16.data() + mamba_offset,
                                   ssm_count * sizeof(__nv_bfloat16),
                                   cudaMemcpyHostToDevice, state.stream));
        mamba_offset += ssm_count;
    }
    size_t gated_delta_offset = 0;
    for (const auto& layer : state.layer_buffers) {
        if (!layer.is_gated_delta) continue;
        const size_t conv_count = layer.conv_state_elements;
        const size_t recurrent_count = layer.recurrent_state_elements;
        CELEG_CUDA(cudaMemcpyAsync(layer.conv_state,
                                 snapshot.gated_delta_state_bf16.data() + gated_delta_offset,
                                 conv_count * sizeof(__nv_bfloat16),
                                 cudaMemcpyHostToDevice, state.stream));
        gated_delta_offset += conv_count;
        CELEG_CUDA(cudaMemcpyAsync(layer.recurrent_state,
                                 snapshot.gated_delta_state_bf16.data() + gated_delta_offset,
                                 recurrent_count * sizeof(__nv_bfloat16),
                                 cudaMemcpyHostToDevice, state.stream));
        gated_delta_offset += recurrent_count;
    }
    if (state.stream != nullptr) {
        CELEG_CUDA(cudaStreamSynchronize(state.stream));
    }
}

}
