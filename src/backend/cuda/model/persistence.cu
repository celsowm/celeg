#include "celeg/backend/cuda/session_store.hpp"
#include "celeg/detail/model/impl.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace celeg {

SessionStore::SessionState Model::Impl::make_session_state() {
    SessionStore::SessionState state{
        .shape = shape_, .max_context = max_context_, .position = position_,
        .kv_cache_mode = options_.kv_cache_mode, .variant = variant_,
        .stream = stream_.get(), .seen_tokens = &seen_tokens_,
        .logits = &logits_, .rng_state = &rng_state_};
    state.layer_buffers.reserve(layers_.size());
    for (Layer& layer : layers_) {
        SessionStore::SessionState::LayerBuffers buffers{};
        if (AttentionLayer* attention = as_attention(layer)) {
            buffers.is_attention = true;
            buffers.key_cache_bf16 = attention->key_cache.data();
            buffers.value_cache_bf16 = attention->value_cache.data();
            buffers.key_cache_int8 = attention->key_cache_int8.data();
            buffers.value_cache_int8 = attention->value_cache_int8.data();
            buffers.key_cache_scales = attention->key_cache_scales.data();
            buffers.value_cache_scales = attention->value_cache_scales.data();
        } else if (ConvolutionLayer* convolution = as_convolution(layer)) {
            buffers.conv_state = convolution->conv_state.data();
            buffers.conv_state_elements = convolution->conv_state.size();
        }
        state.layer_buffers.push_back(buffers);
    }
    return state;
}

void Model::Impl::save_session(const std::string& path) {
    if (!local_kv_cache_available_)
        throw std::runtime_error("save_session requires a model with a local contiguous KV cache");
    if (phase_ != SessionPhase::Ready)
        throw std::runtime_error("cannot save a session before prefill or load_session");
    auto state = make_session_state();
    SessionStore::save(path, state);
}

void Model::Impl::load_session(const std::string& path) {
    reset();
    auto state = make_session_state();
    SessionStore::load(path, state);
    position_ = state.position;
    CELEG_CUDA(cudaMemcpy(position_device_.data(), &position_, sizeof(position_), cudaMemcpyHostToDevice));
    phase_ = SessionPhase::Ready;
    active_segmented_attention_ = use_segmented_attention(position_);
    metrics_ = {};
}

PrefixState Model::Impl::export_prefix_state() const {
    if (phase_ != SessionPhase::Ready)
        throw std::runtime_error("cannot export prefix state before prefill");
    CELEG_CUDA(cudaStreamSynchronize(stream_.get()));
    auto state = const_cast<Impl*>(this)->make_session_state();
    auto snapshot = SessionStore::export_prefix(state);
    PrefixState out;
    out.position = snapshot.position;
    out.seen_tokens = std::move(snapshot.seen_tokens);
    out.logits_bf16 = std::move(snapshot.logits_bf16);
    out.conv_state_bf16 = std::move(snapshot.conv_state_bf16);
    return out;
}

void Model::Impl::restore_prefix_state(const PrefixState& state) {
    SessionStore::PrefixSnapshot snapshot;
    snapshot.position = state.position;
    snapshot.seen_tokens = state.seen_tokens;
    snapshot.logits_bf16 = state.logits_bf16;
    snapshot.conv_state_bf16 = state.conv_state_bf16;
    auto session = make_session_state();
    SessionStore::restore_prefix(snapshot, session, generation_.seed);
    phase_ = SessionPhase::Prefilling;
    position_ = session.position;
    CELEG_CUDA(cudaMemcpyAsync(position_device_.data(), &position_, sizeof(position_),
                             cudaMemcpyHostToDevice, stream_.get()));
    phase_ = SessionPhase::Ready;
    active_segmented_attention_ = use_segmented_attention(position_);
    metrics_ = {};
}

void SessionPersistence::save_session(const std::string& path) const {
    owner_->impl_->save_session(path);
}
void SessionPersistence::load_session(const std::string& path) {
    owner_->impl_->load_session(path);
}
PrefixState SessionPersistence::export_prefix_state() const {
    return owner_->impl_->export_prefix_state();
}
void SessionPersistence::restore_prefix_state(const PrefixState& state) {
    owner_->impl_->restore_prefix_state(state);
}

namespace {

template <typename T>
void write_scalar(std::ofstream& out, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("failed writing session");
}

template <typename T>
void read_scalar(std::ifstream& in, T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("truncated session file");
}

void write_device(std::ofstream& out, const void* device, size_t bytes,
                  cudaStream_t stream) {
    if (bytes == 0) return;
    std::vector<std::byte> host(bytes);
    CELEG_CUDA(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
    out.write(reinterpret_cast<const char*>(host.data()),
              static_cast<std::streamsize>(bytes));
    if (!out) throw std::runtime_error("failed writing session payload");
}

void read_device(std::ifstream& in, void* device, size_t bytes) {
    if (bytes == 0) return;
    std::vector<std::byte> host(bytes);
    in.read(reinterpret_cast<char*>(host.data()),
            static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("truncated session payload");
    CELEG_CUDA(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice));
}

} // namespace

void SessionStore::save(const std::string& path, SessionState& state) {
    if (state.stream != nullptr) {
        CELEG_CUDA(cudaStreamSynchronize(state.stream));
    }
    Header header;
    header.kv_cache_mode =
        state.kv_cache_mode == KvCacheMode::Int8 ? 1U : 0U;
    header.position = state.position;
    header.max_context = state.max_context;
    header.layers = state.shape.num_hidden_layers;
    header.kv_width = state.shape.kv_width;
    header.kv_heads = state.shape.num_key_value_heads;
    header.vocab = state.shape.vocab_size;
    header.attention_layers = state.shape.attention_layer_count;
    if (state.rng_state != nullptr) {
        CELEG_CUDA(cudaMemcpy(&header.rng_state, state.rng_state->data(),
                            sizeof(header.rng_state), cudaMemcpyDeviceToHost));
    }
    if (state.variant != nullptr) {
        const std::string_view id = state.variant->id();
        const size_t copy_size =
            std::min(id.size(), sizeof(header.variant_id) - 1);
        std::memcpy(header.variant_id, id.data(), copy_size);
        header.variant_id[copy_size] = '\0';
    }

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create session file: " + path);
    write_scalar(out, header);

    if (state.seen_tokens != nullptr) {
        write_device(out, state.seen_tokens->data(), state.seen_tokens->bytes(),
                     state.stream);
    }
    if (state.logits != nullptr) {
        write_device(out, state.logits->data(), state.logits->bytes(),
                     state.stream);
    }
    const size_t cache_elements =
        static_cast<size_t>(state.position) * state.shape.kv_width;
    const size_t scale_elements =
        static_cast<size_t>(state.position) * state.shape.num_key_value_heads;
    for (const auto& layer : state.layer_buffers) {
        if (layer.is_attention) {
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
        }
    }
}

void SessionStore::load(const std::string& path, SessionState& state) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open session file: " + path);
    Header header;
    read_scalar(in, header);
    if (header.magic != kMagic || header.version != kVersion) {
        throw std::runtime_error(
            "unsupported session format; this build requires session v2 files");
    }
    const uint32_t expected_kv =
        state.kv_cache_mode == KvCacheMode::Int8 ? 1U : 0U;
    if (header.kv_cache_mode != expected_kv) {
        throw std::runtime_error("session KV cache mode differs from model options");
    }
    if (header.position <= 0 || header.position > state.max_context ||
        header.layers != state.shape.num_hidden_layers ||
        header.kv_width != state.shape.kv_width ||
        header.kv_heads != state.shape.num_key_value_heads ||
        header.vocab != state.shape.vocab_size ||
        header.attention_layers != state.shape.attention_layer_count) {
        throw std::runtime_error("session dimensions are incompatible with this model");
    }
    if (state.variant != nullptr) {
        const std::string_view id = state.variant->id();
        const std::string_view stored(
            header.variant_id,
            strnlen(header.variant_id, sizeof(header.variant_id)));
        if (!stored.empty() && id != stored) {
            throw std::runtime_error(
                "session was written for a different model variant: " +
                std::string(stored));
        }
    }

    if (state.seen_tokens != nullptr) {
        read_device(in, state.seen_tokens->data(), state.seen_tokens->bytes());
    }
    if (state.logits != nullptr) {
        read_device(in, state.logits->data(), state.logits->bytes());
    }
    const size_t cache_elements =
        static_cast<size_t>(header.position) * state.shape.kv_width;
    const size_t scale_elements =
        static_cast<size_t>(header.position) * state.shape.num_key_value_heads;
    for (const auto& layer : state.layer_buffers) {
        if (layer.is_attention) {
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
        }
    }
    char trailing = 0;
    if (in.read(&trailing, 1)) {
        throw std::runtime_error("session file has trailing data");
    }
    if (!in.eof()) throw std::runtime_error("failed reading session file");

    state.position = header.position;
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
        if (!layer.is_attention) {
            conv_elements += layer.conv_state_elements;
        }
    }
    snapshot.conv_state_bf16.resize(conv_elements);
    size_t offset = 0;
    for (const auto& layer : state.layer_buffers) {
        if (layer.is_attention) continue;
        const size_t count = layer.conv_state_elements;
        if (count == 0) continue;
        CELEG_CUDA(cudaMemcpy(snapshot.conv_state_bf16.data() + offset,
                            layer.conv_state,
                            count * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToHost));
        offset += count;
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
    for (const auto& layer : state.layer_buffers) {
        if (!layer.is_attention) {
            expected_conv += layer.conv_state_elements;
        }
    }
    if (snapshot.conv_state_bf16.size() != expected_conv) {
        throw std::invalid_argument("prefix state convolution dimensions differ");
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
    // Prefix computation is deterministic, but generation randomness belongs
    // to the receiving request. Never inherit the seed/RNG stream of the
    // request that populated the shared prefix cache.
    if (state.stream != nullptr && state.rng_state != nullptr) {
        CELEG_CUDA(cudaMemcpyAsync(state.rng_state->data(), &request_seed,
                                 sizeof(request_seed), cudaMemcpyHostToDevice,
                                 state.stream));
    }
    size_t offset = 0;
    for (const auto& layer : state.layer_buffers) {
        if (layer.is_attention) continue;
        const size_t count = layer.conv_state_elements;
        if (count == 0) continue;
        CELEG_CUDA(cudaMemcpyAsync(layer.conv_state,
                                 snapshot.conv_state_bf16.data() + offset,
                                 count * sizeof(__nv_bfloat16),
                                 cudaMemcpyHostToDevice, state.stream));
        offset += count;
    }
    if (state.stream != nullptr) {
        CELEG_CUDA(cudaStreamSynchronize(state.stream));
    }
}

} // namespace celeg

