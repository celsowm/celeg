#include "lfm/session_store.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <type_traits>

namespace lfm {

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
    LFM_CUDA(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
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
    LFM_CUDA(cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice));
}

} // namespace

void SessionStore::save(const std::string& path, SessionState& state) {
    if (state.stream != nullptr) {
        LFM_CUDA(cudaStreamSynchronize(state.stream));
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
        LFM_CUDA(cudaMemcpy(&header.rng_state, state.rng_state->data(),
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
        LFM_CUDA(cudaMemcpy(snapshot.seen_tokens.data(),
                            state.seen_tokens->data(),
                            state.seen_tokens->bytes(),
                            cudaMemcpyDeviceToHost));
    }
    if (state.logits != nullptr) {
        snapshot.logits_bf16.resize(state.logits->size());
        LFM_CUDA(cudaMemcpy(snapshot.logits_bf16.data(),
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
        LFM_CUDA(cudaMemcpy(snapshot.conv_state_bf16.data() + offset,
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
        LFM_CUDA(cudaMemcpyAsync(state.seen_tokens->data(),
                                 snapshot.seen_tokens.data(),
                                 state.seen_tokens->bytes(),
                                 cudaMemcpyHostToDevice, state.stream));
    }
    if (state.stream != nullptr && state.logits != nullptr) {
        LFM_CUDA(cudaMemcpyAsync(state.logits->data(),
                                 snapshot.logits_bf16.data(),
                                 state.logits->bytes(),
                                 cudaMemcpyHostToDevice, state.stream));
    }
    // Prefix computation is deterministic, but generation randomness belongs
    // to the receiving request. Never inherit the seed/RNG stream of the
    // request that populated the shared prefix cache.
    if (state.stream != nullptr && state.rng_state != nullptr) {
        LFM_CUDA(cudaMemcpyAsync(state.rng_state->data(), &request_seed,
                                 sizeof(request_seed), cudaMemcpyHostToDevice,
                                 state.stream));
    }
    size_t offset = 0;
    for (const auto& layer : state.layer_buffers) {
        if (layer.is_attention) continue;
        const size_t count = layer.conv_state_elements;
        if (count == 0) continue;
        LFM_CUDA(cudaMemcpyAsync(layer.conv_state,
                                 snapshot.conv_state_bf16.data() + offset,
                                 count * sizeof(__nv_bfloat16),
                                 cudaMemcpyHostToDevice, state.stream));
        offset += count;
    }
    if (state.stream != nullptr) {
        LFM_CUDA(cudaStreamSynchronize(state.stream));
    }
}

} // namespace lfm
