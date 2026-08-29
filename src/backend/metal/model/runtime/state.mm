#include "detail.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace celeg {

void MetalModel::Impl::reset() {
    position = 0;
    ready = false;
    metrics = {};
    std::fill(seen.begin(), seen.end(), 0);
    std::memset(hidden.contents, 0, model.graph.hidden * sizeof(float));
    std::memset(residual.contents, 0, model.graph.hidden * sizeof(float));
    for (Layer& layer : layers) {
        if (layer.convolution) {
            std::memset(layer.key_cache.contents, 0,
                        static_cast<size_t>(layer.cache_length) *
                            model.graph.hidden * sizeof(float));
        } else if (layer.gated_delta || layer.mamba2) {
            const size_t conv_elements = layer.gated_delta
                ? static_cast<size_t>(2 * layer.recurrent_key_heads *
                    layer.recurrent_key_head_dim +
                    layer.recurrent_value_heads * layer.recurrent_value_head_dim) *
                    layer.recurrent_conv_kernel
                : static_cast<size_t>(layer.recurrent_inner +
                    2 * layer.recurrent_group_count * layer.recurrent_state_size) *
                    layer.recurrent_conv_kernel;
            const size_t recurrent_elements = layer.gated_delta
                ? static_cast<size_t>(layer.recurrent_value_heads) *
                    layer.recurrent_key_head_dim * layer.recurrent_value_head_dim
                : static_cast<size_t>(layer.recurrent_inner) *
                    layer.recurrent_state_size;
            std::memset(layer.recurrent_conv_state.contents, 0,
                        conv_elements * sizeof(float));
            std::memset(layer.recurrent_state.contents, 0,
                        recurrent_elements * sizeof(float));
        } else {
            const size_t pages =
                (static_cast<size_t>(max_context) +
                 static_cast<size_t>(layer.page_tokens) - 1) /
                static_cast<size_t>(layer.page_tokens);
            const size_t elements = pages *
                static_cast<size_t>(layer.page_tokens) *
                static_cast<size_t>(layer.key_value_heads) *
                static_cast<size_t>(layer.head_dim);
            std::memset(layer.key_cache.contents, 0, elements * sizeof(float));
            std::memset(layer.value_cache.contents, 0, elements * sizeof(float));
        }
    }
}


std::vector<float> MetalModel::Impl::copy_buffer(id<MTLBuffer> source, size_t elements) {
    std::vector<float> result(elements);
    if (elements != 0) {
        std::memcpy(result.data(), source.contents, elements * sizeof(float));
    }
    return result;
}

MetalSessionSnapshot MetalModel::Impl::export_snapshot() const {
    MetalSessionSnapshot snapshot;
    snapshot.position = position;
    snapshot.ready = ready;
    snapshot.generation = generation;
    snapshot.rng_state = rng_state;
    snapshot.metrics = metrics;
    snapshot.seen = seen;
    snapshot.hidden = copy_buffer(hidden, static_cast<size_t>(model.graph.hidden));
    snapshot.residual = copy_buffer(residual, static_cast<size_t>(model.graph.hidden));
    snapshot.logits = copy_buffer(logits,
                                  static_cast<size_t>(model.topology.dims.vocab_size));
    snapshot.key_state.reserve(layers.size());
    snapshot.value_state.reserve(layers.size());
    snapshot.mixer_state.reserve(layers.size());
    snapshot.recurrent_state.reserve(layers.size());
    for (const Layer& layer : layers) {
        if (layer.convolution) {
            snapshot.key_state.push_back(copy_buffer(
                layer.key_cache, static_cast<size_t>(layer.cache_length) *
                    static_cast<size_t>(model.graph.hidden)));
            snapshot.value_state.emplace_back();
        } else if (!layer.gated_delta && !layer.mamba2) {
            const size_t pages =
                (static_cast<size_t>(max_context) +
                 static_cast<size_t>(layer.page_tokens) - 1) /
                static_cast<size_t>(layer.page_tokens);
            const size_t elements = pages *
                static_cast<size_t>(layer.page_tokens) *
                static_cast<size_t>(layer.key_value_heads) *
                static_cast<size_t>(layer.head_dim);
            snapshot.key_state.push_back(copy_buffer(layer.key_cache, elements));
            snapshot.value_state.push_back(copy_buffer(layer.value_cache, elements));
        } else {
            snapshot.key_state.emplace_back();
            snapshot.value_state.emplace_back();
        }
        if (layer.gated_delta || layer.mamba2) {
            const size_t conv_elements = layer.gated_delta
                ? static_cast<size_t>(2 * layer.recurrent_key_heads *
                    layer.recurrent_key_head_dim +
                    layer.recurrent_value_heads * layer.recurrent_value_head_dim) *
                    layer.recurrent_conv_kernel
                : static_cast<size_t>(layer.recurrent_inner +
                    2 * layer.recurrent_group_count * layer.recurrent_state_size) *
                    layer.recurrent_conv_kernel;
            const size_t recurrent_elements = layer.gated_delta
                ? static_cast<size_t>(layer.recurrent_value_heads) *
                    layer.recurrent_key_head_dim * layer.recurrent_value_head_dim
                : static_cast<size_t>(layer.recurrent_inner) *
                    layer.recurrent_state_size;
            snapshot.mixer_state.push_back(copy_buffer(
                layer.recurrent_conv_state, conv_elements));
            snapshot.recurrent_state.push_back(copy_buffer(
                layer.recurrent_state, recurrent_elements));
        } else {
            snapshot.mixer_state.emplace_back();
            snapshot.recurrent_state.emplace_back();
        }
    }
    return snapshot;
}

void MetalModel::Impl::restore_snapshot(MetalSessionSnapshot snapshot) {
    snapshot.generation.validate();
    if (snapshot.position < 0 || snapshot.position > max_context ||
        snapshot.seen.size() != static_cast<size_t>(model.topology.dims.vocab_size) ||
        snapshot.hidden.size() != static_cast<size_t>(model.graph.hidden) ||
        snapshot.residual.size() != static_cast<size_t>(model.graph.hidden) ||
        snapshot.logits.size() != static_cast<size_t>(model.topology.dims.vocab_size) ||
        snapshot.key_state.size() != layers.size() ||
        snapshot.value_state.size() != layers.size() ||
        snapshot.mixer_state.size() != layers.size() ||
        snapshot.recurrent_state.size() != layers.size()) {
        throw std::invalid_argument("invalid Metal session snapshot dimensions");
    }
    position = snapshot.position;
    ready = snapshot.ready;
    generation = std::move(snapshot.generation);
    rng_state = snapshot.rng_state;
    metrics = snapshot.metrics;
    seen = std::move(snapshot.seen);
    std::memcpy(hidden.contents, snapshot.hidden.data(),
                snapshot.hidden.size() * sizeof(float));
    std::memcpy(residual.contents, snapshot.residual.data(),
                snapshot.residual.size() * sizeof(float));
    std::memcpy(logits.contents, snapshot.logits.data(),
                snapshot.logits.size() * sizeof(float));
    for (size_t index = 0; index < layers.size(); ++index) {
        const Layer& layer = layers[index];
        if (layer.convolution) {
            const size_t elements = static_cast<size_t>(layer.cache_length) *
                static_cast<size_t>(model.graph.hidden);
            if (snapshot.key_state[index].size() != elements ||
                !snapshot.value_state[index].empty() ||
                !snapshot.mixer_state[index].empty() ||
                !snapshot.recurrent_state[index].empty()) {
                throw std::invalid_argument("invalid Metal convolution snapshot state");
            }
            std::memcpy(layer.key_cache.contents, snapshot.key_state[index].data(),
                        elements * sizeof(float));
        } else if (layer.gated_delta || layer.mamba2) {
            const size_t conv_elements = layer.gated_delta
                ? static_cast<size_t>(2 * layer.recurrent_key_heads *
                    layer.recurrent_key_head_dim +
                    layer.recurrent_value_heads * layer.recurrent_value_head_dim) *
                    layer.recurrent_conv_kernel
                : static_cast<size_t>(layer.recurrent_inner +
                    2 * layer.recurrent_group_count * layer.recurrent_state_size) *
                    layer.recurrent_conv_kernel;
            const size_t recurrent_elements = layer.gated_delta
                ? static_cast<size_t>(layer.recurrent_value_heads) *
                    layer.recurrent_key_head_dim * layer.recurrent_value_head_dim
                : static_cast<size_t>(layer.recurrent_inner) *
                    layer.recurrent_state_size;
            if (!snapshot.key_state[index].empty() ||
                !snapshot.value_state[index].empty() ||
                snapshot.mixer_state[index].size() != conv_elements ||
                snapshot.recurrent_state[index].size() != recurrent_elements) {
                throw std::invalid_argument("invalid Metal recurrent snapshot state");
            }
            std::memcpy(layer.recurrent_conv_state.contents,
                        snapshot.mixer_state[index].data(),
                        conv_elements * sizeof(float));
            std::memcpy(layer.recurrent_state.contents,
                        snapshot.recurrent_state[index].data(),
                        recurrent_elements * sizeof(float));
        } else {
            const size_t elements = ((static_cast<size_t>(max_context) +
                static_cast<size_t>(layer.page_tokens) - 1) /
                static_cast<size_t>(layer.page_tokens)) *
                static_cast<size_t>(layer.page_tokens) *
                static_cast<size_t>(layer.key_value_heads) *
                static_cast<size_t>(layer.head_dim);
            if (snapshot.key_state[index].size() != elements ||
                snapshot.value_state[index].size() != elements ||
                !snapshot.mixer_state[index].empty() ||
                !snapshot.recurrent_state[index].empty()) {
                throw std::invalid_argument("invalid Metal attention snapshot state");
            }
            std::memcpy(layer.key_cache.contents, snapshot.key_state[index].data(),
                        elements * sizeof(float));
            std::memcpy(layer.value_cache.contents, snapshot.value_state[index].data(),
                        elements * sizeof(float));
        }
    }
}

}

