#include "detail/model_internal.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace lfm {

CpuModel::Impl::Impl(std::shared_ptr<Shared> shared_weights,
                     GenerationConfig generation_config,
                     int requested_numa_node)
    : shared(std::move(shared_weights)), generation(generation_config),
      preferred_numa_node(requested_numa_node) {
    if (!shared) throw std::invalid_argument("shared CPU model weights are required");
    generation.validate();
    allocate_state();
    allocate_activations();
    reset();
}

CpuModel::Impl::~Impl() {
    for (LayerState& layer : states) {
        if (auto* attention = std::get_if<AttentionState>(&layer)) {
            release_attention_pages(*attention);
        }
    }
}

void CpuModel::Impl::allocate_state() {
    states.reserve(shared->layers.size());
    for (size_t index = 0; index < shared->layers.size(); ++index) {
        const WeightLayer& layer = shared->layers[index];
        if (std::holds_alternative<AttentionWeights>(layer)) {
            const int pool = shared->layer_to_kv_pool.at(index);
            if (pool < 0) throw std::logic_error("attention layer has no CPU KV page pool");
            AttentionState state;
            state.pool_index = static_cast<size_t>(pool);
            states.emplace_back(std::move(state));
        } else {
            ConvolutionState state;
            state.state.resize(static_cast<size_t>(shared->shape.conv_cache) *
                               shared->shape.hidden);
            states.emplace_back(std::move(state));
        }
    }
}

void CpuModel::Impl::allocate_activations() {
    ensure(1, shared->shape);
    logits.resize(shared->shape.vocab_size);
    seen.resize(shared->shape.vocab_size);
}

const CpuModel::Impl::CommonWeights& CpuModel::Impl::common_weights(
    size_t layer) const {
    return std::visit([](const auto& value) -> const CommonWeights& {
        return value.common;
    }, shared->layers.at(layer));
}

CpuModel::Impl::AttentionState& CpuModel::Impl::attention_state(size_t layer) {
    return std::get<AttentionState>(states.at(layer));
}

const CpuModel::Impl::AttentionState& CpuModel::Impl::attention_state(
    size_t layer) const {
    return std::get<AttentionState>(states.at(layer));
}

CpuModel::Impl::ConvolutionState& CpuModel::Impl::convolution_state(
    size_t layer) {
    return std::get<ConvolutionState>(states.at(layer));
}

const CpuModel::Impl::ConvolutionState& CpuModel::Impl::convolution_state(
    size_t layer) const {
    return std::get<ConvolutionState>(states.at(layer));
}

void CpuModel::Impl::release_attention_pages(AttentionState& state) noexcept {
    if (!shared || state.pool_index >= shared->kv_pools.size()) return;
    auto& pool = *shared->kv_pools[state.pool_index];
    for (CpuKvPageId page : state.pages) {
        try {
            pool.release(page);
        } catch (const std::exception& error) {
            std::clog << "CPU KV page cleanup failed: " << error.what() << '\n';
        } catch (...) {
            std::clog << "CPU KV page cleanup failed: unknown exception\n";
        }
    }
    state.pages.clear();
    state.token_count = 0;
}

void CpuModel::Impl::store_kv(AttentionState& state, int position,
                              const float* key, const float* value) {
    if (position < 0) throw std::invalid_argument("negative CPU KV position");
    CpuKvPagePool& pool = *shared->kv_pools.at(state.pool_index);
    const size_t position_value = static_cast<size_t>(position);
    const size_t page_index = position_value / pool.page_tokens();
    const size_t token_offset = position_value % pool.page_tokens();
    while (state.pages.size() <= page_index) {
        state.pages.push_back(pool.allocate(preferred_numa_node));
    }
    if (token_offset != 0 && pool.reference_count(state.pages[page_index]) > 1) {
        const CpuKvPageId shared_page = state.pages[page_index];
        const CpuKvPageId private_page = pool.clone_prefix(
            shared_page, token_offset, preferred_numa_node);
        state.pages[page_index] = private_page;
        pool.release(shared_page);
    }
    pool.write(state.pages[page_index], token_offset, key, value);
    state.token_count = std::max(state.token_count, position_value + 1);
}

void CpuModel::Impl::run_attention(const AttentionState& state,
                                   const float* q, float* output,
                                   int sequence_length) const {
    if (sequence_length <= 0 || static_cast<size_t>(sequence_length) > state.token_count) {
        throw std::invalid_argument("CPU paged attention sequence length is invalid");
    }
    const CpuKvPagePool& pool = *shared->kv_pools.at(state.pool_index);
    CpuPagedAttentionStats attention_stats;
    cpu_gqa_decode_paged_parallel(
        q, pool, state.pages, output, sequence_length,
        shared->shape.num_attention_heads, shared->shape.num_key_value_heads, shared->shape.head_dim,
        shared->pool,
        CpuPagedAttentionOptions{
            shared->options.attention_parallel_threshold,
            shared->options.attention_page_tile},
        &attention_stats);
    if (attention_stats.parallel) {
        ++const_cast<Impl*>(this)->attention_parallel_calls;
    }
}

CpuPrefixSnapshot CpuModel::Impl::export_prefix_snapshot() const {
    CpuPrefixSnapshot snapshot;
    snapshot.position = static_cast<size_t>(position_value);
    snapshot.numa_node = preferred_numa_node;
    snapshot.logits = logits;
    snapshot.seen_tokens = seen;
    for (const LayerState& layer : states) {
        if (const auto* attention = std::get_if<AttentionState>(&layer)) {
            snapshot.attention_pages.push_back(attention->pages);
            snapshot.attention_token_counts.push_back(attention->token_count);
        } else {
            snapshot.convolution_states.push_back(
                std::get<ConvolutionState>(layer).state);
        }
    }
    return snapshot;
}

void CpuModel::Impl::restore_prefix_snapshot(CpuPrefixSnapshot snapshot,
                                             bool ready_for_decode) {
    size_t expected_attention = 0;
    size_t expected_convolution = 0;
    for (const LayerState& layer : states) {
        if (std::holds_alternative<AttentionState>(layer)) ++expected_attention;
        else ++expected_convolution;
    }
    if (snapshot.attention_pages.size() != expected_attention ||
        snapshot.attention_token_counts.size() != expected_attention ||
        snapshot.convolution_states.size() != expected_convolution ||
        snapshot.logits.size() != logits.size() ||
        snapshot.seen_tokens.size() != seen.size() ||
        snapshot.position > static_cast<size_t>(shared->max_context)) {
        throw std::invalid_argument("CPU prefix snapshot shape is invalid");
    }
    for (size_t index = 0; index < snapshot.attention_token_counts.size(); ++index) {
        const size_t required_pages = snapshot.attention_token_counts[index] == 0 ? 0 :
            (snapshot.attention_token_counts[index] +
             shared->options.kv_page_tokens - 1) /
            shared->options.kv_page_tokens;
        if (snapshot.attention_pages[index].size() < required_pages) {
            throw std::invalid_argument("CPU prefix snapshot page table is incomplete");
        }
    }

    reset();
    size_t attention_index = 0;
    size_t convolution_index = 0;
    try {
        for (LayerState& layer : states) {
            if (auto* attention = std::get_if<AttentionState>(&layer)) {
                attention->pages = std::move(snapshot.attention_pages[attention_index]);
                attention->token_count = snapshot.attention_token_counts[attention_index];
                ++attention_index;
            } else {
                std::get<ConvolutionState>(layer).state =
                    std::move(snapshot.convolution_states[convolution_index]);
                ++convolution_index;
            }
        }
        logits = std::move(snapshot.logits);
        seen = std::move(snapshot.seen_tokens);
        position_value = static_cast<int>(snapshot.position);
        preferred_numa_node = snapshot.numa_node;
        phase = ready_for_decode ? SessionPhase::Ready : SessionPhase::Prefilling;
        rng_state = generation.seed;
    } catch (...) {
        reset();
        // Pages not yet transferred remain owned by snapshot. Release them.
        for (size_t index = attention_index;
             index < snapshot.attention_pages.size(); ++index) {
            auto& pool = *shared->kv_pools.at(index);
            for (CpuKvPageId page : snapshot.attention_pages[index]) {
                try {
                    pool.release(page);
                } catch (const std::exception& error) {
                    std::clog << "CPU prefix snapshot rollback failed: "
                              << error.what() << '\n';
                } catch (...) {
                    std::clog << "CPU prefix snapshot rollback failed: unknown exception\n";
                }
            }
        }
        throw;
    }
}

void CpuModel::Impl::reset() {
    position_value = 0;
    phase = SessionPhase::Empty;
    metrics = {};
    rng_state = generation.seed;
    std::fill(seen.begin(), seen.end(), uint8_t{0});
    std::fill(logits.begin(), logits.end(), 0.0f);
    for (LayerState& state : states) {
        if (auto* attention = std::get_if<AttentionState>(&state)) {
            release_attention_pages(*attention);
        } else {
            auto& convolution = std::get<ConvolutionState>(state);
            std::fill(convolution.state.begin(), convolution.state.end(), 0.0f);
        }
    }
}

CpuModelMemoryStats CpuModel::Impl::memory_stats() const {
    CpuModelMemoryStats stats;
    stats.weights = shared->weights_memory_bytes();
    for (const LayerState& state : states) {
        std::visit([&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, AttentionState>) {
                const auto& pool = *shared->kv_pools.at(value.pool_index);
                stats.kv_cache += value.pages.size() * pool.page_bytes();
                stats.kv_pages_used += value.pages.size();
                stats.kv_pages_total += pool.stats().total_pages;
            } else {
                stats.conv_state += value.state.size() * sizeof(float);
            }
        }, state);
    }
    stats.activations =
        (hidden.size() + residual.size() + normed.size() + op_output.size() +
         qkv.size() + conv_projected.size() + gate_up.size() + activated.size() +
         mlp_output.size() + logits.size() + chunk_hidden.size() +
         chunk_residual.size() + chunk_normed.size() + chunk_op.size() +
         chunk_qkv.size() + chunk_conv.size() + chunk_gate_up.size() +
         chunk_activated.size() + chunk_mlp.size()) * sizeof(float) + seen.size();
    return stats;
}

} // namespace lfm
