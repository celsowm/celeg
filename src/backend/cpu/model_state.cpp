#include "detail/model_internal.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace celeg {

void configure_cpu_expert_backing(CpuCompiledModel::Shared& shared);

CpuCompiledModel::CpuCompiledModel(std::shared_ptr<Shared> shared_weights,
                                   GenerationConfig generation_config,
                                   int requested_numa_node)
    : shared(std::move(shared_weights)), session_(generation_config),
      preferred_numa_node(requested_numa_node) {
    if (!shared) throw std::invalid_argument("shared CPU model weights are required");
    configure_cpu_expert_backing(*shared);
    session_.generation.validate();
    allocate_state();
    allocate_activations();
    reset();
}

CpuCompiledModel::~CpuCompiledModel() {
    for (LayerState& layer : session_.states) {
        if (auto* attention = std::get_if<AttentionState>(&layer)) {
            release_attention_pages(*attention);
        }
    }
}

void CpuCompiledModel::allocate_state() {
    session_.states.reserve(shared->weight_store.layers.size());
    for (size_t index = 0; index < shared->weight_store.layers.size(); ++index) {
        const WeightLayer& layer = shared->weight_store.layers[index];
        const auto emplace_convolution_state = [&]() {
            ConvolutionState state;
            state.state.resize(static_cast<size_t>(shared->shape.conv_cache) *
                               shared->program.hidden);
            session_.states.emplace_back(std::move(state));
        };
        visit_operator_weights(layer,
          [&](const AttentionWeights*) {
            const int pool = shared->layer_to_kv_pool.at(index);
            if (pool < 0) throw std::logic_error("attention layer has no CPU KV page pool");
            AttentionState state;
            state.pool_index = static_cast<size_t>(pool);
            session_.states.emplace_back(std::move(state));
          },
          [&](const ConvolutionWeights*) { emplace_convolution_state(); },
          [&](const Mamba2Weights*) {
            Mamba2State state;
            const auto& spec = std::get<Mamba2Spec>(
                shared->program.layers.at(index).mixer);
            state.conv.resize(static_cast<size_t>(spec.intermediate_size +
                              2 * spec.group_count * spec.state_size) *
                              static_cast<size_t>(spec.conv_kernel));
            state.ssm.resize(static_cast<size_t>(spec.intermediate_size) *
                             static_cast<size_t>(spec.state_size));
            session_.states.emplace_back(std::move(state));
          },
          [&](const GatedDeltaNetWeights* gated_delta) {
            GatedDeltaNetState state;
            const auto& spec = gated_delta->spec;
            const int conv_dim = 2 * spec.key_heads * spec.key_head_dim +
                spec.value_heads * spec.value_head_dim;
            state.conv.resize(static_cast<size_t>(conv_dim) * spec.conv_kernel);
            state.recurrent.resize(static_cast<size_t>(spec.value_heads) *
                spec.key_head_dim * spec.value_head_dim);
            session_.states.emplace_back(std::move(state));
          },
          // MLP-only blocks carry no mixer state; they keep an empty
          // convolution slot so session state stays index-parallel with the
          // layer list and snapshot counts remain stable.
          [&](const MlpOnlyWeights*) { emplace_convolution_state(); });
    }
}

void CpuCompiledModel::allocate_activations() {
    workspace_.ensure(1, shared->workspace_plan);
    workspace_.logits.resize(shared->dims.vocab_size);
    session_.seen.resize(shared->dims.vocab_size);
}

const CpuCompiledModel::CommonWeights& CpuCompiledModel::common_weights(
    size_t layer) const {
    return std::visit([](const auto& value) -> const CommonWeights& {
        return value.common;
    }, shared->weight_store.layers.at(layer));
}

CpuCompiledModel::AttentionState& CpuCompiledModel::attention_state(size_t layer) {
    return std::get<AttentionState>(session_.states.at(layer));
}

const CpuCompiledModel::AttentionState& CpuCompiledModel::attention_state(
    size_t layer) const {
    return std::get<AttentionState>(session_.states.at(layer));
}

CpuCompiledModel::ConvolutionState& CpuCompiledModel::convolution_state(
    size_t layer) {
    return std::get<ConvolutionState>(session_.states.at(layer));
}

const CpuCompiledModel::ConvolutionState& CpuCompiledModel::convolution_state(
    size_t layer) const {
    return std::get<ConvolutionState>(session_.states.at(layer));
}

CpuCompiledModel::Mamba2State& CpuCompiledModel::mamba2_state(size_t layer) {
    return std::get<Mamba2State>(session_.states.at(layer));
}

const CpuCompiledModel::Mamba2State& CpuCompiledModel::mamba2_state(size_t layer) const {
    return std::get<Mamba2State>(session_.states.at(layer));
}

CpuCompiledModel::GatedDeltaNetState& CpuCompiledModel::gated_delta_net_state(size_t layer) {
    return std::get<GatedDeltaNetState>(session_.states.at(layer));
}

const CpuCompiledModel::GatedDeltaNetState& CpuCompiledModel::gated_delta_net_state(
    size_t layer) const {
    return std::get<GatedDeltaNetState>(session_.states.at(layer));
}

void CpuCompiledModel::release_attention_pages(AttentionState& state) noexcept {
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

void CpuCompiledModel::store_kv(AttentionState& state, int position,
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

void CpuCompiledModel::store_latent(AttentionState& state, int position,
                                    const float* key, const float* value,
                                    const float* rotary) {
    if (position < 0) throw std::invalid_argument("negative CPU latent position");
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
    pool.write_latent(state.pages[page_index], token_offset, key, value, rotary);
    state.token_count = std::max(state.token_count, position_value + 1);
}

void CpuCompiledModel::run_attention(const AttentionState& state,
                                     const AttentionSpec& attention,
                                     const float* q, float* output,
                                     int sequence_length,
                                     std::span<const float> relative_bias) const {
    if (sequence_length <= 0 || static_cast<size_t>(sequence_length) > state.token_count) {
        throw std::invalid_argument("CPU paged attention sequence length is invalid");
    }
    const CpuKvPagePool& pool = *shared->kv_pools.at(state.pool_index);
    CpuPagedAttentionStats attention_stats;
    cpu_gqa_decode_paged_parallel(
        q, pool, state.pages, output, sequence_length,
        attention.query_heads, attention.key_value_heads, attention.head_dim,
        shared->pool,
        CpuAttentionPattern::lower(attention.pattern),
        CpuAttentionBias::lower(attention.bias, relative_bias, attention.query_heads),
        CpuPagedAttentionOptions{
            shared->options.attention_parallel_threshold,
            shared->options.attention_page_tile},
        &attention_stats);
    if (attention_stats.parallel) {
        ++const_cast<CpuCompiledModel*>(this)->session_.attention_parallel_calls;
    }
}

void CpuCompiledModel::run_latent_attention(
    const AttentionState& state, const AttentionSpec& attention,
    const float* query_content, const float* query_rope,
    float* output, int sequence_length, int query_position,
    std::span<const float> relative_bias) const {
    const auto& latent = *attention.latent_state();
    if (sequence_length <= 0 || static_cast<size_t>(sequence_length) > state.token_count) {
        throw std::invalid_argument("CPU latent attention sequence length is invalid");
    }
    const CpuKvPagePool& pool = *shared->kv_pools.at(state.pool_index);
    const int rotary_width = latent.decoupled_rope ? latent.rope_head_dim : 0;
    cpu_latent_attention_decode_paged(
        query_content, query_rope, pool, state.pages, output, sequence_length,
        attention.query_heads, latent.latent_rank, rotary_width,
        attention.query_scale,
        CpuAttentionPattern::lower(attention.pattern),
        CpuAttentionBias::lower(attention.bias, relative_bias, attention.query_heads), query_position);
}

void CpuCompiledModel::set_external_attention_memory(
    int slot, std::shared_ptr<const CpuExternalAttentionMemory> memory) {
    if (slot < 0 || !memory) {
        throw std::invalid_argument("external attention memory slot and memory are required");
    }
    if (memory->token_count() == 0) {
        throw std::invalid_argument("external attention memory cannot be empty");
    }
    shared->external_attention_memory[slot] = std::move(memory);
}

void CpuCompiledModel::run_external_attention(
    const AttentionSpec& attention, const CpuExternalAttentionMemory& memory,
    const float* q, float* output, std::span<const float> relative_bias) const {
    if (memory.token_count() == 0) {
        throw std::invalid_argument("external attention memory is empty");
    }
    if (memory.pool().key_width() != static_cast<size_t>(
            attention.key_value_heads * attention.head_dim) ||
        memory.pool().value_width() != static_cast<size_t>(
            attention.key_value_heads * attention.head_dim)) {
        throw std::invalid_argument("external attention memory width does not match attention");
    }
    cpu_gqa_decode_paged(
        q, memory.pool(), memory.pages(), output,
        static_cast<int>(memory.token_count()), attention.query_heads,
        attention.key_value_heads, attention.head_dim,
        CpuAttentionPattern::lower(BidirectionalPattern{}),
        CpuAttentionBias::lower(attention.bias, relative_bias, attention.query_heads),
        static_cast<int>(memory.token_count()) - 1);
}

CpuPrefixSnapshot CpuCompiledModel::export_prefix_snapshot() const {
    CpuPrefixSnapshot snapshot;
    snapshot.position = static_cast<size_t>(session_.position_value);
    snapshot.numa_node = preferred_numa_node;
    snapshot.logits = workspace_.logits;
    snapshot.seen_tokens = session_.seen;
    for (const LayerState& layer : session_.states) {
        visit_exhaustive(layer,
          [&](const AttentionState* attention) {
            snapshot.attention_pages.push_back(attention->pages);
            snapshot.attention_token_counts.push_back(attention->token_count);
          },
          [&](const ConvolutionState* convolution) {
            snapshot.convolution_states.push_back(convolution->state);
          },
          [&](const GatedDeltaNetState* gated_delta) {
            auto state = gated_delta->conv;
            state.insert(state.end(), gated_delta->recurrent.begin(),
                         gated_delta->recurrent.end());
            snapshot.gated_delta_states.push_back(std::move(state));
          },
          [&](const Mamba2State* mamba) {
            auto state = mamba->conv;
            state.insert(state.end(), mamba->ssm.begin(), mamba->ssm.end());
            snapshot.mamba_states.push_back(std::move(state));
          });
    }
    return snapshot;
}

void CpuCompiledModel::restore_prefix_snapshot(CpuPrefixSnapshot snapshot,
                                               bool ready_for_decode) {
    size_t expected_attention = 0;
    size_t expected_convolution = 0;
    size_t expected_gated_delta = 0;
    size_t expected_mamba = 0;
    for (const LayerState& layer : session_.states) {
        visit_exhaustive(layer,
          [&](const AttentionState*) { ++expected_attention; },
          [&](const ConvolutionState*) { ++expected_convolution; },
          [&](const GatedDeltaNetState*) { ++expected_gated_delta; },
          [&](const Mamba2State*) { ++expected_mamba; });
    }
    if (snapshot.attention_pages.size() != expected_attention ||
        snapshot.attention_token_counts.size() != expected_attention ||
        snapshot.convolution_states.size() != expected_convolution ||
        snapshot.gated_delta_states.size() != expected_gated_delta ||
        snapshot.mamba_states.size() != expected_mamba ||
        snapshot.logits.size() != workspace_.logits.size() ||
        snapshot.seen_tokens.size() != session_.seen.size() ||
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
    size_t gated_delta_index = 0;
    size_t mamba_index = 0;
    try {
        for (LayerState& layer : session_.states) {
            visit_exhaustive(layer,
              [&](AttentionState* attention) {
                attention->pages = std::move(snapshot.attention_pages[attention_index]);
                attention->token_count = snapshot.attention_token_counts[attention_index];
                ++attention_index;
              },
              [&](ConvolutionState* convolution) {
                convolution->state = std::move(snapshot.convolution_states[convolution_index++]);
              },
              [&](GatedDeltaNetState* gated_delta) {
                const auto& packed = snapshot.gated_delta_states[gated_delta_index++];
                if (packed.size() != gated_delta->conv.size() + gated_delta->recurrent.size()) {
                    throw std::invalid_argument("CPU GatedDeltaNet snapshot state shape is invalid");
                }
                std::copy_n(packed.begin(), gated_delta->conv.size(), gated_delta->conv.begin());
                std::copy(packed.begin() + static_cast<std::ptrdiff_t>(gated_delta->conv.size()),
                          packed.end(), gated_delta->recurrent.begin());
              },
              [&](Mamba2State* mamba) {
                const auto& packed = snapshot.mamba_states[mamba_index++];
                if (packed.size() != mamba->conv.size() + mamba->ssm.size()) {
                    throw std::invalid_argument("CPU Mamba snapshot state shape is invalid");
                }
                std::copy_n(packed.begin(), mamba->conv.size(), mamba->conv.begin());
                std::copy(packed.begin() + static_cast<std::ptrdiff_t>(mamba->conv.size()),
                          packed.end(), mamba->ssm.begin());
              });
        }
        workspace_.logits = std::move(snapshot.logits);
        session_.seen = std::move(snapshot.seen_tokens);
        session_.position_value = static_cast<int>(snapshot.position);
        preferred_numa_node = snapshot.numa_node;
        session_.phase = ready_for_decode ? SessionPhase::Ready : SessionPhase::Prefilling;
        session_.rng_state = session_.generation.seed;
    } catch (...) {
        reset();
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

void CpuCompiledModel::reset() {
    session_.position_value = 0;
    session_.next_rope_position = {0, 0, 0};
    session_.phase = SessionPhase::Empty;
    session_.metrics = {};
    session_.rng_state = session_.generation.seed;
    std::fill(session_.seen.begin(), session_.seen.end(), uint8_t{0});
    std::fill(workspace_.logits.begin(), workspace_.logits.end(), 0.0f);
    for (LayerState& state : session_.states) {
        visit_exhaustive(state,
          [&](AttentionState* attention) { release_attention_pages(*attention); },
          [](ConvolutionState* convolution) {
            std::fill(convolution->state.begin(), convolution->state.end(), 0.0f);
          },
          [](GatedDeltaNetState* gated_delta) {
            std::fill(gated_delta->conv.begin(), gated_delta->conv.end(), 0.0f);
            std::fill(gated_delta->recurrent.begin(), gated_delta->recurrent.end(), 0.0f);
          },
          [](Mamba2State* mamba) {
            std::fill(mamba->conv.begin(), mamba->conv.end(), 0.0f);
            std::fill(mamba->ssm.begin(), mamba->ssm.end(), 0.0f);
          });
    }
}

void CpuCompiledModel::set_generation(GenerationConfig config) {
    config.validate();
    session_.generation = std::move(config);
    session_.rng_state = session_.generation.seed;
}

CpuModelMemoryStats CpuCompiledModel::memory_stats() const {
    CpuModelMemoryStats stats;
    stats.weights = shared->weights_memory_bytes();
    if (shared->expert_backing_store) {
        stats.weights += shared->expert_backing_store->metrics().resident_bytes;
    }
    for (const LayerState& state : session_.states) {
        visit_exhaustive(state,
          [&](const AttentionState* attention) {
            const auto& pool = *shared->kv_pools.at(attention->pool_index);
            stats.kv_cache += attention->pages.size() * pool.page_bytes();
            stats.kv_pages_used += attention->pages.size();
            stats.kv_pages_total += pool.stats().total_pages;
          },
          [&](const ConvolutionState* convolution) {
            stats.conv_state += convolution->state.size() * sizeof(float);
          },
          [&](const GatedDeltaNetState* gated_delta) {
            stats.conv_state +=
                (gated_delta->conv.size() + gated_delta->recurrent.size()) * sizeof(float);
          },
          [&](const Mamba2State* mamba) {
            stats.conv_state += (mamba->conv.size() + mamba->ssm.size()) * sizeof(float);
          });
    }
    stats.activations =
        (workspace_.hidden.size() + workspace_.residual.size() + workspace_.normed.size() + workspace_.op_output.size() +
         workspace_.qkv.size() + workspace_.conv_projected.size() + workspace_.gate_up.size() + workspace_.activated.size() +
         workspace_.mlp_output.size() + workspace_.logits.size() + workspace_.chunk_hidden.size() +
         workspace_.chunk_residual.size() + workspace_.chunk_normed.size() + workspace_.chunk_op.size() +
         workspace_.chunk_qkv.size() + workspace_.chunk_attention_gate.size() +
         workspace_.chunk_conv.size() + workspace_.chunk_gate_up.size() +
         workspace_.chunk_activated.size() + workspace_.chunk_mlp.size() +
         workspace_.gated_delta_qkv.size() + workspace_.gated_delta_z.size() +
         workspace_.gated_delta_b.size() + workspace_.gated_delta_a.size() +
         workspace_.gated_delta_output.size() + workspace_.chunk_gated_delta_qkv.size() +
         workspace_.chunk_gated_delta_z.size() + workspace_.chunk_gated_delta_b.size() +
         workspace_.chunk_gated_delta_a.size() + workspace_.chunk_gated_delta_output.size()) *
         sizeof(float) + session_.seen.size();
    return stats;
}

} // namespace celeg
