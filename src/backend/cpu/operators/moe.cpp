#include "moe.hpp"
#include "common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace celeg {

void configure_cpu_expert_backing(CpuCompiledModel::Shared& shared);

namespace {

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

} // namespace

CpuMoeRoute route_cpu_moe(const RouterProgram& program,
                          std::span<const float> logits,
                          std::span<const float> expert_bias) {
    if (logits.size() != static_cast<size_t>(program.expert_count)) {
        throw std::invalid_argument("CPU MoE router logit width does not match program");
    }
    std::vector<float> probabilities(logits.size());
    const float maximum = program.score == MoeRouterScoreKind::SoftmaxLogits
        ? *std::max_element(logits.begin(), logits.end()) : 0.0f;
    float probability_sum = 0.0f;
    for (size_t index = 0; index < logits.size(); ++index) {
        probabilities[index] = program.score == MoeRouterScoreKind::SoftmaxLogits
            ? std::exp(logits[index] - maximum) : sigmoid(logits[index]);
        probability_sum += probabilities[index];
    }
    if (program.score == MoeRouterScoreKind::SoftmaxLogits) {
        for (float& probability : probabilities) probability /= probability_sum;
    }

    std::vector<std::pair<float, int>> scored(logits.size());
    for (size_t index = 0; index < logits.size(); ++index) {
        const float bias = program.has_expert_bias && index < expert_bias.size()
            ? expert_bias[index] : 0.0f;
        scored[index] = {probabilities[index] + bias, static_cast<int>(index)};
    }
    std::partial_sort(scored.begin(), scored.begin() + program.experts_per_token,
                      scored.end(), [](const auto& left, const auto& right) {
        return left.first == right.first ? left.second < right.second
                                         : left.first > right.first;
    });

    CpuMoeRoute result;
    result.experts.resize(static_cast<size_t>(program.experts_per_token));
    result.weights.resize(result.experts.size());
    float selected_sum = 0.0f;
    for (int route = 0; route < program.experts_per_token; ++route) {
        const int expert = scored[static_cast<size_t>(route)].second;
        result.experts[static_cast<size_t>(route)] = expert;
        result.weights[static_cast<size_t>(route)] = probabilities[static_cast<size_t>(expert)];
        selected_sum += result.weights[static_cast<size_t>(route)];
    }
    if (program.normalization == MoeNormalizationKind::SumSelected) {
        const float inverse = 1.0f / (selected_sum + 1.0e-6f);
        for (float& weight : result.weights) weight *= inverse;
    }
    for (float& weight : result.weights) weight *= program.routed_scaling;
    return result;
}

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void prepare_expert_backing(CpuCompiledModel::Shared& shared,
                            const CpuCompiledModel::MoeWeights& weights) {
    if (shared.options.expert_backing == CpuExpertBacking::DiskCached &&
        !weights.disk_cached && !shared.native_checkpoint) {
        configure_cpu_expert_backing(shared);
    }
}

const CpuExpertWeights* acquire_expert(
    CpuCompiledModel::Shared& shared,
    const CpuCompiledModel::MoeWeights& weights,
    int expert,
    std::shared_ptr<const CpuExpertWeights>& lease) {
    if (weights.disk_cached) {
        lease = shared.acquire_expert(weights.layer_index, expert);
        return lease.get();
    }
    return nullptr;
}

} // namespace

void execute_cpu_moe_token(CpuCompiledModel& model, size_t layer,
                           const CpuCompiledModel::MoeWeights& weights,
                           const MoeLayerProgram& semantics) {
    auto& shared = *model.shared;
    auto& workspace = model.workspace_;
    const size_t hidden = static_cast<size_t>(shared.shape.hidden);
    const int experts = semantics.router.expert_count;
    const int selected = semantics.router.experts_per_token;
    const int intermediate = semantics.routed.mlp.intermediate_size;
    prepare_expert_backing(shared, weights);

    const bool profile = model.session_.phase == SessionPhase::Prefilling;
    const auto router_started = Clock::now();
    workspace.moe_router_logits.resize(static_cast<size_t>(experts));
    workspace.moe_selected.resize(static_cast<size_t>(selected));
    workspace.moe_weights.resize(static_cast<size_t>(selected));
    shared.linear.gemv_raw(weights.router.data(), workspace.normed.data(),
                           workspace.moe_router_logits.data(), experts,
                           shared.shape.hidden);
    const CpuMoeRoute route = route_cpu_moe(
        semantics.router, workspace.moe_router_logits, weights.router_bias);
    std::copy(route.experts.begin(), route.experts.end(), workspace.moe_selected.begin());
    std::copy(route.weights.begin(), route.weights.end(), workspace.moe_weights.begin());
    if (profile) model.session_.prefill_profile.moe_router_ms += elapsed_ms(router_started);

    const auto expert_started = Clock::now();
    std::fill(workspace.mlp_output.begin(), workspace.mlp_output.begin() + hidden, 0.0f);
    for (int route_index = 0; route_index < selected; ++route_index) {
        const int expert = workspace.moe_selected[static_cast<size_t>(route_index)];
        if (expert < 0 || expert >= experts) continue;
        std::shared_ptr<const CpuExpertWeights> lease;
        const CpuLinearWeight* w13 = nullptr;
        const CpuLinearWeight* w2 = nullptr;
        if (weights.disk_cached) {
            const CpuExpertWeights* cached = acquire_expert(shared, weights, expert, lease);
            w13 = &cached->w13;
            w2 = &cached->w2;
        } else {
            w13 = &weights.expert_w13[static_cast<size_t>(expert)];
            w2 = &weights.expert_w2[static_cast<size_t>(expert)];
        }
        shared.linear.gemv(*w13, workspace.normed.data(), workspace.gate_up.data());
        cpu_swiglu(workspace.gate_up.data(), workspace.activated.data(), intermediate);
        shared.linear.gemv(*w2, workspace.activated.data(), workspace.op_output.data());
        const float route_weight = workspace.moe_weights[static_cast<size_t>(route_index)];
        for (size_t d = 0; d < hidden; ++d) {
            workspace.mlp_output[d] += route_weight * workspace.op_output[d];
        }
    }

    if (semantics.shared) {
        const int shared_intermediate = semantics.shared->mlp.intermediate_size;
        shared.linear.gemv(weights.shared_w13, workspace.normed.data(),
                           workspace.gate_up.data());
        cpu_swiglu(workspace.gate_up.data(), workspace.activated.data(),
                   shared_intermediate);
        shared.linear.gemv(weights.shared_w2, workspace.activated.data(),
                           workspace.shared_output.data());
        shared.linear.gemv(weights.shared_gate, workspace.normed.data(),
                           workspace.shared_gate.data());
        const float gate = cpu_moe_sigmoid(workspace.shared_gate.front());
        for (size_t d = 0; d < hidden; ++d) {
            workspace.mlp_output[d] += gate * workspace.shared_output[d];
        }
    }
    if (profile) model.session_.prefill_profile.moe_expert_ms += elapsed_ms(expert_started);
    (void)layer;
}

void execute_cpu_moe_chunk(CpuCompiledModel& model, size_t layer,
                           const CpuCompiledModel::MoeWeights& weights,
                           const MoeLayerProgram& semantics,
                           size_t rows, bool& normed_q8_ready) {
    auto& shared = *model.shared;
    auto& workspace = model.workspace_;
    const size_t hidden = static_cast<size_t>(shared.shape.hidden);
    const int experts = semantics.router.expert_count;
    const int selected = semantics.router.experts_per_token;
    const int intermediate = semantics.routed.mlp.intermediate_size;
    const size_t routes = rows * static_cast<size_t>(selected);
    prepare_expert_backing(shared, weights);

    std::fill(workspace.chunk_mlp.begin(),
              workspace.chunk_mlp.begin() + rows * hidden, 0.0f);
    workspace.moe_router_logits.resize(rows * static_cast<size_t>(experts));
    workspace.moe_selected.resize(routes);
    workspace.moe_weights.resize(routes);
    workspace.moe_route_rows.resize(routes);
    workspace.moe_route_experts.resize(routes);
    workspace.moe_route_weights.resize(routes);
    workspace.moe_group_offsets.assign(static_cast<size_t>(experts) + 1, 0);
    workspace.moe_group_cursor.resize(static_cast<size_t>(experts));
    workspace.moe_route_order.resize(routes);

    const auto router_started = Clock::now();
    shared.linear.gemm_raw(weights.router.data(), workspace.chunk_normed.data(),
                           workspace.moe_router_logits.data(), rows, experts,
                           shared.shape.hidden);
    for (size_t row = 0; row < rows; ++row) {
        const float* logits = workspace.moe_router_logits.data() +
            row * static_cast<size_t>(experts);
        const CpuMoeRoute route = route_cpu_moe(
            semantics.router, {logits, static_cast<size_t>(experts)},
            weights.router_bias);
        for (int route_index = 0; route_index < selected; ++route_index) {
            const size_t index = row * static_cast<size_t>(selected) +
                static_cast<size_t>(route_index);
            const int expert = route.experts[static_cast<size_t>(route_index)];
            workspace.moe_selected[index] = expert;
            workspace.moe_weights[index] = route.weights[static_cast<size_t>(route_index)] /
                semantics.router.routed_scaling;
            workspace.moe_route_weights[index] =
                route.weights[static_cast<size_t>(route_index)];
            workspace.moe_route_rows[index] = static_cast<int>(row);
            workspace.moe_route_experts[index] = expert;
            ++workspace.moe_group_offsets[static_cast<size_t>(expert) + 1];
        }
    }
    for (int expert = 0; expert < experts; ++expert) {
        workspace.moe_group_offsets[static_cast<size_t>(expert) + 1] +=
            workspace.moe_group_offsets[static_cast<size_t>(expert)];
        workspace.moe_group_cursor[static_cast<size_t>(expert)] =
            workspace.moe_group_offsets[static_cast<size_t>(expert)];
    }
    for (size_t route = 0; route < routes; ++route) {
        const int expert = workspace.moe_route_experts[route];
        workspace.moe_route_order[workspace.moe_group_cursor[static_cast<size_t>(expert)]++] = route;
    }
    if (model.session_.phase == SessionPhase::Prefilling) {
        model.session_.prefill_profile.moe_router_ms += elapsed_ms(router_started);
    }

    const auto expert_started = Clock::now();
    workspace.moe_gathered_normed.resize(routes * hidden);
    workspace.moe_gathered_gate_up.resize(routes * 2ULL * static_cast<size_t>(intermediate));
    workspace.moe_gathered_activated.resize(routes * static_cast<size_t>(intermediate));
    workspace.moe_gathered_output.resize(routes * hidden);
    workspace.moe_cached_experts.clear();
    workspace.moe_cached_experts.resize(static_cast<size_t>(experts));
    workspace.moe_gemm_jobs.clear();
    for (int expert = 0; expert < experts; ++expert) {
        const size_t begin = workspace.moe_group_offsets[static_cast<size_t>(expert)];
        const size_t end = workspace.moe_group_offsets[static_cast<size_t>(expert) + 1];
        if (begin == end) continue;
        const CpuLinearWeight* w13 = nullptr;
        if (weights.disk_cached) {
            workspace.moe_cached_experts[static_cast<size_t>(expert)] =
                shared.acquire_expert(weights.layer_index, expert);
            w13 = &workspace.moe_cached_experts[static_cast<size_t>(expert)]->w13;
        } else {
            w13 = &weights.expert_w13[static_cast<size_t>(expert)];
        }
        workspace.moe_gemm_jobs.push_back({w13, begin, end - begin});
    }
    for (size_t packed_route = 0; packed_route < routes; ++packed_route) {
        const size_t route = workspace.moe_route_order[packed_route];
        const size_t row = static_cast<size_t>(workspace.moe_route_rows[route]);
        std::copy_n(workspace.chunk_normed.data() + row * hidden, hidden,
                    workspace.moe_gathered_normed.data() + packed_route * hidden);
    }
    shared.linear.gemm_grouped(workspace.moe_gemm_jobs,
                               workspace.moe_gathered_normed.data(),
                               workspace.moe_gathered_gate_up.data());
    cpu_parallel_rows(shared.pool, routes, [&](size_t route) {
        cpu_swiglu(workspace.moe_gathered_gate_up.data() +
                       route * 2ULL * static_cast<size_t>(intermediate),
                   workspace.moe_gathered_activated.data() +
                       route * static_cast<size_t>(intermediate), intermediate);
    });
    workspace.moe_gemm_jobs.clear();
    for (int expert = 0; expert < experts; ++expert) {
        const size_t begin = workspace.moe_group_offsets[static_cast<size_t>(expert)];
        const size_t end = workspace.moe_group_offsets[static_cast<size_t>(expert) + 1];
        if (begin == end) continue;
        const CpuLinearWeight* w2 = weights.disk_cached
            ? &workspace.moe_cached_experts[static_cast<size_t>(expert)]->w2
            : &weights.expert_w2[static_cast<size_t>(expert)];
        workspace.moe_gemm_jobs.push_back({w2, begin, end - begin});
    }
    shared.linear.gemm_grouped(workspace.moe_gemm_jobs,
                               workspace.moe_gathered_activated.data(),
                               workspace.moe_gathered_output.data());
    for (size_t packed_route = 0; packed_route < routes; ++packed_route) {
        const size_t route = workspace.moe_route_order[packed_route];
        const size_t row = static_cast<size_t>(workspace.moe_route_rows[route]);
        float* destination = workspace.chunk_mlp.data() + row * hidden;
        const float* source = workspace.moe_gathered_output.data() + packed_route * hidden;
        const float weight = workspace.moe_route_weights[route];
        for (size_t d = 0; d < hidden; ++d) destination[d] += weight * source[d];
    }

    if (semantics.shared) {
        const int shared_intermediate = semantics.shared->mlp.intermediate_size;
        cpu_layer_gemm(shared, workspace, weights.shared_w13,
                       workspace.chunk_normed.data(), workspace.chunk_gate_up.data(),
                       rows, hidden, normed_q8_ready);
        cpu_parallel_rows(shared.pool, rows, [&](size_t row) {
            cpu_swiglu(workspace.chunk_gate_up.data() +
                           row * 2ULL * static_cast<size_t>(shared_intermediate),
                       workspace.chunk_activated.data() +
                           row * static_cast<size_t>(shared_intermediate),
                       shared_intermediate);
        });
        cpu_layer_gemm(shared, workspace, weights.shared_w2,
                       workspace.chunk_activated.data(), workspace.shared_output.data(),
                       rows, hidden, normed_q8_ready);
        cpu_layer_gemm(shared, workspace, weights.shared_gate,
                       workspace.chunk_normed.data(), workspace.shared_gate.data(),
                       rows, hidden, normed_q8_ready);
        cpu_parallel_rows(shared.pool, rows, [&](size_t row) {
            const float gate = cpu_moe_sigmoid(workspace.shared_gate[row]);
            float* destination = workspace.chunk_mlp.data() + row * hidden;
            const float* source = workspace.shared_output.data() + row * hidden;
            for (size_t d = 0; d < hidden; ++d) destination[d] += gate * source[d];
        });
    }
    if (model.session_.phase == SessionPhase::Prefilling) {
        model.session_.prefill_profile.moe_expert_ms += elapsed_ms(expert_started);
    }
    (void)layer;
}

} // namespace celeg
