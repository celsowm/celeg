#include "celeg/backend/cuda/packed.hpp"
#include "celeg/backend/cuda/packed_workspace.hpp"
#include "celeg/backend/cuda/packed_metadata.hpp"
#include "celeg/backend/cuda/packed_commit.hpp"
#include "celeg/backend/cuda/packed_layer_program.hpp"
#include "celeg/backend/cuda/packed_gemm_runtime.hpp"
#include "celeg/backend/cuda/packed_metadata_cache.hpp"
#include "celeg/backend/cuda/packed_operators.hpp"
#include "celeg/backend/cuda/packed_pipelines.hpp"

#include "celeg/backend/cuda/kernels/kernels.cuh"
#include "celeg/backend/cuda/moe.hpp"
#include "celeg/detail/model/compiled_model.hpp"
#include "celeg/backend/cuda/paged_kv.hpp"
#include "celeg/backend/cuda/gemm_dispatcher.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <string>
#include <utility>

namespace celeg {

struct PackedDecodeExecutorImpl : PackedWorkspace {
    explicit PackedDecodeExecutorImpl(size_t maximum_batch_value,
                                      size_t maximum_prefill_tokens_value,
                                      PhysicalPagedKvCache* paged_kv_value,
                                      const RuntimeTopology& shape,
                                      CudaExecutionPlan plan_value)
        : PackedWorkspace(maximum_batch_value, maximum_prefill_tokens_value,
                          paged_kv_value, shape),
          layer_program_(PackedLayerProgram::compile(shape)),
          plan_(std::move(plan_value)),
          metadata_cache_(maximum_batch) {
        // Segmented attention workspace is part of the executor's fixed
        // capacity contract. Allocate it before request execution so the
        // scheduler never grows device storage on the hot path.
        ensure_segmented_workspace(
            static_cast<int>(maximum_batch), plan_.attention_chunks());
        decode_pipeline_ = std::make_unique<PackedDecodePipeline>(*this);
        prefill_pipeline_ = std::make_unique<PackedPrefillPipeline>(*this);
    }

    PackedLayerProgram layer_program_;
    CudaExecutionPlan plan_;
    PackedBatchValidator validator_;
    PackedDecodeMetrics metric;
    PackedGemmRuntime gemm_;
    PackedMetadataCache metadata_cache_;
    std::unique_ptr<PackedDecodePipeline> decode_pipeline_;
    std::unique_ptr<PackedPrefillPipeline> prefill_pipeline_;

    void ensure_gemm_dispatcher(const CudaModelOptions& options) {
        gemm_.ensure(stream.get(), options);
    }

    static bool options_compatible(const PackedSessionContext& left,
                                   const PackedSessionContext& right,
                                   std::string* reason) {
        if (!(left.compatibility_key == right.compatibility_key)) {
            if (reason) *reason = "sessions use incompatible packed execution keys";
            return false;
        }
        return true;
    }

    PackedEligibility validate_session(const PackedSessionContext& model,
                                       PackedOperation operation) const {
        return validator_.validate_session(
            model, operation, PackedExecutorCapabilities{paged_kv != nullptr});
    }

    void linear(const __nv_bfloat16* x,
                const LinearWeight& weight,
                __nv_bfloat16* y,
                int m,
                int n,
                int k_width,
                float beta = 0.0f) {
        if (weight.rows != n || weight.cols != k_width) {
            throw std::runtime_error("packed linear shape mismatch");
        }
        gemm_.dispatcher().linear(x, weight, y, m, n, k_width, beta, plan_);
    }

    bool session_layout_changed(
        const std::vector<PackedSessionContext>& models) const {
        return metadata_cache_.changed(models);
    }

    void copy_persistent_metadata(
        const std::vector<PackedSessionContext>& models) {
        stage_packed_persistent_metadata(*this, models, shape_);
    }

    void copy_step_metadata(const std::vector<PackedSessionContext>& models,
                            const std::vector<std::vector<uint32_t>>* page_tables) {
        stage_packed_step_metadata(*this, models, page_tables);
    }


    const PackedSessionContext& validate_decode_batch(
        const std::vector<PackedSessionContext>& models) const {
        const PackedEligibility common = validate_packed_batch_common(
            models, maximum_batch, PackedOperation::Decode);
        if (!common) throw std::invalid_argument(common.reason);
        const PackedSessionContext& reference = models.front();
        if (reference.compatibility_key.execution_plan_fingerprint != plan_.fingerprint() ||
            reference.compatibility_key.device_ordinal != plan_.device().device_ordinal) {
            throw std::invalid_argument(
                "packed session plan does not match executor device policy");
        }
        PackedEligibility eligibility =
            validate_session(reference, PackedOperation::Decode);
        if (!eligibility) throw std::invalid_argument(eligibility.reason);
        for (size_t row = 1; row < models.size(); ++row) {
            for (size_t previous = 0; previous < row; ++previous) {
                if (models[previous].owner == models[row].owner) {
                    throw std::invalid_argument("duplicate session in packed batch");
                }
            }
            eligibility = validate_session(models[row], PackedOperation::Decode);
            if (!eligibility ||
                !options_compatible(reference, models[row], &eligibility.reason)) {
                throw std::invalid_argument(eligibility.reason);
            }
        }
        return reference;
    }

        const PackedSessionContext& validate_prefill_batch(
        const std::vector<PackedSessionContext>& models,
        const std::vector<int32_t>& explicit_tokens,
        const std::vector<PackedPrefillRow>& rows) const {
        const PackedEligibility common = validate_packed_batch_common(
            models, maximum_batch, PackedOperation::Prefill);
        if (!common) throw std::invalid_argument(common.reason);
        if (rows.size() != models.size()) {
            throw std::invalid_argument("ragged prefill needs one row descriptor per row");
        }
        size_t consumed = 0;
        for (const PackedPrefillRow& row : rows) {
            if (row.token_offset != consumed) {
                throw std::invalid_argument("ragged prefill rows must be densely packed");
            }
            consumed += row.token_count;
        }
        if (consumed != explicit_tokens.size()) {
            throw std::invalid_argument("ragged prefill token buffer length mismatch");
        }
        if (models.front().owner == nullptr) {
            throw std::invalid_argument("null packed session");
        }
        const PackedSessionContext& reference = models.front();
        if (reference.compatibility_key.execution_plan_fingerprint != plan_.fingerprint() ||
            reference.compatibility_key.device_ordinal != plan_.device().device_ordinal) {
            throw std::invalid_argument(
                "packed session plan does not match executor device policy");
        }
        PackedEligibility eligibility =
            validate_session(reference, PackedOperation::Prefill);
        if (!eligibility) throw std::invalid_argument(eligibility.reason);
        if (!paged_kv) {
            throw std::invalid_argument("ragged packed prefill requires physical paged KV");
        }
        for (size_t row = 1; row < models.size(); ++row) {
            if (models[row].owner == nullptr) {
                throw std::invalid_argument("null packed session");
            }
            for (size_t previous = 0; previous < row; ++previous) {
                if (models[previous].owner == models[row].owner) {
                    throw std::invalid_argument("duplicate session in packed batch");
                }
            }
            eligibility = validate_session(models[row], PackedOperation::Prefill);
            if (!eligibility ||
                !options_compatible(reference, models[row], &eligibility.reason)) {
                throw std::invalid_argument(eligibility.reason);
            }
        }
        return reference;
    }

    struct AttentionBatchPlan {
        bool segmented = false;
        int chunks = 0;
    };

    AttentionBatchPlan prepare_batch_metadata(
        const std::vector<PackedSessionContext>& models,
        const std::vector<std::vector<uint32_t>>* page_tables) {
        if (session_layout_changed(models)) {
            copy_persistent_metadata(models);
            metadata_cache_.update(models);
        }
        copy_step_metadata(models, page_tables);
        int maximum_position = 0;
        AttentionBatchPlan plan;
        for (size_t row = 0; row < models.size(); ++row) {
            maximum_position = std::max(maximum_position,
                                        h_positions.data()[row]);
            plan.segmented = plan.segmented ||
                models[row].use_segmented_attention(
                    h_positions.data()[row]);
        }
        if (plan.segmented) {
            const int chunk_tokens =
                models.front().options().attention_chunk_tokens;
            plan.chunks = (maximum_position + 1 + chunk_tokens - 1) /
                          chunk_tokens;
            ensure_segmented_workspace(static_cast<int>(models.size()),
                                       plan.chunks);
        }
        return plan;
    }

    AttentionBatchPlan prepare_flat_prefill_metadata(
        const std::vector<PackedSessionContext>& models,
        const std::vector<std::vector<uint32_t>>& page_tables,
        const std::vector<PackedPrefillRow>& descriptors) {
        const size_t tokens = [&] {
            size_t value = 0;
            for (const PackedPrefillRow& row : descriptors) value += row.token_count;
            return value;
        }();
        const int page_stride = paged_kv->max_pages_per_request();
        std::fill_n(h_page_tables.data(), tokens * static_cast<size_t>(page_stride),
                    std::numeric_limits<uint32_t>::max());
        size_t flat = 0;
        int maximum_position = 0;
        for (size_t request = 0; request < models.size(); ++request) {
            const int start = models[request].position();
            const auto& pages = page_tables[request];
            for (size_t token = 0; token < descriptors[request].token_count; ++token, ++flat) {
                h_positions.data()[flat] = start + static_cast<int>(token);
                maximum_position = std::max(maximum_position, h_positions.data()[flat]);
                std::copy(pages.begin(), pages.end(),
                          h_page_tables.data() + flat * static_cast<size_t>(page_stride));
            }
        }
        CELEG_CUDA(cudaMemcpyAsync(positions.data(), h_positions.data(),
                                 tokens * sizeof(int32_t), cudaMemcpyHostToDevice,
                                 stream.get()));
        CELEG_CUDA(cudaMemcpyAsync(d_page_tables.data(), h_page_tables.data(),
                                 tokens * static_cast<size_t>(page_stride) * sizeof(uint32_t),
                                 cudaMemcpyHostToDevice, stream.get()));
        AttentionBatchPlan plan;
        for (size_t request = 0; request < models.size(); ++request) {
            plan.segmented = plan.segmented || models[request].use_segmented_attention(
                models[request].position() + static_cast<int>(descriptors[request].token_count) - 1);
        }
        if (plan.segmented) {
            const int chunk_tokens = models.front().options().attention_chunk_tokens;
            plan.chunks = (maximum_position + 1 + chunk_tokens - 1) / chunk_tokens;
            ensure_segmented_workspace(static_cast<int>(tokens), plan.chunks);
        }
        return plan;
    }

    void launch_embedding_rows(const PackedSessionContext& reference, int rows) {
        if (reference.embedding()->int4_quantized()) {
            launch_embedding_int4_batch(
                sampled.data(), rows, reference.embedding()->int4,
                reference.embedding()->scales, hidden.data(),
                shape_.hidden, stream.get());
        } else if (reference.embedding()->int8_quantized()) {
            launch_embedding_int8_batch(
                sampled.data(), rows, reference.embedding()->int8,
                reference.embedding()->scales, hidden.data(),
                shape_.hidden, stream.get());
        } else {
            launch_embedding_batch(
                sampled.data(), rows, reference.embedding()->bf16,
                hidden.data(), shape_.hidden, stream.get());
        }
        launch_scale(hidden.data(), rows * shape_.hidden,
                shape_.numerical_policy.embedding_multiplier, stream.get());
    }

    void run_attention_layer(const PackedSessionContext& reference,
                             const AttentionLayer& attention,
                             int rows, int layer_index,
                             bool segmented_attention,
                             int segmented_chunks) {
        PackedOperatorContext context{*this, gemm_.dispatcher(), plan_, shape_};
        PackedAttentionExecutor::run(context, reference, attention, rows,
                                     layer_index, segmented_attention,
                                     segmented_chunks);
    }

    void run_convolution_layer(
        const PackedSessionContext& reference,
        const ConvolutionLayer& convolution,
        int rows, int layer_index, int ragged_requests = 0) {
        PackedOperatorContext context{*this, gemm_.dispatcher(), plan_, shape_};
        PackedConvolutionExecutor::run(context, reference, convolution,
                                       rows, layer_index, ragged_requests);
    }

    void run_mlp_layer(const PackedSessionContext& reference,
                       const LayerCommon& common_layer,
                       int rows,
                       const std::vector<PackedSessionContext>* batch_models = nullptr,
                       int layer_index = -1) {
        if (const MoeFfnWeights* moe = as_moe_ffn(common_layer.feed_forward)) {
            (void)moe;
            PackedOperatorContext context{*this, gemm_.dispatcher(), plan_, shape_};
            PackedMoeExecutor::run(context, reference, common_layer, rows,
                                   batch_models, layer_index);
            return;
        }
        PackedOperatorContext context{*this, gemm_.dispatcher(), plan_, shape_};
        PackedDenseFfnExecutor::run(context, reference, common_layer,
                                    rows, layer_index);
    }

    void run_transformer_layers(const PackedSessionContext& reference, int rows,
                                bool segmented_attention,
                                int segmented_chunks,
                                const std::vector<PackedSessionContext>* batch_models = nullptr,
                                int ragged_requests = 0,
                                const std::vector<PackedPrefillRow>* row_descriptors = nullptr) {
        for (size_t layer_index = 0; layer_index < layer_program_.size();
             ++layer_index) {
            const auto& layer = reference.layers()[layer_index];
            const auto& common_layer = common(layer);
            if (!reference.options().fused_residuals) {
                CELEG_CUDA(cudaMemcpyAsync(
                    residual.data(), hidden.data(),
                    static_cast<size_t>(rows) * shape_.hidden *
                        sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream.get()));
            }
            launch_rmsnorm(hidden.data(), common_layer.operator_norm,
                           normed.data(), rows, shape_.hidden,
                           shape_.numerical_policy.norm_eps, stream.get());
            PackedOperatorContext context{*this, gemm_.dispatcher(), plan_, shape_};
            if (layer_program_.kind(layer_index) == PackedLayerKind::Mamba2 ||
                layer_program_.kind(layer_index) == PackedLayerKind::MlpOnly) {
                throw std::runtime_error(
                    "packed CUDA execution for Nemotron-H requires tokenwise scheduling");
            }
            if (layer_program_.kind(layer_index) == PackedLayerKind::Attention) {
                const auto* attention = as_attention(layer);
                if (!attention) {
                    throw std::logic_error("packed layer program disagrees with layer binding");
                }
                run_attention_layer(reference, *attention, rows, layer_index,
                                    segmented_attention, segmented_chunks);
            } else if (layer_program_.kind(layer_index) == PackedLayerKind::GatedDeltaNet) {
                const auto* gated_delta = as_gated_delta_net(layer);
                if (!gated_delta) {
                    throw std::logic_error("packed layer program disagrees with GatedDeltaNet binding");
                }
                PackedGatedDeltaNetExecutor::run(
                    context, reference,
                    *gated_delta, rows, static_cast<int>(layer_index),
                    batch_models, row_descriptors);
            } else {
                const auto* convolution = as_convolution(layer);
                if (!convolution) {
                    throw std::logic_error("packed layer program disagrees with layer binding");
                }
                run_convolution_layer(reference, *convolution, rows,
                                     static_cast<int>(layer_index), ragged_requests);
            }
            if (!reference.options().fused_residuals) {
                launch_residual_add(hidden.data(), residual.data(),
                                    rows * shape_.hidden, stream.get());
            }
            run_mlp_layer(reference, common_layer, rows, batch_models, layer_index);
        }
    }

    void decode_into(
        const std::vector<PackedSessionContext>& models,
        const std::vector<std::vector<uint32_t>>* page_tables,
        std::span<int32_t> output,
        const PackedSessionContext* validated_reference = nullptr) {
        if (models.empty()) return;
        const PackedSessionContext& reference = validated_reference
            ? *validated_reference : validate_decode_batch(models);
        const int rows = static_cast<int>(models.size());
        if (output.size() < models.size()) {
            throw std::invalid_argument("packed decode output buffer is too small");
        }
        ensure_gemm_dispatcher(reference.options());
        const auto started = std::chrono::steady_clock::now();
        CudaEvent gpu_begin;
        CudaEvent gpu_end;
        const AttentionBatchPlan attention =
            prepare_batch_metadata(models, page_tables);
        gpu_begin.record(stream.get());
        const auto host_prepare_done = std::chrono::steady_clock::now();

        launch_packed_sample_topk(
            d_logits.data(), d_seen.data(), d_rng.data(),
            temperatures.data(), repetition_penalties.data(),
            top_k.data(), top_p.data(), sampling_scores.data(),
            selected_values.data(), selected_indices.data(), rows,
            shape_.vocab_size, sampled.data(), stream.get());

        launch_embedding_rows(reference, rows);

        run_transformer_layers(reference, rows, attention.segmented,
                               attention.chunks, &models);
        launch_rmsnorm(hidden.data(), reference.final_norm(), normed.data(),
                               rows, shape_.hidden, shape_.numerical_policy.norm_eps,
                       stream.get());
        linear(normed.data(), *reference.logits_weight(), logits.data(), rows,
               shape_.vocab_size, shape_.hidden);
        launch_scale(logits.data(), rows * shape_.vocab_size,
                     1.0f / shape_.numerical_policy.logits_divisor, stream.get());
        if (shape_.numerical_policy.final_logit_softcap > 0.0f) {
            launch_tanh_softcap(logits.data(), rows * shape_.vocab_size,
                                shape_.numerical_policy.final_logit_softcap, stream.get());
        }
        launch_scatter_bf16_rows(
            logits.data(), d_logits.data(), rows, shape_.vocab_size,
            stream.get());
        launch_scatter_decode_state(
            sampled.data(), positions.data(), d_sampled_dest.data(),
            d_position_dest.data(), rows, stream.get());
        CELEG_CUDA(cudaMemcpyAsync(sampled_host.data(), sampled.data(),
                                 static_cast<size_t>(rows) * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost, stream.get()));
        CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        gpu_end.record(stream.get());
        gpu_end.synchronize();

        const auto gpu_done = std::chrono::steady_clock::now();
        const auto commit_started = gpu_done;
        const double gpu_elapsed_ms = CudaEvent::elapsed_ms(gpu_begin, gpu_end);
        commit_packed_decode(models,
                             std::span<const int32_t>(sampled_host.data(), rows),
                             gpu_elapsed_ms, output);
        const auto commit_done = std::chrono::steady_clock::now();
        const auto ended = commit_done;
        metric.last_decode_timing.host_prepare_ms =
            std::chrono::duration<double, std::milli>(host_prepare_done - started).count();
        metric.last_decode_timing.gpu_execute_ms = gpu_elapsed_ms;
        metric.last_decode_timing.host_commit_ms =
            std::chrono::duration<double, std::milli>(commit_done - commit_started).count();
        metric.last_decode_timing.end_to_end_ms =
            std::chrono::duration<double, std::milli>(ended - started).count();
        ++metric.steps;
        metric.tokens += static_cast<uint64_t>(rows);
        if (attention.segmented && paged_kv) {
            ++metric.segmented_paged_steps;
            metric.segmented_paged_tokens += static_cast<uint64_t>(rows);
        }
        metric.maximum_batch = std::max(metric.maximum_batch,
                                        static_cast<size_t>(rows));
        metric.cumulative_ms += gpu_elapsed_ms;
    }

    std::vector<int32_t> decode(
        const std::vector<PackedSessionContext>& models,
        const std::vector<std::vector<uint32_t>>* page_tables) {
        std::vector<int32_t> result(models.size());
        decode_into(models, page_tables, result);
        return result;
    }

    void prefill(const std::vector<PackedSessionContext>& models,
                 const std::vector<std::vector<uint32_t>>* page_tables,
                 const std::vector<int32_t>& explicit_tokens,
                 const std::vector<PackedPrefillRow>& row_descriptors,
                 const PackedSessionContext* validated_reference = nullptr) {
        if (models.empty()) return;
        const PackedSessionContext& reference = validated_reference
            ? *validated_reference
            : validate_prefill_batch(models, explicit_tokens, row_descriptors);
        if (!page_tables || page_tables->size() != models.size()) {
            throw std::invalid_argument("ragged packed prefill needs one page table per request");
        }
        const int requests = static_cast<int>(models.size());
        const int rows = static_cast<int>(explicit_tokens.size());
        if (static_cast<size_t>(rows) > maximum_prefill_token_capacity) {
            throw std::invalid_argument("ragged prefill exceeds flattened token capacity");
        }
        const int page_stride = paged_kv->max_pages_per_request();
        for (size_t request = 0; request < models.size(); ++request) {
            const PackedPrefillRow& descriptor = row_descriptors[request];
            if (descriptor.token_count == 0) {
                throw std::invalid_argument("ragged prefill rows must be non-empty");
            }
            const size_t end = static_cast<size_t>(models[request].position()) +
                descriptor.token_count;
            if (end > static_cast<size_t>(models[request].max_context())) {
                throw std::invalid_argument("ragged prefill exceeds context limit");
            }
            const size_t pages_needed =
                (end + static_cast<size_t>(paged_kv->page_tokens()) - 1) /
                static_cast<size_t>(paged_kv->page_tokens());
        if (page_tables->at(request).size() < pages_needed ||
                page_tables->at(request).size() > static_cast<size_t>(page_stride)) {
                throw std::invalid_argument("ragged prefill page table has invalid length");
            }
        }
        for (const int32_t token : explicit_tokens) {
            if (token < 0 || token >= shape_.vocab_size) {
                throw std::invalid_argument("CUDA token id is outside the vocabulary");
            }
        }
        ensure_gemm_dispatcher(reference.options());
        const auto started = std::chrono::steady_clock::now();
        CudaEvent gpu_begin;
        CudaEvent gpu_end;
        copy_persistent_metadata(models);
        metadata_cache_.update(models);
        for (int request = 0; request < requests; ++request) {
            h_span_offsets.data()[request] = static_cast<int>(row_descriptors[request].token_offset);
            h_span_counts.data()[request] = static_cast<int>(row_descriptors[request].token_count);
            h_final_rows.data()[request] = h_span_offsets.data()[request] + h_span_counts.data()[request] - 1;
        }
        CELEG_CUDA(cudaMemcpyAsync(d_span_offsets.data(), h_span_offsets.data(),
                                 requests * sizeof(int32_t), cudaMemcpyHostToDevice, stream.get()));
        CELEG_CUDA(cudaMemcpyAsync(d_span_counts.data(), h_span_counts.data(),
                                 requests * sizeof(int32_t), cudaMemcpyHostToDevice, stream.get()));
        CELEG_CUDA(cudaMemcpyAsync(d_final_rows.data(), h_final_rows.data(),
                                 requests * sizeof(int32_t), cudaMemcpyHostToDevice, stream.get()));
        const AttentionBatchPlan attention =
            prepare_flat_prefill_metadata(models, *page_tables, row_descriptors);
        CELEG_CUDA(cudaMemcpyAsync(sampled.data(), explicit_tokens.data(),
                                 static_cast<size_t>(rows) * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, stream.get()));
        // Reuse the request pointer table by expanding only the seen-token
        // pointers for this launch: repeated writes set the same byte to one.
        for (int request = 0; request < requests; ++request) {
            std::fill_n(h_flat_seen.data() + h_span_offsets.data()[request],
                        h_span_counts.data()[request], h_seen.data()[request]);
        }
        CELEG_CUDA(cudaMemcpyAsync(d_flat_seen.data(), h_flat_seen.data(),
                                 static_cast<size_t>(rows) * sizeof(uint8_t*),
                                 cudaMemcpyHostToDevice, stream.get()));
        gpu_begin.record(stream.get());
        const auto host_prepare_done = std::chrono::steady_clock::now();
        launch_mark_seen_batch_ptrs(sampled.data(), d_flat_seen.data(), rows,
                                    shape_.vocab_size, stream.get());
        launch_embedding_rows(reference, rows);
        run_transformer_layers(reference, rows, attention.segmented,
                               attention.chunks, &models, requests,
                               &row_descriptors);
        const bool any_finalize = std::any_of(row_descriptors.begin(), row_descriptors.end(),
            [](const PackedPrefillRow& row) { return row.finalize != 0; });
        if (any_finalize) {
            int finalized = 0;
            for (size_t request = 0; request < row_descriptors.size(); ++request) {
                const auto& descriptor = row_descriptors[request];
                if (descriptor.finalize != 0) {
                    h_selected_final_rows.data()[finalized++] = h_final_rows.data()[request];
                    h_selected_logits.data()[finalized - 1] = h_logits.data()[request];
                }
            }
            CELEG_CUDA(cudaMemcpyAsync(d_selected_final_rows.data(), h_selected_final_rows.data(),
                                       static_cast<size_t>(finalized) * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream.get()));
            CELEG_CUDA(cudaMemcpyAsync(d_selected_logits.data(), h_selected_logits.data(),
                                       static_cast<size_t>(finalized) * sizeof(__nv_bfloat16*),
                                       cudaMemcpyHostToDevice, stream.get()));
            launch_gather_bf16_rows(hidden.data(), d_selected_final_rows.data(),
                                    normed.data(), finalized, shape_.hidden, stream.get());
            launch_rmsnorm(normed.data(), reference.final_norm(), normed.data(),
                           finalized, shape_.hidden, shape_.numerical_policy.norm_eps, stream.get());
            linear(normed.data(), *reference.logits_weight(), logits.data(), finalized,
                   shape_.vocab_size, shape_.hidden);
            launch_scale(logits.data(), finalized * shape_.vocab_size,
                   1.0f / shape_.numerical_policy.logits_divisor, stream.get());
            if (shape_.numerical_policy.final_logit_softcap > 0.0f) {
                launch_tanh_softcap(logits.data(), finalized * shape_.vocab_size,
                                    shape_.numerical_policy.final_logit_softcap, stream.get());
            }
            launch_scatter_bf16_selected_rows(logits.data(), d_selected_final_rows.data(),
                                               d_selected_logits.data(), finalized,
                                               shape_.vocab_size, stream.get());
        }
        launch_scatter_selected_decode_state(
            sampled.data(), positions.data(), d_final_rows.data(),
            d_sampled_dest.data(), d_position_dest.data(), requests, stream.get());
        metric.ragged_prefill_tokens += static_cast<uint64_t>(rows);
        metric.maximum_prefill_batch = std::max(metric.maximum_prefill_batch,
                                                static_cast<size_t>(requests));
        ++metric.ragged_prefill_transformer_passes;

        gpu_end.record(stream.get());
        gpu_end.synchronize();

        const auto gpu_done = std::chrono::steady_clock::now();
        const auto commit_started = gpu_done;
        // Host-visible session state is committed only after the completion
        // event succeeds. A launch failure therefore cannot leave a lane
        // advanced while its device state is incomplete.
        commit_packed_prefill(models,
                              explicit_tokens,
                              std::span<const int32_t>(h_final_rows.data(), requests),
                              row_descriptors);
        const auto commit_done = std::chrono::steady_clock::now();

        const auto ended = std::chrono::steady_clock::now();
        const double gpu_elapsed_ms = CudaEvent::elapsed_ms(gpu_begin, gpu_end);
        const double per_token_ms = explicit_tokens.empty()
            ? 0.0
            : gpu_elapsed_ms / static_cast<double>(explicit_tokens.size());
        for (int row = 0; row < requests; ++row) {
            const PackedSessionContext& model = models[static_cast<size_t>(row)];
            model.metrics().last_prefill_ms +=
                per_token_ms * static_cast<double>(row_descriptors[static_cast<size_t>(row)].token_count);
        }
        ++metric.ragged_prefill_steps;
        metric.cumulative_prefill_ms += gpu_elapsed_ms;
        metric.last_prefill_timing.host_prepare_ms =
            std::chrono::duration<double, std::milli>(host_prepare_done - started).count();
        metric.last_prefill_timing.gpu_execute_ms = gpu_elapsed_ms;
        metric.last_prefill_timing.host_commit_ms =
            std::chrono::duration<double, std::milli>(commit_done - commit_started).count();
        metric.last_prefill_timing.end_to_end_ms =
            std::chrono::duration<double, std::milli>(ended - started).count();
    }
};

PackedDecodePipeline::PackedDecodePipeline(PackedDecodeExecutorImpl& state)
    : state_(&state) {}

void PackedDecodePipeline::run(
    const std::vector<PackedSessionContext>& sessions,
    const std::vector<std::vector<uint32_t>>* page_tables,
    std::span<int32_t> output) {
    if (sessions.empty()) return;
    const PackedSessionContext& reference = state_->validate_decode_batch(sessions);
    if (output.size() < sessions.size()) {
        throw std::invalid_argument("packed decode output buffer is too small");
    }
    state_->decode_into(sessions, page_tables, output, &reference);
}

PackedPrefillPipeline::PackedPrefillPipeline(PackedDecodeExecutorImpl& state)
    : state_(&state) {}

void PackedPrefillPipeline::run(
    const std::vector<PackedSessionContext>& sessions,
    const std::vector<std::vector<uint32_t>>* page_tables,
    const std::vector<int32_t>& tokens,
    const std::vector<PackedPrefillRow>& rows) {
    if (sessions.empty()) return;
    const PackedSessionContext& reference =
        state_->validate_prefill_batch(sessions, tokens, rows);
    (void)reference;
    state_->prefill(sessions, page_tables, tokens, rows, &reference);
}

PackedDecodeExecutor::PackedDecodeExecutor(size_t maximum_sessions,
                                           size_t maximum_prefill_tokens,
                                           PhysicalPagedKvCache* paged_kv,
                                           const RuntimeTopology& shape,
                                           CudaExecutionPlan plan)
    : state_(std::make_unique<PackedDecodeExecutorImpl>(
          maximum_sessions, maximum_prefill_tokens, paged_kv, shape, std::move(plan))) {}

PackedDecodeExecutor::~PackedDecodeExecutor() = default;

PackedEligibility PackedDecodeExecutor::validate_session(
    const PackedSessionContext& model, PackedOperation operation) const {
    return state_->validate_session(model, operation);
}

std::vector<int32_t> PackedDecodeExecutor::decode(
    const std::vector<PackedSessionContext>& models) {
    return state_->decode(models, nullptr);
}

std::vector<int32_t> PackedDecodeExecutor::decode(
    const std::vector<PackedSessionContext>& models,
    const std::vector<std::vector<uint32_t>>& page_tables) {
    return state_->decode(models, &page_tables);
}

void PackedDecodeExecutor::decode_into(
    const std::vector<PackedSessionContext>& models,
    std::span<int32_t> output) {
    state_->decode_pipeline_->run(models, nullptr, output);
}

void PackedDecodeExecutor::decode_into(
    const std::vector<PackedSessionContext>& models,
    const std::vector<std::vector<uint32_t>>& page_tables,
    std::span<int32_t> output) {
    state_->decode_pipeline_->run(models, &page_tables, output);
}

void PackedDecodeExecutor::prefill(
    const std::vector<PackedSessionContext>& models,
    const std::vector<std::vector<uint32_t>>& page_tables,
    const std::vector<int32_t>& tokens,
    const std::vector<PackedPrefillRow>& rows) {
    state_->prefill_pipeline_->run(models, &page_tables, tokens, rows);
}

size_t PackedDecodeExecutor::maximum_batch() const {
    return state_->maximum_batch;
}

size_t PackedDecodeExecutor::maximum_prefill_tokens() const {
    return state_->maximum_prefill_token_capacity;
}

PackedDecodeMetrics PackedDecodeExecutor::metrics() const {
    return state_->metric;
}

} // namespace celeg
