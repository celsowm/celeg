#include "lfm/backend/cuda/packed.hpp"

#include "lfm/backend/cuda/kernels/kernels.cuh"
#include "lfm/runtime/moe.hpp"
#include "lfm/detail/model/impl.hpp"
#include "lfm/backend/cuda/paged_kv.hpp"
#include "lfm/backend/cuda/gemm_dispatcher.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <optional>

namespace lfm {

struct PackedDecodeExecutorImpl {
    explicit PackedDecodeExecutorImpl(size_t maximum_batch_value,
                                      size_t maximum_prefill_tokens_value,
                                      PhysicalPagedKvCache* paged_kv_value,
                                      const ModelShape& shape)
        : maximum_batch(maximum_batch_value),
          maximum_prefill_token_capacity(maximum_prefill_tokens_value),
          paged_kv(paged_kv_value),
          shape_(shape),
          stream(),
          cublas(stream.get()),
          positions(maximum_prefill_token_capacity),
          sampled(maximum_prefill_token_capacity),
          sampled_host(maximum_batch),
          temperatures(maximum_batch),
          repetition_penalties(maximum_batch),
          top_k(maximum_batch),
          top_p(maximum_batch),
          hidden(maximum_prefill_token_capacity * shape_.hidden),
          residual(maximum_prefill_token_capacity * shape_.hidden),
          normed(maximum_prefill_token_capacity * shape_.hidden),
          op_output(maximum_prefill_token_capacity * shape_.hidden),
          qkv_output(maximum_prefill_token_capacity * shape_.qkv_width),
          q(maximum_prefill_token_capacity * shape_.q_width),
          k(maximum_prefill_token_capacity * shape_.kv_width),
          v(maximum_prefill_token_capacity * shape_.kv_width),
          conv_projected(maximum_prefill_token_capacity * 3 * shape_.hidden),
          gate_up(maximum_prefill_token_capacity * 2 * shape_.intermediate),
          activated(maximum_prefill_token_capacity * shape_.intermediate),
          mlp_output(maximum_prefill_token_capacity * shape_.hidden),
          moe_hidden_float(maximum_prefill_token_capacity * shape_.hidden),
          moe_sel(maximum_prefill_token_capacity * shape_.experts_per_token),
          moe_routing_w(maximum_prefill_token_capacity * shape_.experts_per_token),
          moe_router_scratch(maximum_prefill_token_capacity * shape_.num_experts),
          moe_output(maximum_prefill_token_capacity * shape_.hidden),
          moe_output_accum(maximum_prefill_token_capacity * shape_.hidden),
          moe_gu_scratch(static_cast<size_t>(maximum_prefill_token_capacity) *
                         shape_.experts_per_token * 2 * shape_.moe_intermediate),
          moe_act_scratch(static_cast<size_t>(maximum_prefill_token_capacity) *
                          shape_.experts_per_token * shape_.moe_intermediate),
          logits(maximum_prefill_token_capacity * shape_.vocab_size),
          sampling_scores(maximum_batch * shape_.vocab_size),
          selected_values(maximum_batch * static_cast<size_t>(kMaxTopK)),
          selected_indices(maximum_batch * static_cast<size_t>(kMaxTopK)),
          h_positions(maximum_prefill_token_capacity),
          h_temperatures(maximum_batch),
          h_repetition_penalties(maximum_batch),
          h_top_k(maximum_batch),
          h_top_p(maximum_batch),
          h_logits(maximum_batch),
          h_seen(maximum_batch),
          h_rng(maximum_batch),
          h_sampled_dest(maximum_batch),
          h_position_dest(maximum_batch),
          d_logits(maximum_batch),
          d_seen(maximum_batch),
          d_rng(maximum_batch),
          d_sampled_dest(maximum_batch),
          d_position_dest(maximum_batch),
          h_key_bf16(maximum_batch * shape_.num_hidden_layers),
          h_value_bf16(maximum_batch * shape_.num_hidden_layers),
          h_key_int8(maximum_batch * shape_.num_hidden_layers),
          h_value_int8(maximum_batch * shape_.num_hidden_layers),
          h_key_scales(maximum_batch * shape_.num_hidden_layers),
          h_value_scales(maximum_batch * shape_.num_hidden_layers),
          h_conv_states(maximum_batch * shape_.num_hidden_layers),
          d_key_bf16(maximum_batch * shape_.num_hidden_layers),
          d_value_bf16(maximum_batch * shape_.num_hidden_layers),
          d_key_int8(maximum_batch * shape_.num_hidden_layers),
          d_value_int8(maximum_batch * shape_.num_hidden_layers),
          d_key_scales(maximum_batch * shape_.num_hidden_layers),
          d_value_scales(maximum_batch * shape_.num_hidden_layers),
          d_conv_states(maximum_batch * shape_.num_hidden_layers),
          h_page_tables(paged_kv_value ? maximum_prefill_token_capacity * static_cast<size_t>(paged_kv_value->max_pages_per_request()) : 0),
          d_page_tables(paged_kv_value ? maximum_prefill_token_capacity * static_cast<size_t>(paged_kv_value->max_pages_per_request()) : 0),
          h_span_offsets(maximum_batch),
          h_span_counts(maximum_batch),
          h_final_rows(maximum_batch),
          d_span_offsets(maximum_batch),
          d_span_counts(maximum_batch),
          d_final_rows(maximum_batch) {
        if (maximum_batch == 0 || maximum_prefill_token_capacity == 0) {
            throw std::invalid_argument("packed capacities must be positive");
        }
        std::fill_n(h_logits.data(), maximum_batch, nullptr);
        std::fill_n(h_seen.data(), maximum_batch, nullptr);
        std::fill_n(h_rng.data(), maximum_batch, nullptr);
        std::fill_n(h_sampled_dest.data(), maximum_batch, nullptr);
        std::fill_n(h_position_dest.data(), maximum_batch, nullptr);
        const size_t layer_slots = maximum_batch * shape_.num_hidden_layers;
        std::fill_n(h_key_bf16.data(), layer_slots, nullptr);
        std::fill_n(h_value_bf16.data(), layer_slots, nullptr);
        std::fill_n(h_key_int8.data(), layer_slots, nullptr);
        std::fill_n(h_value_int8.data(), layer_slots, nullptr);
        std::fill_n(h_key_scales.data(), layer_slots, nullptr);
        std::fill_n(h_value_scales.data(), layer_slots, nullptr);
        std::fill_n(h_conv_states.data(), layer_slots, nullptr);
    }

    size_t maximum_batch;
    size_t maximum_prefill_token_capacity;
    PhysicalPagedKvCache* paged_kv = nullptr;
    ModelShape shape_;
    CudaStream stream;
    CublasHandle cublas;
    PackedDecodeMetrics metric;

    DeviceBuffer<int32_t> positions;
    DeviceBuffer<int32_t> sampled;
    PinnedBuffer<int32_t> sampled_host;
    DeviceBuffer<float> temperatures;
    DeviceBuffer<float> repetition_penalties;
    DeviceBuffer<int32_t> top_k;
    DeviceBuffer<float> top_p;

    DeviceBuffer<__nv_bfloat16> hidden;
    DeviceBuffer<__nv_bfloat16> residual;
    DeviceBuffer<__nv_bfloat16> normed;
    DeviceBuffer<__nv_bfloat16> op_output;
    DeviceBuffer<__nv_bfloat16> qkv_output;
    DeviceBuffer<__nv_bfloat16> q;
    DeviceBuffer<__nv_bfloat16> k;
    DeviceBuffer<__nv_bfloat16> v;
    DeviceBuffer<__nv_bfloat16> conv_projected;
    DeviceBuffer<__nv_bfloat16> gate_up;
    DeviceBuffer<__nv_bfloat16> activated;
    DeviceBuffer<__nv_bfloat16> mlp_output;
    DeviceBuffer<__nv_bfloat16> logits;

    // MoE scratch (reused across layers; sized for the maximum packed batch).
    // The router runs in float, so normed activations are cast to float and
    // the selected expert gate/up and activated projections are buffered here.
    DeviceBuffer<float> moe_hidden_float;       // [batch * hidden]
    DeviceBuffer<int32_t> moe_sel;              // [batch * experts_per_token]
    DeviceBuffer<float> moe_routing_w;          // [batch * experts_per_token]
    DeviceBuffer<float> moe_router_scratch;     // [batch * num_experts]
    DeviceBuffer<__nv_bfloat16> moe_output;     // [batch * hidden]
    DeviceBuffer<float> moe_output_accum;      // [batch * hidden] FP32 accumulator
    DeviceBuffer<__nv_bfloat16> moe_gu_scratch; // [batch * K * 2 * moe_inter]
    DeviceBuffer<__nv_bfloat16> moe_act_scratch; // [batch * K * moe_inter]
    DeviceBuffer<float> sampling_scores;
    DeviceBuffer<float> selected_values;
    DeviceBuffer<int32_t> selected_indices;

    PinnedBuffer<int32_t> h_positions;
    PinnedBuffer<float> h_temperatures;
    PinnedBuffer<float> h_repetition_penalties;
    PinnedBuffer<int32_t> h_top_k;
    PinnedBuffer<float> h_top_p;

    PinnedBuffer<__nv_bfloat16*> h_logits;
    PinnedBuffer<uint8_t*> h_seen;
    PinnedBuffer<uint64_t*> h_rng;
    PinnedBuffer<int32_t*> h_sampled_dest;
    PinnedBuffer<int32_t*> h_position_dest;
    DeviceBuffer<__nv_bfloat16*> d_logits;
    DeviceBuffer<uint8_t*> d_seen;
    DeviceBuffer<uint64_t*> d_rng;
    DeviceBuffer<int32_t*> d_sampled_dest;
    DeviceBuffer<int32_t*> d_position_dest;

    PinnedBuffer<__nv_bfloat16*> h_key_bf16;
    PinnedBuffer<__nv_bfloat16*> h_value_bf16;
    PinnedBuffer<int8_t*> h_key_int8;
    PinnedBuffer<int8_t*> h_value_int8;
    PinnedBuffer<float*> h_key_scales;
    PinnedBuffer<float*> h_value_scales;
    PinnedBuffer<__nv_bfloat16*> h_conv_states;
    DeviceBuffer<__nv_bfloat16*> d_key_bf16;
    DeviceBuffer<__nv_bfloat16*> d_value_bf16;
    DeviceBuffer<int8_t*> d_key_int8;
    DeviceBuffer<int8_t*> d_value_int8;
    DeviceBuffer<float*> d_key_scales;
    DeviceBuffer<float*> d_value_scales;
    DeviceBuffer<__nv_bfloat16*> d_conv_states;
    PinnedBuffer<uint32_t> h_page_tables;
    DeviceBuffer<uint32_t> d_page_tables;
    PinnedBuffer<int32_t> h_span_offsets;
    PinnedBuffer<int32_t> h_span_counts;
    PinnedBuffer<int32_t> h_final_rows;
    DeviceBuffer<int32_t> d_span_offsets;
    DeviceBuffer<int32_t> d_span_counts;
    DeviceBuffer<int32_t> d_final_rows;
    DeviceBuffer<float> segmented_partial_max;
    DeviceBuffer<float> segmented_partial_denom;
    DeviceBuffer<float> segmented_partial_accum;
    size_t segmented_scalar_capacity = 0;
    size_t segmented_accum_capacity = 0;

    std::unique_ptr<GemmDispatcher> gemm_;
    ModelOptions cached_options_;
    std::optional<ExecutionPlan> active_plan_;
    std::vector<PackedSessionContext> cached_sessions_;
    size_t cached_rows_ = 0;

    void ensure_gemm_dispatcher(const ModelOptions& options) {
        if (!gemm_ || cached_options_.gemm_backend != options.gemm_backend ||
            cached_options_.lt_workspace_bytes != options.lt_workspace_bytes ||
            cached_options_.lt_heuristics != options.lt_heuristics ||
            cached_options_.lt_autotune != options.lt_autotune) {
            cached_options_ = options;
            gemm_ = std::make_unique<GemmDispatcher>(stream.get(), cached_options_);
        }
    }

    void ensure_segmented_workspace(int rows, int chunks) {
        const size_t scalar_count = static_cast<size_t>(rows) *
            shape_.num_attention_heads * static_cast<size_t>(chunks);
        const size_t accum_count = scalar_count * shape_.head_dim;
        if (scalar_count > segmented_scalar_capacity) {
            segmented_partial_max.reset(scalar_count);
            segmented_partial_denom.reset(scalar_count);
            segmented_scalar_capacity = scalar_count;
        }
        if (accum_count > segmented_accum_capacity) {
            segmented_partial_accum.reset(accum_count);
            segmented_accum_capacity = accum_count;
        }
    }

    static bool options_compatible(const PackedSessionContext& left,
                                   const PackedSessionContext& right,
                                   std::string* reason) {
        const ModelOptions& a = left.options();
        const ModelOptions& b = right.options();
        if (left.weights().get() != right.weights().get()) {
            if (reason) *reason = "sessions do not share the same device weights";
            return false;
        }
        if (left.max_context() != right.max_context() ||
            a.weight_mode != b.weight_mode ||
            a.kv_cache_mode != b.kv_cache_mode ||
            a.fast_attention != b.fast_attention ||
            a.fused_projections != b.fused_projections ||
            a.fused_residuals != b.fused_residuals) {
            if (reason) *reason = "sessions use incompatible model options";
            return false;
        }
        return true;
    }

    bool eligible(const PackedSessionContext& model, std::string* reason) const {
        if (model.phase() != SessionPhase::Ready) {
            if (reason) *reason = "session has not completed prefill";
            return false;
        }
        if (model.phase() == SessionPhase::DecodePending) {
            if (reason) *reason = "session already has a pending decode";
            return false;
        }
        if (model.position() >= model.max_context()) {
            if (reason) *reason = "session reached max_context";
            return false;
        }
        if (!paged_kv && model.use_segmented_attention(model.position())) {
            if (reason) {
                *reason = "segmented attention requires physical paged KV in packed decode";
            }
            return false;
        }
        if (!paged_kv && !model.local_kv_cache_available()) {
            if (reason) *reason = "session released its local KV cache";
            return false;
        }
        return true;
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
        if (!active_plan_) {
            throw std::runtime_error("no active execution plan set for packed linear");
        }
        gemm_->linear(x, weight, y, m, n, k_width, beta, *active_plan_);
    }

    bool session_layout_changed(
        const std::vector<PackedSessionContext>& models) const {
        if (cached_rows_ != models.size() || cached_sessions_.size() != models.size()) {
            return true;
        }
        for (size_t row = 0; row < models.size(); ++row) {
            if (cached_sessions_[row].owner != models[row].owner) return true;
        }
        return false;
    }

    void copy_persistent_metadata(
        const std::vector<PackedSessionContext>& models) {
        const size_t rows = models.size();
        cached_sessions_ = models;
        cached_rows_ = rows;
        for (size_t row = 0; row < rows; ++row) {
            const PackedSessionContext& model = models[row];
            h_logits.data()[row] = model.logits().data();
            h_seen.data()[row] = model.seen_tokens().data();
            h_rng.data()[row] = model.rng_state().data();
            h_sampled_dest.data()[row] = model.sampled_device().data();
            h_position_dest.data()[row] = model.position_device().data();
            for (int layer_index = 0; layer_index < shape_.num_hidden_layers;
                 ++layer_index) {
                const size_t index = static_cast<size_t>(layer_index) *
                                     maximum_batch + row;
                Layer& layer = model.layers()[layer_index];
                if (auto* attention = as_attention(layer)) {
                    h_key_bf16.data()[index] = attention->key_cache.data();
                    h_value_bf16.data()[index] = attention->value_cache.data();
                    h_key_int8.data()[index] = attention->key_cache_int8.data();
                    h_value_int8.data()[index] = attention->value_cache_int8.data();
                    h_key_scales.data()[index] = attention->key_cache_scales.data();
                    h_value_scales.data()[index] = attention->value_cache_scales.data();
                    h_conv_states.data()[index] = nullptr;
                } else {
                    auto* convolution = as_convolution(layer);
                    h_key_bf16.data()[index] = nullptr;
                    h_value_bf16.data()[index] = nullptr;
                    h_key_int8.data()[index] = nullptr;
                    h_value_int8.data()[index] = nullptr;
                    h_key_scales.data()[index] = nullptr;
                    h_value_scales.data()[index] = nullptr;
                    h_conv_states.data()[index] = convolution->conv_state.data();
                }
            }
        }

        const size_t pointer_count = rows * sizeof(void*);
        LFM_CUDA(cudaMemcpyAsync(d_logits.data(), h_logits.data(), pointer_count,
                                 cudaMemcpyHostToDevice, stream.get()));
        LFM_CUDA(cudaMemcpyAsync(d_seen.data(), h_seen.data(), pointer_count,
                                 cudaMemcpyHostToDevice, stream.get()));
        LFM_CUDA(cudaMemcpyAsync(d_rng.data(), h_rng.data(), pointer_count,
                                 cudaMemcpyHostToDevice, stream.get()));
        LFM_CUDA(cudaMemcpyAsync(d_sampled_dest.data(), h_sampled_dest.data(),
                                 pointer_count, cudaMemcpyHostToDevice,
                                 stream.get()));
        LFM_CUDA(cudaMemcpyAsync(d_position_dest.data(), h_position_dest.data(),
                                 pointer_count, cudaMemcpyHostToDevice,
                                 stream.get()));

        const size_t layer_pointer_count =
            static_cast<size_t>(shape_.num_hidden_layers) * maximum_batch * sizeof(void*);
        LFM_CUDA(cudaMemcpyAsync(d_key_bf16.data(), h_key_bf16.data(),
                                 layer_pointer_count, cudaMemcpyHostToDevice,
                                 stream.get()));
        LFM_CUDA(cudaMemcpyAsync(d_value_bf16.data(), h_value_bf16.data(),
                                 layer_pointer_count, cudaMemcpyHostToDevice,
                                 stream.get()));
        LFM_CUDA(cudaMemcpyAsync(d_key_int8.data(), h_key_int8.data(),
                                 layer_pointer_count, cudaMemcpyHostToDevice,
                                 stream.get()));
        LFM_CUDA(cudaMemcpyAsync(d_value_int8.data(), h_value_int8.data(),
                                 layer_pointer_count, cudaMemcpyHostToDevice,
                                 stream.get()));
        LFM_CUDA(cudaMemcpyAsync(d_key_scales.data(), h_key_scales.data(),
                                 layer_pointer_count, cudaMemcpyHostToDevice,
                                 stream.get()));
        LFM_CUDA(cudaMemcpyAsync(d_value_scales.data(), h_value_scales.data(),
                                 layer_pointer_count, cudaMemcpyHostToDevice,
                                 stream.get()));
        LFM_CUDA(cudaMemcpyAsync(d_conv_states.data(), h_conv_states.data(),
                                 layer_pointer_count, cudaMemcpyHostToDevice,
                                 stream.get()));
    }

    void copy_step_metadata(const std::vector<PackedSessionContext>& models,
                            const std::vector<std::vector<uint32_t>>* page_tables) {
        const size_t rows = models.size();
        const int page_stride = paged_kv ? paged_kv->max_pages_per_request() : 0;
        if (paged_kv) {
            if (!page_tables || page_tables->size() != rows) {
                throw std::invalid_argument("paged packed decode requires one page table per row");
            }
            std::fill_n(h_page_tables.data(), rows * static_cast<size_t>(page_stride),
                        std::numeric_limits<uint32_t>::max());
        }
        for (size_t row = 0; row < rows; ++row) {
            const PackedSessionContext& model = models[row];
            h_positions.data()[row] = model.position();
            h_temperatures.data()[row] = model.generation().temperature;
            h_repetition_penalties.data()[row] = model.generation().repetition_penalty;
            h_top_k.data()[row] = model.generation().top_k;
            h_top_p.data()[row] = model.generation().top_p;
            if (paged_kv) {
                const auto& pages = page_tables->at(row);
                const size_t needed = (static_cast<size_t>(model.position()) +
                    static_cast<size_t>(paged_kv->page_tokens())) /
                    static_cast<size_t>(paged_kv->page_tokens());
                if (pages.size() < needed || pages.size() > static_cast<size_t>(page_stride)) {
                    throw std::invalid_argument("paged packed decode page table has invalid length");
                }
                std::copy(pages.begin(), pages.end(),
                          h_page_tables.data() + row * static_cast<size_t>(page_stride));
            }
        }
        const size_t scalar_i32 = rows * sizeof(int32_t);
        const size_t scalar_f32 = rows * sizeof(float);
        LFM_CUDA(cudaMemcpyAsync(positions.data(), h_positions.data(),
                                 scalar_i32, cudaMemcpyHostToDevice,
                                 stream.get()));
        LFM_CUDA(cudaMemcpyAsync(temperatures.data(), h_temperatures.data(),
                                 scalar_f32, cudaMemcpyHostToDevice,
                                 stream.get()));
        LFM_CUDA(cudaMemcpyAsync(repetition_penalties.data(),
                                 h_repetition_penalties.data(), scalar_f32,
                                 cudaMemcpyHostToDevice, stream.get()));
        LFM_CUDA(cudaMemcpyAsync(top_k.data(), h_top_k.data(), scalar_i32,
                                 cudaMemcpyHostToDevice, stream.get()));
        LFM_CUDA(cudaMemcpyAsync(top_p.data(), h_top_p.data(), scalar_f32,
                                 cudaMemcpyHostToDevice, stream.get()));
        if (paged_kv) {
            const size_t page_bytes = rows * static_cast<size_t>(page_stride) * sizeof(uint32_t);
            LFM_CUDA(cudaMemcpyAsync(d_page_tables.data(), h_page_tables.data(),
                                     page_bytes, cudaMemcpyHostToDevice,
                                     stream.get()));
        }
    }


    const PackedSessionContext& validate_decode_batch(
        const std::vector<PackedSessionContext>& models) const {
        if (models.empty()) throw std::invalid_argument("packed batch is empty");
        if (models.size() > maximum_batch) {
            throw std::invalid_argument("packed batch exceeds executor capacity");
        }
        if (models.front().owner == nullptr) {
            throw std::invalid_argument("null packed session");
        }
        const PackedSessionContext& reference = models.front();
        std::string reason;
        if (!eligible(reference, &reason)) throw std::invalid_argument(reason);
        std::unordered_set<void*> seen;
        seen.reserve(models.size());
        seen.insert(models[0].owner);
        for (size_t row = 1; row < models.size(); ++row) {
            if (models[row].owner == nullptr) {
                throw std::invalid_argument("null packed session");
            }
            if (!seen.insert(models[row].owner).second) {
                throw std::invalid_argument("duplicate session in packed batch");
            }
            if (!eligible(models[row], &reason) ||
                !options_compatible(reference, models[row], &reason)) {
                throw std::invalid_argument(reason);
            }
        }
        return reference;
    }

    const PackedSessionContext& validate_prefill_batch(
        const std::vector<PackedSessionContext>& models,
        const std::vector<int32_t>& explicit_tokens,
        const std::vector<PackedPrefillRow>& rows) const {
        if (models.empty()) throw std::invalid_argument("packed batch is empty");
        if (rows.size() != models.size()) {
            throw std::invalid_argument("ragged prefill needs one row descriptor per row");
        }
        if (models.size() > maximum_batch) {
            throw std::invalid_argument("packed batch exceeds executor capacity");
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
        std::string reason;
        const auto eligible_prefill = [&](const PackedSessionContext& model) {
            if (model.phase() == SessionPhase::DecodePending) {
                reason = "session already has a pending decode";
                return false;
            }
            if (model.position() >= model.max_context()) {
                reason = "session reached max_context";
                return false;
            }
            if (!paged_kv) {
                reason = "ragged packed prefill requires physical paged KV";
                return false;
            }
            return true;
        };
        if (!eligible_prefill(reference)) throw std::invalid_argument(reason);
        std::unordered_set<void*> seen;
        seen.reserve(models.size());
        seen.insert(models[0].owner);
        for (size_t row = 1; row < models.size(); ++row) {
            if (models[row].owner == nullptr) {
                throw std::invalid_argument("null packed session");
            }
            if (!seen.insert(models[row].owner).second) {
                throw std::invalid_argument("duplicate session in packed batch");
            }
            if (!eligible_prefill(models[row]) ||
                !options_compatible(reference, models[row], &reason)) {
                throw std::invalid_argument(reason);
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
        LFM_CUDA(cudaMemcpyAsync(positions.data(), h_positions.data(),
                                 tokens * sizeof(int32_t), cudaMemcpyHostToDevice,
                                 stream.get()));
        LFM_CUDA(cudaMemcpyAsync(d_page_tables.data(), h_page_tables.data(),
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
                     shape_.embedding_multiplier, stream.get());
    }

    void project_attention_qkv(
        const PackedSessionContext& reference,
        const AttentionLayer& attention,
        int rows) {
        if (reference.options().fused_projections) {
            linear(normed.data(), *attention.qkv, qkv_output.data(), rows,
                   shape_.qkv_width, shape_.hidden);
            launch_split_qkv_rows(
                qkv_output.data(), q.data(), k.data(), v.data(), rows,
                shape_.q_width, shape_.kv_width, stream.get());
        } else {
            const auto q_weight = slice_rows(
                *attention.qkv, 0, shape_.q_width);
            const auto k_weight = slice_rows(
                *attention.qkv, shape_.q_width, shape_.kv_width);
            const auto v_weight = slice_rows(
                *attention.qkv, shape_.q_width + shape_.kv_width,
                shape_.kv_width);
            linear(normed.data(), q_weight, q.data(), rows,
                   shape_.q_width, shape_.hidden);
            linear(normed.data(), k_weight, k.data(), rows,
                   shape_.kv_width, shape_.hidden);
            linear(normed.data(), v_weight, v.data(), rows,
                   shape_.kv_width, shape_.hidden);
        }
        if (!shape_.query_key_norm) {
            launch_rope_batch_positions(
                q.data(), k.data(), reference.rope_cos(), reference.rope_sin(),
                positions.data(), rows, shape_.num_attention_heads,
                shape_.num_key_value_heads, shape_.head_dim, stream.get());
            const float ratio = shape_.attention_multiplier /
                (1.0f / std::sqrt(static_cast<float>(shape_.head_dim)));
            launch_scale(q.data(), rows * shape_.q_width, ratio, stream.get());
        } else {
            launch_qk_norm_rope_batch_positions(
                q.data(), k.data(), attention.q_norm, attention.k_norm,
                reference.rope_cos(), reference.rope_sin(),
                positions.data(), rows, shape_.num_attention_heads, shape_.num_key_value_heads,
                shape_.head_dim, shape_.norm_eps,
                reference.options().fast_attention, stream.get());
        }
    }

    void run_paged_attention_cache(const PackedSessionContext& reference, int rows,
                                   int layer_index,
                                   bool segmented_attention,
                                   int segmented_chunks) {
        const int slot = paged_kv ? paged_kv->attention_slot(layer_index) : -1;
        const int stride = paged_kv->max_pages_per_request();
        if (reference.options().kv_cache_mode == KvCacheMode::Int8) {
            launch_store_kv_int8_paged_batch(
                k.data(), v.data(), paged_kv->key_int8(),
                paged_kv->value_int8(), paged_kv->key_scales(),
                paged_kv->value_scales(), d_page_tables.data(), stride,
                positions.data(), rows, slot, paged_kv->page_tokens(),
                paged_kv->attention_layers(), shape_.num_key_value_heads,
                shape_.head_dim, stream.get());
            if (segmented_attention) {
                launch_gqa_decode_int8_paged_segmented_batch(
                    q.data(), paged_kv->key_int8(), paged_kv->value_int8(),
                    paged_kv->key_scales(), paged_kv->value_scales(),
                    d_page_tables.data(), stride, op_output.data(),
                    positions.data(), rows, slot, paged_kv->page_tokens(),
                    paged_kv->attention_layers(),
                    shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim,
                    reference.options().attention_chunk_tokens,
                    segmented_chunks, segmented_partial_max.data(),
                    segmented_partial_denom.data(),
                    segmented_partial_accum.data(), stream.get());
            } else {
                launch_gqa_decode_int8_paged_batch(
                    q.data(), paged_kv->key_int8(), paged_kv->value_int8(),
                    paged_kv->key_scales(), paged_kv->value_scales(),
                    d_page_tables.data(), stride, op_output.data(),
                    positions.data(), rows, slot, paged_kv->page_tokens(),
                    paged_kv->attention_layers(),
                    shape_.num_attention_heads, shape_.num_key_value_heads,
                    shape_.head_dim,
                    reference.options().fast_attention, stream.get());
            }
            return;
        }
        launch_store_kv_paged_batch(
            k.data(), v.data(), paged_kv->key_bf16(), paged_kv->value_bf16(),
            d_page_tables.data(), stride, positions.data(), rows, slot,
            paged_kv->page_tokens(), paged_kv->attention_layers(),
            shape_.num_key_value_heads, shape_.head_dim, stream.get());
        if (segmented_attention) {
            launch_gqa_decode_paged_segmented_batch(
                q.data(), paged_kv->key_bf16(), paged_kv->value_bf16(),
                d_page_tables.data(), stride, op_output.data(),
                positions.data(), rows, slot, paged_kv->page_tokens(),
                paged_kv->attention_layers(),
                shape_.num_attention_heads, shape_.num_key_value_heads,
                shape_.head_dim,
                reference.options().attention_chunk_tokens,
                segmented_chunks, segmented_partial_max.data(),
                segmented_partial_denom.data(), segmented_partial_accum.data(),
                stream.get());
        } else {
            launch_gqa_decode_paged_batch(
                q.data(), paged_kv->key_bf16(), paged_kv->value_bf16(),
                d_page_tables.data(), stride, op_output.data(),
                positions.data(), rows, slot, paged_kv->page_tokens(),
                paged_kv->attention_layers(),
                shape_.num_attention_heads, shape_.num_key_value_heads,
                shape_.head_dim,
                reference.options().fast_attention, stream.get());
        }
    }

    void run_local_attention_cache(const PackedSessionContext& reference, int rows,
                                   int layer_index) {
        const size_t offset = static_cast<size_t>(layer_index) * maximum_batch;
        if (reference.options().kv_cache_mode == KvCacheMode::Int8) {
            launch_store_kv_int8_batch_ptrs(
                k.data(), v.data(), d_key_int8.data() + offset,
                d_value_int8.data() + offset, d_key_scales.data() + offset,
                d_value_scales.data() + offset, positions.data(), rows,
                shape_.num_key_value_heads, shape_.head_dim, stream.get());
            launch_gqa_decode_int8_batch_ptrs(
                q.data(), d_key_int8.data() + offset,
                d_value_int8.data() + offset, d_key_scales.data() + offset,
                d_value_scales.data() + offset, op_output.data(),
                positions.data(), rows, shape_.num_attention_heads,
                shape_.num_key_value_heads, shape_.head_dim,
                reference.options().fast_attention, stream.get());
            return;
        }
        launch_store_kv_batch_ptrs(
            k.data(), v.data(), d_key_bf16.data() + offset,
            d_value_bf16.data() + offset, positions.data(), rows,
            shape_.kv_width, stream.get());
        launch_gqa_decode_batch_ptrs(
            q.data(), d_key_bf16.data() + offset,
            d_value_bf16.data() + offset, op_output.data(), positions.data(),
            rows, shape_.num_attention_heads, shape_.num_key_value_heads,
            shape_.head_dim, reference.options().fast_attention,
            stream.get());
    }

    void run_attention_layer(const PackedSessionContext& reference,
                             const AttentionLayer& attention,
                             int rows, int layer_index,
                             bool segmented_attention,
                             int segmented_chunks) {
        project_attention_qkv(reference, attention, rows);
        if (paged_kv) {
            run_paged_attention_cache(reference, rows, layer_index,
                                      segmented_attention, segmented_chunks);
        } else {
            run_local_attention_cache(reference, rows, layer_index);
        }
        linear(op_output.data(), *attention.out, hidden.data(), rows,
               shape_.hidden, shape_.hidden,
               reference.options().fused_residuals ? 1.0f : 0.0f);
        launch_scale(hidden.data(), rows * shape_.hidden,
                     shape_.residual_multiplier, stream.get());
    }

    void run_convolution_layer(
        const PackedSessionContext& reference,
        const ConvolutionLayer& convolution,
        int rows, int layer_index, int ragged_requests = 0) {
        linear(normed.data(), *convolution.conv_in, conv_projected.data(),
               rows, 3 * shape_.hidden, shape_.hidden);
        const size_t offset = static_cast<size_t>(layer_index) * maximum_batch;
        if (ragged_requests != 0) {
            launch_conv_ragged_prefill(
                conv_projected.data(), convolution.conv_weight,
                d_conv_states.data() + offset, op_output.data(), positions.data(),
                d_span_offsets.data(), d_span_counts.data(), ragged_requests,
                shape_.hidden, shape_.conv_cache, stream.get());
        } else {
            launch_conv_decode_batch_ptrs(
                conv_projected.data(), convolution.conv_weight,
                d_conv_states.data() + offset, op_output.data(), positions.data(),
                rows, shape_.hidden, shape_.conv_cache, stream.get());
        }
        linear(op_output.data(), *convolution.conv_out, hidden.data(), rows,
               shape_.hidden, shape_.hidden,
               reference.options().fused_residuals ? 1.0f : 0.0f);
    }

    void run_mlp_layer(const PackedSessionContext& reference,
                       const LayerCommon& common_layer,
                       int rows,
                       const std::vector<PackedSessionContext>* batch_models = nullptr,
                       int layer_index = -1) {
        if (const MoeFfnWeights* moe = as_moe_ffn(common_layer.feed_forward)) {
            (void)moe;
            run_mlp_moe_layer(reference, common_layer, rows,
                              batch_models, layer_index);
            return;
        }
        launch_rmsnorm(hidden.data(), common_layer.ffn_norm, normed.data(),
                       rows, shape_.hidden, shape_.norm_eps,
                       stream.get());
        if (reference.options().fused_projections) {
            linear(normed.data(), *as_dense_ffn(common_layer.feed_forward)->w13, gate_up.data(), rows,
                    2 * shape_.intermediate, shape_.hidden);
            launch_swiglu_interleaved(gate_up.data(), activated.data(), rows,
                                       shape_.intermediate, stream.get());
        } else {
            const auto w1 = slice_rows(
                *as_dense_ffn(common_layer.feed_forward)->w13, 0, shape_.intermediate);
            const auto w3 = slice_rows(
                *as_dense_ffn(common_layer.feed_forward)->w13, shape_.intermediate,
                shape_.intermediate);
            const size_t plane =
                static_cast<size_t>(rows) * shape_.intermediate;
            linear(normed.data(), w1, gate_up.data(), rows,
                    shape_.intermediate, shape_.hidden);
            linear(normed.data(), w3, gate_up.data() + plane, rows,
                    shape_.intermediate, shape_.hidden);
            launch_swiglu_fused(gate_up.data(), activated.data(),
                                static_cast<int>(plane), stream.get());
        }
        if (reference.options().fused_residuals) {
            linear(activated.data(), *as_dense_ffn(common_layer.feed_forward)->w2, hidden.data(), rows,
                    shape_.hidden, shape_.intermediate, 1.0f);
        } else {
            linear(activated.data(), *as_dense_ffn(common_layer.feed_forward)->w2, mlp_output.data(), rows,
                    shape_.hidden, shape_.intermediate);
            launch_scale(mlp_output.data(), rows * shape_.hidden,
                         shape_.residual_multiplier, stream.get());
            launch_residual_add(hidden.data(), mlp_output.data(),
                                rows * shape_.hidden, stream.get());
        }
    }

    // MoE feed-forward for packed decode/ragged prefill. Mirrors the standalone
    // run_mlp_moe_decode path: RMSNorm -> cast -> router -> expert residency
    // resolution -> selected-expert FFN -> residual add. Router runs in float.
    void run_mlp_moe_layer(const PackedSessionContext& reference,
                            const LayerCommon& common_layer, int rows,
                            const std::vector<PackedSessionContext>* batch_models = nullptr,
                            int layer_index = -1) {
        const MoeFfnWeights& moe = *as_moe_ffn(common_layer.feed_forward);
        launch_rmsnorm(hidden.data(), common_layer.ffn_norm, normed.data(),
                       rows, shape_.hidden, shape_.norm_eps, stream.get());
        launch_cast_bf16_to_float(normed.data(), moe_hidden_float.data(),
                                  rows * shape_.hidden, stream.get());

        const lfm::MoeRouterConfig cfg = moe_router_config(shape_);
        lfm::MoeRouterDevice rdev;
        rdev.router_weight = moe.router_float;
        rdev.expert_bias = moe.expert_bias;
        rdev.hidden_data = moe_hidden_float.data();
        rdev.selected_experts = moe_sel.data();
        rdev.routing_weights = moe_routing_w.data();
        rdev.rows = rows;
        rdev.hidden_dim = shape_.hidden;
        launch_moe_router(rdev, cfg, moe_router_scratch.data(), stream.get());

        // Ensure selected experts are GPU-resident before the FFN reads them.
        // Resolve residency once for the entire batch of tokens on the shared layer cache.
        if (batch_models && !batch_models->empty() && layer_index >= 0) {
            const PackedSessionContext& session = batch_models->front();
            session.ensure_moe_experts_resident_packed(
                layer_index,
                moe_sel.data(),
                rows, stream.get(),
                moe.offloaded() ? moe_router_scratch.data() : nullptr);
        }

        moe_output_accum.zero_async(stream.get());
        const lfm::MoeFfnDevice fdev = moe_ffn_device(moe, shape_);
        launch_moe_ffn(fdev, moe_sel.data(), moe_routing_w.data(),
                       normed.data(), moe_output_accum.data(), rows,
                       shape_.experts_per_token, moe_gu_scratch.data(),
                       moe_act_scratch.data(), stream.get());
        launch_finalize_moe_output(moe_output_accum.data(), moe_output.data(),
                                   rows * shape_.hidden, stream.get());

        launch_residual_add(hidden.data(), moe_output.data(),
                            rows * shape_.hidden, stream.get());
    }

    void run_transformer_layers(const PackedSessionContext& reference, int rows,
                                bool segmented_attention,
                                int segmented_chunks,
                                const std::vector<PackedSessionContext>* batch_models = nullptr,
                                int ragged_requests = 0) {
        for (int layer_index = 0; layer_index < shape_.num_hidden_layers;
             ++layer_index) {
            const auto& layer = reference.layers()[layer_index];
            const auto& common_layer = common(layer);
            if (!reference.options().fused_residuals) {
                LFM_CUDA(cudaMemcpyAsync(
                    residual.data(), hidden.data(),
                    static_cast<size_t>(rows) * shape_.hidden *
                        sizeof(__nv_bfloat16),
                    cudaMemcpyDeviceToDevice, stream.get()));
            }
            launch_rmsnorm(hidden.data(), common_layer.operator_norm,
                           normed.data(), rows, shape_.hidden,
                           shape_.norm_eps, stream.get());
            if (const auto* attention =
                    as_attention(layer)) {
                run_attention_layer(reference, *attention, rows, layer_index,
                                    segmented_attention, segmented_chunks);
            } else {
                run_convolution_layer(
                    reference, *as_convolution(layer), rows,
                    layer_index, ragged_requests);
            }
            if (!reference.options().fused_residuals) {
                launch_residual_add(hidden.data(), residual.data(),
                                    rows * shape_.hidden, stream.get());
            }
            run_mlp_layer(reference, common_layer, rows, batch_models, layer_index);
        }
    }

    std::vector<int32_t> decode(
        const std::vector<PackedSessionContext>& models,
                                const std::vector<std::vector<uint32_t>>* page_tables) {
        if (models.empty()) return {};
        const PackedSessionContext& reference = validate_decode_batch(models);
        const int rows = static_cast<int>(models.size());
        active_plan_ = ExecutionPlan::compile(reference.options(), reference.max_context());
        ensure_gemm_dispatcher(reference.options());
        const auto started = std::chrono::steady_clock::now();
        const AttentionBatchPlan attention =
            prepare_batch_metadata(models, page_tables);

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
                       rows, shape_.hidden, shape_.norm_eps,
                       stream.get());
        linear(normed.data(), *reference.logits_weight(), logits.data(), rows,
               shape_.vocab_size, shape_.hidden);
        launch_scale(logits.data(), rows * shape_.vocab_size,
                     1.0f / shape_.logits_divisor, stream.get());
        launch_scatter_bf16_rows(
            logits.data(), d_logits.data(), rows, shape_.vocab_size,
            stream.get());
        launch_scatter_decode_state(
            sampled.data(), positions.data(), d_sampled_dest.data(),
            d_position_dest.data(), rows, stream.get());
        LFM_CUDA(cudaMemcpyAsync(sampled_host.data(), sampled.data(),
                                 static_cast<size_t>(rows) * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost, stream.get()));
        LFM_CUDA(cudaStreamSynchronize(stream.get()));

        const auto ended = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(ended - started).count();
        std::vector<int32_t> result(static_cast<size_t>(rows));
        for (int row = 0; row < rows; ++row) {
            const PackedSessionContext& model = models[static_cast<size_t>(row)];
            result[static_cast<size_t>(row)] = sampled_host.data()[row];
            model.set_sampled_host_value(sampled_host.data()[row]);
            model.set_position(model.position() + 1);
            model.metrics().cumulative_decode_ms += elapsed_ms;
            model.metrics().decoded_tokens += 1;
        }
        ++metric.steps;
        metric.tokens += static_cast<uint64_t>(rows);
        if (attention.segmented && paged_kv) {
            ++metric.segmented_paged_steps;
            metric.segmented_paged_tokens += static_cast<uint64_t>(rows);
        }
        metric.maximum_batch = std::max(metric.maximum_batch,
                                        static_cast<size_t>(rows));
        metric.cumulative_ms += elapsed_ms;
        active_plan_.reset();
        return result;
    }

    void prefill(const std::vector<PackedSessionContext>& models,
                 const std::vector<std::vector<uint32_t>>* page_tables,
                 const std::vector<int32_t>& explicit_tokens,
                 const std::vector<PackedPrefillRow>& row_descriptors) {
        if (models.empty()) return;
        const PackedSessionContext& reference =
            validate_prefill_batch(models, explicit_tokens, row_descriptors);
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
        active_plan_ = ExecutionPlan::compile(reference.options(), reference.max_context());
        ensure_gemm_dispatcher(reference.options());
        const auto started = std::chrono::steady_clock::now();
        copy_persistent_metadata(models);
        for (int request = 0; request < requests; ++request) {
            h_span_offsets.data()[request] = static_cast<int>(row_descriptors[request].token_offset);
            h_span_counts.data()[request] = static_cast<int>(row_descriptors[request].token_count);
            h_final_rows.data()[request] = h_span_offsets.data()[request] + h_span_counts.data()[request] - 1;
        }
        LFM_CUDA(cudaMemcpyAsync(d_span_offsets.data(), h_span_offsets.data(),
                                 requests * sizeof(int32_t), cudaMemcpyHostToDevice, stream.get()));
        LFM_CUDA(cudaMemcpyAsync(d_span_counts.data(), h_span_counts.data(),
                                 requests * sizeof(int32_t), cudaMemcpyHostToDevice, stream.get()));
        LFM_CUDA(cudaMemcpyAsync(d_final_rows.data(), h_final_rows.data(),
                                 requests * sizeof(int32_t), cudaMemcpyHostToDevice, stream.get()));
        const AttentionBatchPlan attention =
            prepare_flat_prefill_metadata(models, *page_tables, row_descriptors);
        LFM_CUDA(cudaMemcpyAsync(sampled.data(), explicit_tokens.data(),
                                 static_cast<size_t>(rows) * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, stream.get()));
        // Reuse the request pointer table by expanding only the seen-token
        // pointers for this launch: repeated writes set the same byte to one.
        std::vector<uint8_t*> flat_seen(static_cast<size_t>(rows));
        for (int request = 0; request < requests; ++request) {
            std::fill_n(flat_seen.data() + h_span_offsets.data()[request],
                        h_span_counts.data()[request], h_seen.data()[request]);
        }
        DeviceBuffer<uint8_t*> d_flat_seen(static_cast<size_t>(rows));
        LFM_CUDA(cudaMemcpyAsync(d_flat_seen.data(), flat_seen.data(),
                                 static_cast<size_t>(rows) * sizeof(uint8_t*),
                                 cudaMemcpyHostToDevice, stream.get()));
        launch_mark_seen_batch_ptrs(sampled.data(), d_flat_seen.data(), rows,
                                    shape_.vocab_size, stream.get());
        launch_embedding_rows(reference, rows);
        run_transformer_layers(reference, rows, attention.segmented,
                               attention.chunks, &models, requests);
        const bool any_finalize = std::any_of(row_descriptors.begin(), row_descriptors.end(),
            [](const PackedPrefillRow& row) { return row.finalize != 0; });
        if (any_finalize) {
            launch_rmsnorm(hidden.data(), reference.final_norm(), normed.data(),
                           rows, shape_.hidden, shape_.norm_eps, stream.get());
            linear(normed.data(), *reference.logits_weight(), logits.data(), rows,
                   shape_.vocab_size, shape_.hidden);
            launch_scale(logits.data(), rows * shape_.vocab_size,
                         1.0f / shape_.logits_divisor, stream.get());
            launch_scatter_bf16_selected_rows(logits.data(), d_final_rows.data(),
                                               d_logits.data(), requests,
                                               shape_.vocab_size, stream.get());
        }
        launch_scatter_selected_decode_state(
            sampled.data(), positions.data(), d_final_rows.data(),
            d_sampled_dest.data(), d_position_dest.data(), requests, stream.get());
        for (int request = 0; request < requests; ++request) {
            const PackedSessionContext& model = models[static_cast<size_t>(request)];
            const PackedPrefillRow& descriptor = row_descriptors[static_cast<size_t>(request)];
            model.set_sampled_host_value(explicit_tokens[static_cast<size_t>(h_final_rows.data()[request])]);
            model.set_position(model.position() + static_cast<int>(descriptor.token_count));
            model.metrics().prefill_tokens += descriptor.token_count;
            model.set_phase(descriptor.finalize != 0 ? SessionPhase::Ready : SessionPhase::Prefilling);
            if (model.phase() == SessionPhase::Ready) {
                model.set_active_segmented_attention(model.use_segmented_attention(model.position()));
            }
        }
        metric.ragged_prefill_tokens += static_cast<uint64_t>(rows);
        metric.maximum_prefill_batch = std::max(metric.maximum_prefill_batch,
                                                static_cast<size_t>(requests));
        ++metric.ragged_prefill_transformer_passes;

        const auto ended = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(ended - started).count();
        const double per_token_ms = explicit_tokens.empty()
            ? 0.0
            : elapsed_ms / static_cast<double>(explicit_tokens.size());
        for (int row = 0; row < requests; ++row) {
            const PackedSessionContext& model = models[static_cast<size_t>(row)];
            model.metrics().last_prefill_ms +=
                per_token_ms * static_cast<double>(row_descriptors[static_cast<size_t>(row)].token_count);
        }
        ++metric.ragged_prefill_steps;
        metric.cumulative_prefill_ms += elapsed_ms;
        active_plan_.reset();
    }
};

PackedDecodeExecutor::PackedDecodeExecutor(size_t maximum_sessions,
                                           size_t maximum_prefill_tokens,
                                           PhysicalPagedKvCache* paged_kv,
                                           const ModelShape& shape)
    : impl_(std::make_unique<PackedDecodeExecutorImpl>(
          maximum_sessions, maximum_prefill_tokens, paged_kv, shape)) {}

PackedDecodeExecutor::~PackedDecodeExecutor() = default;

bool PackedDecodeExecutor::eligible(const PackedSessionContext& model,
                                    std::string* reason) const {
    return impl_->eligible(model, reason);
}

std::vector<int32_t> PackedDecodeExecutor::decode(
    const std::vector<PackedSessionContext>& models) {
    return impl_->decode(models, nullptr);
}

std::vector<int32_t> PackedDecodeExecutor::decode(
    const std::vector<PackedSessionContext>& models,
    const std::vector<std::vector<uint32_t>>& page_tables) {
    return impl_->decode(models, &page_tables);
}

void PackedDecodeExecutor::prefill(
    const std::vector<PackedSessionContext>& models,
    const std::vector<std::vector<uint32_t>>& page_tables,
    const std::vector<int32_t>& tokens,
    const std::vector<PackedPrefillRow>& rows) {
    impl_->prefill(models, &page_tables, tokens, rows);
}

size_t PackedDecodeExecutor::maximum_batch() const {
    return impl_->maximum_batch;
}

size_t PackedDecodeExecutor::maximum_prefill_tokens() const {
    return impl_->maximum_prefill_token_capacity;
}

PackedDecodeMetrics PackedDecodeExecutor::metrics() const {
    return impl_->metric;
}

} // namespace lfm
