#include "lfm/packed.hpp"

#include "lfm/kernels.cuh"
#include "lfm/detail/model_impl.hpp"
#include "lfm/paged_kv.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <string>
#include <utility>

namespace lfm {

struct PackedDecodeExecutorImpl {
    explicit PackedDecodeExecutorImpl(size_t maximum_batch_value,
                                      PhysicalPagedKvCache* paged_kv_value,
                                      const ModelShape& shape)
        : maximum_batch(maximum_batch_value),
          paged_kv(paged_kv_value),
          shape_(shape),
          stream(),
          cublas(stream.get()),
          positions(maximum_batch),
          sampled(maximum_batch),
          sampled_host(maximum_batch),
          temperatures(maximum_batch),
          repetition_penalties(maximum_batch),
          top_k(maximum_batch),
          top_p(maximum_batch),
          hidden(maximum_batch * shape_.hidden),
          residual(maximum_batch * shape_.hidden),
          normed(maximum_batch * shape_.hidden),
          op_output(maximum_batch * shape_.hidden),
          qkv_output(maximum_batch * shape_.qkv_width),
          q(maximum_batch * shape_.q_width),
          k(maximum_batch * shape_.kv_width),
          v(maximum_batch * shape_.kv_width),
          conv_projected(maximum_batch * 3 * shape_.hidden),
          gate_up(maximum_batch * 2 * shape_.intermediate),
          activated(maximum_batch * shape_.intermediate),
          mlp_output(maximum_batch * shape_.hidden),
          logits(maximum_batch * shape_.vocab_size),
          sampling_scores(maximum_batch * shape_.vocab_size),
          selected_values(maximum_batch * static_cast<size_t>(kMaxTopK)),
          selected_indices(maximum_batch * static_cast<size_t>(kMaxTopK)),
          h_positions(maximum_batch),
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
          h_page_tables(paged_kv_value ? maximum_batch * static_cast<size_t>(paged_kv_value->max_pages_per_request()) : 0),
          d_page_tables(paged_kv_value ? maximum_batch * static_cast<size_t>(paged_kv_value->max_pages_per_request()) : 0) {
        if (maximum_batch == 0) {
            throw std::invalid_argument("packed maximum_batch must be positive");
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
    DeviceBuffer<float> segmented_partial_max;
    DeviceBuffer<float> segmented_partial_denom;
    DeviceBuffer<float> segmented_partial_accum;
    size_t segmented_scalar_capacity = 0;
    size_t segmented_accum_capacity = 0;

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

    static bool options_compatible(const IPackedSession& left,
                                   const IPackedSession& right,
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

    bool eligible(const IPackedSession& model, std::string* reason) const {
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
        weight.validate_storage();
        switch (weight.kind) {
            case LinearStorageKind::Int4:
                launch_w4a16_linear(x, weight.int4, weight.scales, y,
                                    m, n, k_width, beta, stream.get());
                return;
            case LinearStorageKind::Int8:
                launch_w8a16_linear(x, weight.int8, weight.scales, y,
                                    m, n, k_width, beta, stream.get());
                return;
            case LinearStorageKind::Bf16: {
                const float alpha = 1.0f;
                LFM_CUBLAS(cublasGemmEx(
                    cublas.get(), CUBLAS_OP_T, CUBLAS_OP_N,
                    n, m, k_width, &alpha,
                    weight.bf16, CUDA_R_16BF, k_width,
                    x, CUDA_R_16BF, k_width,
                    &beta, y, CUDA_R_16BF, n,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                return;
            }
        }
        throw std::runtime_error("unknown packed linear storage");
    }

    void copy_metadata(const std::vector<IPackedSession*>& models,
                       const std::vector<std::vector<uint32_t>>* page_tables) {
        const size_t rows = models.size();
        const int page_stride = paged_kv ? paged_kv->max_pages_per_request() : 0;
        if (paged_kv) {
            if (!page_tables || page_tables->size() != rows) {
                throw std::invalid_argument("paged packed decode requires one page table per row");
            }
            std::fill_n(h_page_tables.data(), maximum_batch * static_cast<size_t>(page_stride),
                        std::numeric_limits<uint32_t>::max());
        }
        for (size_t row = 0; row < rows; ++row) {
            IPackedSession& model = *models[row];
            h_positions.data()[row] = model.position();
            h_temperatures.data()[row] = model.generation().temperature;
            h_repetition_penalties.data()[row] =
                model.generation().repetition_penalty;
            h_top_k.data()[row] = model.generation().top_k;
            h_top_p.data()[row] = model.generation().top_p;
            h_logits.data()[row] = model.logits().data();
            h_seen.data()[row] = model.seen_tokens().data();
            h_rng.data()[row] = model.rng_state().data();
            h_sampled_dest.data()[row] = model.sampled_device().data();
            h_position_dest.data()[row] = model.position_device().data();
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
        if (paged_kv) {
            const size_t page_bytes = rows * static_cast<size_t>(page_stride) * sizeof(uint32_t);
            LFM_CUDA(cudaMemcpyAsync(d_page_tables.data(), h_page_tables.data(),
                                     page_bytes, cudaMemcpyHostToDevice,
                                     stream.get()));
        }

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


    IPackedSession& validate_decode_batch(const std::vector<IPackedSession*>& models) const {
        if (models.empty()) throw std::invalid_argument("packed batch is empty");
        if (models.size() > maximum_batch) {
            throw std::invalid_argument("packed batch exceeds executor capacity");
        }
        if (!models.front()) throw std::invalid_argument("null packed session");
        IPackedSession& reference = *models.front();
        std::string reason;
        if (!eligible(reference, &reason)) throw std::invalid_argument(reason);
        for (size_t row = 1; row < models.size(); ++row) {
            if (!models[row]) throw std::invalid_argument("null packed session");
            if (std::find(models.begin(), models.begin() + static_cast<ptrdiff_t>(row),
                          models[row]) != models.begin() + static_cast<ptrdiff_t>(row)) {
                throw std::invalid_argument("duplicate session in packed batch");
            }
            if (!eligible(*models[row], &reason) ||
                !options_compatible(reference, *models[row], &reason)) {
                throw std::invalid_argument(reason);
            }
        }
        return reference;
    }

    IPackedSession& validate_prefill_batch(
        const std::vector<IPackedSession*>& models,
        const std::vector<int32_t>& explicit_tokens,
        const std::vector<uint8_t>& finalize_rows) const {
        if (models.empty()) throw std::invalid_argument("packed batch is empty");
        if (explicit_tokens.size() != models.size() ||
            finalize_rows.size() != models.size()) {
            throw std::invalid_argument(
                "ragged prefill needs one token and finalize flag per row");
        }
        if (models.size() > maximum_batch) {
            throw std::invalid_argument("packed batch exceeds executor capacity");
        }
        if (!models.front()) throw std::invalid_argument("null packed session");
        IPackedSession& reference = *models.front();
        std::string reason;
        const auto eligible_prefill = [&](const IPackedSession& model) {
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
        for (size_t row = 1; row < models.size(); ++row) {
            if (!models[row]) throw std::invalid_argument("null packed session");
            if (std::find(models.begin(), models.begin() + static_cast<ptrdiff_t>(row),
                          models[row]) != models.begin() + static_cast<ptrdiff_t>(row)) {
                throw std::invalid_argument("duplicate session in packed batch");
            }
            if (!eligible_prefill(*models[row]) ||
                !options_compatible(reference, *models[row], &reason)) {
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
        const std::vector<IPackedSession*>& models,
        const std::vector<std::vector<uint32_t>>* page_tables) {
        copy_metadata(models, page_tables);
        int maximum_position = 0;
        AttentionBatchPlan plan;
        for (size_t row = 0; row < models.size(); ++row) {
            maximum_position = std::max(maximum_position,
                                        h_positions.data()[row]);
            plan.segmented = plan.segmented ||
                models[row]->use_segmented_attention(
                    h_positions.data()[row]);
        }
        if (plan.segmented) {
            const int chunk_tokens =
                models.front()->options().attention_chunk_tokens;
            plan.chunks = (maximum_position + 1 + chunk_tokens - 1) /
                          chunk_tokens;
            ensure_segmented_workspace(static_cast<int>(models.size()),
                                       plan.chunks);
        }
        return plan;
    }

    void launch_embedding_rows(const IPackedSession& reference, int rows) {
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
    }

    void project_attention_qkv(
        IPackedSession& reference,
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
        launch_qk_norm_rope_batch_positions(
            q.data(), k.data(), attention.q_norm, attention.k_norm,
            reference.rope_cos(), reference.rope_sin(),
            positions.data(), rows, shape_.num_attention_heads, shape_.num_key_value_heads,
            shape_.head_dim, shape_.norm_eps,
            reference.options().fast_attention, stream.get());
    }

    void run_paged_attention_cache(IPackedSession& reference, int rows,
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

    void run_local_attention_cache(IPackedSession& reference, int rows,
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

    void run_attention_layer(IPackedSession& reference,
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
    }

    void run_convolution_layer(
        IPackedSession& reference,
        const ConvolutionLayer& convolution,
        int rows, int layer_index) {
        linear(normed.data(), *convolution.conv_in, conv_projected.data(),
               rows, 3 * shape_.hidden, shape_.hidden);
        const size_t offset = static_cast<size_t>(layer_index) * maximum_batch;
        launch_conv_decode_batch_ptrs(
            conv_projected.data(), convolution.conv_weight,
            d_conv_states.data() + offset, op_output.data(), positions.data(),
            rows, shape_.hidden, shape_.conv_cache, stream.get());
        linear(op_output.data(), *convolution.conv_out, hidden.data(), rows,
               shape_.hidden, shape_.hidden,
               reference.options().fused_residuals ? 1.0f : 0.0f);
    }

    void run_mlp_layer(IPackedSession& reference,
                       const LayerCommon& common_layer,
                       int rows) {
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
            launch_residual_add(hidden.data(), mlp_output.data(),
                                rows * shape_.hidden, stream.get());
        }
    }

    void run_transformer_layers(IPackedSession& reference, int rows,
                                bool segmented_attention,
                                int segmented_chunks) {
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
                    layer_index);
            }
            if (!reference.options().fused_residuals) {
                launch_residual_add(hidden.data(), residual.data(),
                                    rows * shape_.hidden, stream.get());
            }
            run_mlp_layer(reference, common_layer, rows);
        }
    }

    std::vector<int32_t> decode(const std::vector<IPackedSession*>& models,
                                const std::vector<std::vector<uint32_t>>* page_tables) {
        if (models.empty()) return {};
        IPackedSession& reference = validate_decode_batch(models);
        const int rows = static_cast<int>(models.size());
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
                               attention.chunks);
        launch_rmsnorm(hidden.data(), reference.final_norm(), normed.data(),
                       rows, shape_.hidden, shape_.norm_eps,
                       stream.get());
        linear(normed.data(), *reference.embedding(), logits.data(), rows,
               shape_.vocab_size, shape_.hidden);
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
            IPackedSession& model = *models[static_cast<size_t>(row)];
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
        return result;
    }

    void prefill_step(const std::vector<IPackedSession*>& models,
                      const std::vector<std::vector<uint32_t>>* page_tables,
                      const std::vector<int32_t>& explicit_tokens,
                      const std::vector<uint8_t>& finalize_rows) {
        if (models.empty()) return;
        IPackedSession& reference =
            validate_prefill_batch(models, explicit_tokens, finalize_rows);
        const int rows = static_cast<int>(models.size());
        const bool any_finalize = std::any_of(
            finalize_rows.begin(), finalize_rows.end(),
            [](uint8_t value) { return value != 0; });
        const auto started = std::chrono::steady_clock::now();
        const AttentionBatchPlan attention =
            prepare_batch_metadata(models, page_tables);

        LFM_CUDA(cudaMemcpyAsync(sampled.data(), explicit_tokens.data(),
                                 static_cast<size_t>(rows) * sizeof(int32_t),
                                 cudaMemcpyHostToDevice, stream.get()));
        launch_mark_seen_batch_ptrs(sampled.data(), d_seen.data(), rows,
                                    shape_.vocab_size, stream.get());

        launch_embedding_rows(reference, rows);

        run_transformer_layers(reference, rows, attention.segmented,
                               attention.chunks);
        if (any_finalize) {
            launch_rmsnorm(hidden.data(), reference.final_norm(), normed.data(),
                           rows, shape_.hidden, shape_.norm_eps,
                           stream.get());
            linear(normed.data(), *reference.embedding(), logits.data(), rows,
                   shape_.vocab_size, shape_.hidden);
            launch_scatter_bf16_rows(
                logits.data(), d_logits.data(), rows, shape_.vocab_size,
                stream.get());
        }
        launch_scatter_decode_state(
            sampled.data(), positions.data(), d_sampled_dest.data(),
            d_position_dest.data(), rows, stream.get());
        LFM_CUDA(cudaStreamSynchronize(stream.get()));

        const auto ended = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(ended - started).count();
        for (int row = 0; row < rows; ++row) {
            IPackedSession& model = *models[static_cast<size_t>(row)];
            model.set_sampled_host_value(
                explicit_tokens[static_cast<size_t>(row)]);
            model.set_position(model.position() + 1);
            model.metrics().prefill_tokens += 1;
            model.metrics().last_prefill_ms += elapsed_ms /
                static_cast<double>(rows);
            model.set_phase(finalize_rows[static_cast<size_t>(row)] != 0
                ? SessionPhase::Ready
                : SessionPhase::Prefilling);
            if (model.phase() == SessionPhase::Ready) {
                model.set_active_segmented_attention(
                    model.use_segmented_attention(model.position()));
            }
        }
        ++metric.ragged_prefill_steps;
        metric.ragged_prefill_tokens += static_cast<uint64_t>(rows);
        metric.maximum_prefill_batch = std::max(
            metric.maximum_prefill_batch, static_cast<size_t>(rows));
        metric.cumulative_prefill_ms += elapsed_ms;
    }
};

PackedDecodeExecutor::PackedDecodeExecutor(size_t maximum_batch,
                                                 PhysicalPagedKvCache* paged_kv,
                                                 const ModelShape& shape)
    : impl_(std::make_unique<PackedDecodeExecutorImpl>(maximum_batch, paged_kv, shape)) {}

PackedDecodeExecutor::~PackedDecodeExecutor() = default;

bool PackedDecodeExecutor::eligible(const IPackedSession& model,
                                    std::string* reason) const {
    return impl_->eligible(model, reason);
}

bool PackedDecodeExecutor::eligible(const LfmModel& model,
                                    std::string* reason) const {
    return eligible(model.packed_session(), reason);
}

std::vector<int32_t> PackedDecodeExecutor::decode(
    const std::vector<IPackedSession*>& models) {
    return impl_->decode(models, nullptr);
}

std::vector<int32_t> PackedDecodeExecutor::decode(
    const std::vector<LfmModel*>& models) {
    std::vector<IPackedSession*> sessions;
    sessions.reserve(models.size());
    for (LfmModel* model : models) {
        sessions.push_back(&model->packed_session());
    }
    return decode(sessions);
}

std::vector<int32_t> PackedDecodeExecutor::decode(
    const std::vector<IPackedSession*>& models,
    const std::vector<std::vector<uint32_t>>& page_tables) {
    return impl_->decode(models, &page_tables);
}

std::vector<int32_t> PackedDecodeExecutor::decode(
    const std::vector<LfmModel*>& models,
    const std::vector<std::vector<uint32_t>>& page_tables) {
    std::vector<IPackedSession*> sessions;
    sessions.reserve(models.size());
    for (LfmModel* model : models) {
        sessions.push_back(&model->packed_session());
    }
    return decode(sessions, page_tables);
}

void PackedDecodeExecutor::prefill_step(
    const std::vector<IPackedSession*>& models,
    const std::vector<std::vector<uint32_t>>& page_tables,
    const std::vector<int32_t>& tokens,
    const std::vector<uint8_t>& finalize_rows) {
    impl_->prefill_step(models, &page_tables, tokens, finalize_rows);
}

void PackedDecodeExecutor::prefill_step(
    const std::vector<LfmModel*>& models,
    const std::vector<std::vector<uint32_t>>& page_tables,
    const std::vector<int32_t>& tokens,
    const std::vector<uint8_t>& finalize_rows) {
    std::vector<IPackedSession*> sessions;
    sessions.reserve(models.size());
    for (LfmModel* model : models) {
        sessions.push_back(&model->packed_session());
    }
    prefill_step(sessions, page_tables, tokens, finalize_rows);
}

size_t PackedDecodeExecutor::maximum_batch() const {
    return impl_->maximum_batch;
}

PackedDecodeMetrics PackedDecodeExecutor::metrics() const {
    return impl_->metric;
}

} // namespace lfm
