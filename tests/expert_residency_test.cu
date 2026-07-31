#include "celeg/runtime/moe/expert_residency.hpp"
#include "celeg/runtime/moe.hpp"
#include "celeg/backend/cuda/utils.cuh"
#include "celeg/detail/model/types.hpp"
#include "celeg/model/resolved.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

__nv_bfloat16 to_bf16(float v) { return __float2bfloat16(v); }
float from_bf16(__nv_bfloat16 v) { return __bfloat162float(v); }

struct Problem {
    int rows = 3;
    int hidden = 8;
    int inter = 16;
    int experts = 6;
    int K = 4;
    std::vector<__nv_bfloat16> gate_up;   // [E * 2*inter * hidden]
    std::vector<__nv_bfloat16> down;      // [E * hidden * inter]
    std::vector<__nv_bfloat16> hidden_vec; // [rows * hidden]
    std::vector<int> sel;                 // [rows * K]
    std::vector<float> wts;               // [rows * K]
};

Problem build() {
    Problem p;
    p.gate_up.resize(static_cast<size_t>(p.experts) * 2 * p.inter * p.hidden);
    for (size_t i = 0; i < p.gate_up.size(); ++i)
        p.gate_up[i] = to_bf16(std::cos(static_cast<float>(i) * 0.11f) * 0.8f);
    p.down.resize(static_cast<size_t>(p.experts) * p.hidden * p.inter);
    for (size_t i = 0; i < p.down.size(); ++i)
        p.down[i] = to_bf16(std::sin(static_cast<float>(i) * 0.07f) * 0.8f);
    p.hidden_vec.resize(static_cast<size_t>(p.rows) * p.hidden);
    for (size_t i = 0; i < p.hidden_vec.size(); ++i)
        p.hidden_vec[i] = to_bf16(std::sin(static_cast<float>(i) * 0.3f) + 0.5f);
    // Deterministic expert selection: each token picks 4 distinct experts.
    p.sel = {0, 2, 4, 5,  1, 3, 5, 0,  2, 4, 1, 3};
    p.wts = {0.4f, 0.3f, 0.2f, 0.1f,  0.25f, 0.25f, 0.25f, 0.25f,
             0.5f, 0.2f, 0.2f, 0.1f};
    return p;
}

// Runs launch_moe_ffn with a fully-contiguous device layout (baseline).
std::vector<__nv_bfloat16> run_contiguous(const Problem& p, celeg::CudaStream& stream) {
    celeg::DeviceBuffer<__nv_bfloat16> d_gate_up(p.gate_up.size());
    celeg::DeviceBuffer<__nv_bfloat16> d_down(p.down.size());
    celeg::DeviceBuffer<__nv_bfloat16> d_hidden(p.hidden_vec.size());
    celeg::DeviceBuffer<__nv_bfloat16> d_out(p.hidden_vec.size());
    celeg::DeviceBuffer<float> d_accum(p.hidden_vec.size());
    celeg::DeviceBuffer<int> d_sel(p.sel.size());
    celeg::DeviceBuffer<float> d_wts(p.wts.size());
    celeg::DeviceBuffer<__nv_bfloat16> d_gu_scratch(
        static_cast<size_t>(p.rows) * p.K * 2 * p.inter);
    celeg::DeviceBuffer<__nv_bfloat16> d_act_scratch(
        static_cast<size_t>(p.rows) * p.K * p.inter);
    CELEG_CUDA(cudaMemcpy(d_gate_up.data(), p.gate_up.data(), d_gate_up.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(d_down.data(), p.down.data(), d_down.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(d_hidden.data(), p.hidden_vec.data(), d_hidden.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(d_sel.data(), p.sel.data(), d_sel.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(d_wts.data(), p.wts.data(), d_wts.bytes(), cudaMemcpyHostToDevice));
    d_accum.zero_async(stream.get());

    celeg::MoeFfnDevice fdev;
    fdev.gate_up = d_gate_up.data();
    fdev.down = d_down.data();
    fdev.num_experts = p.experts;
    fdev.inter = p.inter;
    fdev.hidden_dim = p.hidden;
    fdev.expert_gate_up_stride = static_cast<size_t>(2) * p.inter * p.hidden;
    fdev.expert_down_stride = static_cast<size_t>(p.hidden) * p.inter;
    celeg::launch_moe_ffn(fdev, d_sel.data(), d_wts.data(), d_hidden.data(),
                        d_accum.data(), p.rows, p.K, d_gu_scratch.data(),
                        d_act_scratch.data(), stream.get());
    celeg::launch_finalize_moe_output(d_accum.data(), d_out.data(),
                                    static_cast<int>(p.hidden_vec.size()),
                                    stream.get());
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    std::vector<__nv_bfloat16> out(p.hidden_vec.size());
    CELEG_CUDA(cudaMemcpy(out.data(), d_out.data(), d_out.bytes(), cudaMemcpyDeviceToHost));
    return out;
}

// Runs launch_moe_ffn through the residency layer with `capacity` GPU-resident
// experts (the rest served from host-mapped memory), so it exercises both the
// GPU cache pointers and the host-mapped fallback pointers in one kernel.
std::vector<__nv_bfloat16> run_offload(const Problem& p, int capacity,
                                       const std::vector<int>& resident,
                                       celeg::CudaStream& stream) {
    const size_t gate_up_per_expert = static_cast<size_t>(2) * p.inter * p.hidden;
    const size_t down_per_expert = static_cast<size_t>(p.hidden) * p.inter;
    const size_t gate_up_bytes = gate_up_per_expert * sizeof(__nv_bfloat16);
    const size_t down_bytes = down_per_expert * sizeof(__nv_bfloat16);

    // Host expert store: register each expert's bytes and get device pointers.
    celeg::HostExpertStore store;
    std::vector<const __nv_bfloat16*> gu_host(p.experts);
    std::vector<const __nv_bfloat16*> dw_host(p.experts);
    for (int e = 0; e < p.experts; ++e) {
        gu_host[e] = static_cast<const __nv_bfloat16*>(store.store_pinned_copy(
            p.gate_up.data() + static_cast<size_t>(e) * gate_up_per_expert, gate_up_bytes));
        dw_host[e] = static_cast<const __nv_bfloat16*>(store.store_pinned_copy(
            p.down.data() + static_cast<size_t>(e) * down_per_expert, down_bytes));
    }

    celeg::ExpertLayerCache cache(p.experts, capacity, gate_up_bytes, down_bytes);
    cache.set_host_sources(gu_host, dw_host);
    cache.seed(resident, stream.get());
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));

    celeg::DeviceBuffer<__nv_bfloat16> d_hidden(p.hidden_vec.size());
    celeg::DeviceBuffer<__nv_bfloat16> d_out(p.hidden_vec.size());
    celeg::DeviceBuffer<float> d_accum(p.hidden_vec.size());
    celeg::DeviceBuffer<int> d_sel(p.sel.size());
    celeg::DeviceBuffer<float> d_wts(p.wts.size());
    celeg::DeviceBuffer<__nv_bfloat16> d_gu_scratch(
        static_cast<size_t>(p.rows) * p.K * 2 * p.inter);
    celeg::DeviceBuffer<__nv_bfloat16> d_act_scratch(
        static_cast<size_t>(p.rows) * p.K * p.inter);
    CELEG_CUDA(cudaMemcpy(d_hidden.data(), p.hidden_vec.data(), d_hidden.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(d_sel.data(), p.sel.data(), d_sel.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(d_wts.data(), p.wts.data(), d_wts.bytes(), cudaMemcpyHostToDevice));
    d_accum.zero_async(stream.get());

    celeg::MoeFfnDevice fdev;
    fdev.num_experts = p.experts;
    fdev.inter = p.inter;
    fdev.hidden_dim = p.hidden;
    fdev.gate_up_ptrs = cache.gate_up_ptrs();
    fdev.down_ptrs = cache.down_ptrs();
    celeg::launch_moe_ffn(fdev, d_sel.data(), d_wts.data(), d_hidden.data(),
                        d_accum.data(), p.rows, p.K, d_gu_scratch.data(),
                        d_act_scratch.data(), stream.get());
    celeg::launch_finalize_moe_output(d_accum.data(), d_out.data(),
                                    static_cast<int>(p.hidden_vec.size()),
                                    stream.get());
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    std::vector<__nv_bfloat16> out(p.hidden_vec.size());
    CELEG_CUDA(cudaMemcpy(out.data(), d_out.data(), d_out.bytes(), cudaMemcpyDeviceToHost));
    return out;
}

// Runtime check that is NOT compiled out under NDEBUG (unlike assert).
bool g_failed = false;
void check(bool cond, const char* what) {
    if (!cond) {
        std::cerr << "expert_residency_test CHECK FAILED: " << what << "\n";
        g_failed = true;
    }
}

// Both paths read the same BF16 expert bytes and run the same kernel, so the
// only difference is the non-deterministic BF16 atomic accumulation order in
// moe_ffn_kernel (BF16 addition is non-associative). Compare against the
// magnitude of the reference rather than requiring bit-exact equality; a real
// routing/residency bug would send a wrong expert and blow well past this.
void check_close(const std::vector<__nv_bfloat16>& a,
                 const std::vector<__nv_bfloat16>& b, const char* what) {
    check(a.size() == b.size(), "size mismatch");
    float max_abs = 0.0f;
    float max_ref = 1e-6f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(from_bf16(a[i]) - from_bf16(b[i])));
        max_ref = std::max(max_ref, std::fabs(from_bf16(b[i])));
    }
    const float rel = max_abs / max_ref;
    std::cout << "expert_residency_test: " << what << " max abs diff = " << max_abs
              << ", rel = " << rel << "\n";
    check(rel < 0.05f, what);
}

} // namespace

// Drives the exact model-level path used by Model::Impl when offload is
// enabled: build an offloaded MoeFfnWeights with indirect pointer tables from
// an ExpertLayerCache, resolve it through moe_ffn_device(), and run the kernel.
std::vector<__nv_bfloat16> run_model_offload(const Problem& p, int capacity,
                                             const std::vector<int>& resident,
                                             celeg::CudaStream& stream) {
    const size_t gu_pe = static_cast<size_t>(2) * p.inter * p.hidden;
    const size_t dw_pe = static_cast<size_t>(p.hidden) * p.inter;
    const size_t gu_bytes = gu_pe * sizeof(__nv_bfloat16);
    const size_t dw_bytes = dw_pe * sizeof(__nv_bfloat16);

    celeg::HostExpertStore store;
    std::vector<const __nv_bfloat16*> gu(p.experts), dw(p.experts);
    for (int e = 0; e < p.experts; ++e) {
        gu[e] = static_cast<const __nv_bfloat16*>(store.store_pinned_copy(
            p.gate_up.data() + static_cast<size_t>(e) * gu_pe, gu_bytes));
        dw[e] = static_cast<const __nv_bfloat16*>(store.store_pinned_copy(
            p.down.data() + static_cast<size_t>(e) * dw_pe, dw_bytes));
    }

    celeg::ExpertLayerCache cache(p.experts, capacity, gu_bytes, dw_bytes);
    cache.set_host_sources(gu, dw);
    cache.seed(resident, stream.get());
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));

    celeg::MoeFfnWeights moe;
    moe.gate_up_ptrs = cache.gate_up_ptrs();
    moe.down_ptrs = cache.down_ptrs();

    celeg::RuntimeTopology shape;
    shape.num_experts = p.experts;
    shape.experts_per_token = p.K;
    shape.moe_intermediate = p.inter;
    shape.hidden = p.hidden;

    celeg::MoeFfnDevice fdev = celeg::moe_ffn_device(moe, shape);
    check(fdev.gate_up_ptrs == cache.gate_up_ptrs(),
          "moe_ffn_device indirect gate_up_ptrs");
    check(fdev.down_ptrs == cache.down_ptrs(),
          "moe_ffn_device indirect down_ptrs");

    celeg::DeviceBuffer<__nv_bfloat16> d_hidden(p.hidden_vec.size());
    celeg::DeviceBuffer<__nv_bfloat16> d_out(p.hidden_vec.size());
    celeg::DeviceBuffer<float> d_accum(p.hidden_vec.size());
    celeg::DeviceBuffer<int> d_sel(p.sel.size());
    celeg::DeviceBuffer<float> d_wts(p.wts.size());
    celeg::DeviceBuffer<__nv_bfloat16> d_gu_scratch(
        static_cast<size_t>(p.rows) * p.K * 2 * p.inter);
    celeg::DeviceBuffer<__nv_bfloat16> d_act_scratch(
        static_cast<size_t>(p.rows) * p.K * p.inter);
    CELEG_CUDA(cudaMemcpy(d_hidden.data(), p.hidden_vec.data(), d_hidden.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(d_sel.data(), p.sel.data(), d_sel.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(d_wts.data(), p.wts.data(), d_wts.bytes(), cudaMemcpyHostToDevice));
    d_accum.zero_async(stream.get());

    celeg::launch_moe_ffn(fdev, d_sel.data(), d_wts.data(), d_hidden.data(),
                        d_accum.data(), p.rows, p.K, d_gu_scratch.data(),
                        d_act_scratch.data(), stream.get());
    celeg::launch_finalize_moe_output(d_accum.data(), d_out.data(),
                                    static_cast<int>(p.hidden_vec.size()),
                                    stream.get());
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    std::vector<__nv_bfloat16> out(p.hidden_vec.size());
    CELEG_CUDA(cudaMemcpy(out.data(), d_out.data(), d_out.bytes(), cudaMemcpyDeviceToHost));
    return out;
}

int main() {
    try {
        const Problem p = build();
        celeg::CudaStream stream;

        const std::vector<__nv_bfloat16> baseline = run_contiguous(p, stream);

        // All experts host-mapped (capacity 0 would allocate no slots; use a
        // small cache and seed a subset so both paths fire).
        const std::vector<__nv_bfloat16> partial =
            run_offload(p, /*capacity=*/3, /*resident=*/{0, 3, 5}, stream);
        check_close(baseline, partial, "partial cache (mixed cache+host)");

        // All experts resident in cache.
        const std::vector<__nv_bfloat16> full =
            run_offload(p, /*capacity=*/p.experts,
                        /*resident=*/{0, 1, 2, 3, 4, 5}, stream);
        check_close(baseline, full, "full cache");

        // Model-level path: MoeFfnWeights + moe_ffn_device() indirection.
        const std::vector<__nv_bfloat16> model_off =
            run_model_offload(p, /*capacity=*/3, /*resident=*/{0, 3, 5}, stream);
        check_close(baseline, model_off, "model-level offload (moe_ffn_device)");

        // Promotion/eviction bookkeeping.
        {
            const size_t gu_bytes =
                static_cast<size_t>(2) * p.inter * p.hidden * sizeof(__nv_bfloat16);
            const size_t dw_bytes =
                static_cast<size_t>(p.hidden) * p.inter * sizeof(__nv_bfloat16);
            celeg::HostExpertStore store;
            std::vector<const __nv_bfloat16*> gu(p.experts), dw(p.experts);
            const size_t gu_pe = static_cast<size_t>(2) * p.inter * p.hidden;
            const size_t dw_pe = static_cast<size_t>(p.hidden) * p.inter;
            for (int e = 0; e < p.experts; ++e) {
                gu[e] = static_cast<const __nv_bfloat16*>(store.store_pinned_copy(
                    p.gate_up.data() + static_cast<size_t>(e) * gu_pe, gu_bytes));
                dw[e] = static_cast<const __nv_bfloat16*>(store.store_pinned_copy(
                    p.down.data() + static_cast<size_t>(e) * dw_pe, dw_bytes));
            }
            celeg::ExpertLayerCache cache(p.experts, 2, gu_bytes, dw_bytes);
            cache.set_host_sources(gu, dw);
            cache.promote(4, 0, stream.get());
            cache.promote(1, 1, stream.get());
            CELEG_CUDA(cudaStreamSynchronize(stream.get()));
            check(cache.resident(4) && cache.expert_slot(4) == 0, "promote expert 4 -> slot 0");
            check(cache.resident(1) && cache.expert_slot(1) == 1, "promote expert 1 -> slot 1");
            check(!cache.resident(0), "expert 0 host-resident");
            // Evict then re-promote a different expert into slot 0.
            cache.evict(0, stream.get());
            CELEG_CUDA(cudaStreamSynchronize(stream.get()));
            check(!cache.resident(4), "expert 4 evicted");
            cache.promote(2, 0, stream.get());
            CELEG_CUDA(cudaStreamSynchronize(stream.get()));
            check(cache.resident(2) && cache.slot_expert(0) == 2, "promote expert 2 -> slot 0");
        }

        // ensure_resident / LRU promotion semantics.
        {
            const size_t gu_bytes =
                static_cast<size_t>(2) * p.inter * p.hidden * sizeof(__nv_bfloat16);
            const size_t dw_bytes =
                static_cast<size_t>(p.hidden) * p.inter * sizeof(__nv_bfloat16);
            const size_t gu_pe = static_cast<size_t>(2) * p.inter * p.hidden;
            const size_t dw_pe = static_cast<size_t>(p.hidden) * p.inter;
            celeg::HostExpertStore store;
            std::vector<const __nv_bfloat16*> gu(p.experts), dw(p.experts);
            for (int e = 0; e < p.experts; ++e) {
                gu[e] = static_cast<const __nv_bfloat16*>(store.store_pinned_copy(
                    p.gate_up.data() + static_cast<size_t>(e) * gu_pe, gu_bytes));
                dw[e] = static_cast<const __nv_bfloat16*>(store.store_pinned_copy(
                    p.down.data() + static_cast<size_t>(e) * dw_pe, dw_bytes));
            }
            celeg::ExpertLayerCache cache(p.experts, 2, gu_bytes, dw_bytes);
            cache.set_host_sources(gu, dw);
            celeg::CudaStream stream;
            check(cache.ensure_resident(3, stream.get()), "ensure_resident promotes cold expert 3");
            check(cache.ensure_resident(3, stream.get()) == false,
                  "ensure_resident no-op when already resident");
            check(cache.resident(3), "expert 3 resident after ensure_resident");
            check(cache.ensure_resident(1, stream.get()), "ensure_resident promotes expert 1");
            // Capacity 2 full; promoting a new expert evicts the LRU (expert 3).
            check(cache.ensure_resident(5, stream.get()), "ensure_resident promotes expert 5");
            check(!cache.resident(3), "LRU victim expert 3 evicted");
            check(cache.resident(1) && cache.resident(5), "experts 1 and 5 resident");
            CELEG_CUDA(cudaStreamSynchronize(stream.get()));
        }

        if (g_failed) {
            std::cerr << "expert_residency_test: FAILED\n";
            return 1;
        }
        std::cout << "expert_residency_test: ok\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "expert_residency_test FAILED: " << e.what() << "\n";
        return 1;
    }
}
