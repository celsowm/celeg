#include "sampling_tests.hpp"

#include "celeg/backend/cuda/utils.cuh"
#include "../support/assertions.hpp"
#include "../support/cuda_kernel_assertions.cuh"
#include "celeg/backend/cuda/kernels/kernels.cuh"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace celeg::cuda_test {

void run_sampling_tests(celeg::CudaStream& stream) {
// Greedy argmax.
{
    std::vector<__nv_bfloat16> logits = {
        to_bf16(-1.0f), to_bf16(8.0f), to_bf16(8.0f), to_bf16(2.0f)};
    celeg::DeviceBuffer<__nv_bfloat16> device_logits(logits.size());
    std::vector<uint8_t> seen(logits.size(), 0);
    celeg::DeviceBuffer<uint8_t> device_seen(seen.size());
    celeg::DeviceBuffer<int32_t> result(1);
    CELEG_CUDA(cudaMemcpy(device_logits.data(), logits.data(),
                        device_logits.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(device_seen.data(), seen.data(), device_seen.bytes(),
                          cudaMemcpyHostToDevice));
    celeg::launch_argmax_bf16(device_logits.data(), device_seen.data(),
                              static_cast<int>(logits.size()), 1.0f,
                              result.data(), stream.get());
    int32_t index = -1;
    CELEG_CUDA(cudaMemcpyAsync(&index, result.data(), sizeof(index),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    CELEG_TEST_CHECK(index == 1);
}

// Greedy argmax applies repetition penalty without the top-k path.
{
    std::vector<__nv_bfloat16> logits = {
        to_bf16(5.0f), to_bf16(4.0f), to_bf16(3.0f), to_bf16(2.0f)};
    std::vector<uint8_t> seen = {1, 0, 0, 0};
    celeg::DeviceBuffer<__nv_bfloat16> device_logits(logits.size());
    celeg::DeviceBuffer<uint8_t> device_seen(seen.size());
    celeg::DeviceBuffer<int32_t> result(1);
    CELEG_CUDA(cudaMemcpy(device_logits.data(), logits.data(), device_logits.bytes(),
                          cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(device_seen.data(), seen.data(), device_seen.bytes(),
                          cudaMemcpyHostToDevice));
    celeg::launch_argmax_bf16(device_logits.data(), device_seen.data(),
                              static_cast<int>(logits.size()), 2.0f,
                              result.data(), stream.get());
    int32_t index = -1;
    CELEG_CUDA(cudaMemcpyAsync(&index, result.data(), sizeof(index),
                               cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    CELEG_TEST_CHECK(index == 1);
}



// GPU repetition penalty + top-k=1 sampling selects the adjusted maximum.
{
    std::vector<__nv_bfloat16> logits = {
        to_bf16(5.0f), to_bf16(4.0f), to_bf16(3.0f), to_bf16(2.0f)};
    std::vector<uint8_t> seen = {1, 0, 0, 0};
    uint64_t seed = 123;
    celeg::DeviceBuffer<__nv_bfloat16> dlogits(logits.size());
    celeg::DeviceBuffer<uint8_t> dseen(seen.size());
    celeg::DeviceBuffer<float> scores(logits.size()), values(1);
    celeg::DeviceBuffer<int32_t> indices(1), result(1);
    celeg::DeviceBuffer<uint64_t> rng(1);
    CELEG_CUDA(cudaMemcpy(dlogits.data(), logits.data(), dlogits.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dseen.data(), seen.data(), dseen.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(rng.data(), &seed, sizeof(seed), cudaMemcpyHostToDevice));
    celeg::launch_prepare_sampling_scores(dlogits.data(), dseen.data(), scores.data(),
                                        4, 1.0f, 2.0f, stream.get());
    celeg::launch_select_topk(scores.data(), values.data(), indices.data(), 0, 4, stream.get());
    celeg::launch_sample_topk(values.data(), indices.data(), 1, 1.0f,
                            rng.data(), result.data(), stream.get());
    int32_t token = -1;
    CELEG_CUDA(cudaMemcpyAsync(&token, result.data(), sizeof(token),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    CELEG_TEST_CHECK(token == 1); // token 0 is penalized from 5 to 2.5
}



// Top-p truncation within top-k is deterministic for a fixed seed.
{
    std::vector<float> values = {3.0f, 2.0f, 1.0f};
    std::vector<int32_t> indices = {10, 20, 30};
    uint64_t seed = 3;
    celeg::DeviceBuffer<float> dvalues(values.size());
    celeg::DeviceBuffer<int32_t> dindices(indices.size()), result(1);
    celeg::DeviceBuffer<uint64_t> rng(1);
    CELEG_CUDA(cudaMemcpy(dvalues.data(), values.data(), dvalues.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dindices.data(), indices.data(), dindices.bytes(), cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(rng.data(), &seed, sizeof(seed), cudaMemcpyHostToDevice));
    celeg::launch_sample_topk(dvalues.data(), dindices.data(), 3, 0.7f,
                            rng.data(), result.data(), stream.get());
    int32_t token = -1;
    CELEG_CUDA(cudaMemcpyAsync(&token, result.data(), sizeof(token),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    CELEG_TEST_CHECK(token == 20);
}



// Fused sampler matches the reference multi-launch pipeline.
{
    std::vector<__nv_bfloat16> logits = {
        to_bf16(4.0f), to_bf16(3.0f), to_bf16(2.0f),
        to_bf16(1.0f), to_bf16(0.0f), to_bf16(-1.0f)};
    std::vector<uint8_t> seen = {1, 0, 0, 0, 0, 0};
    constexpr int top_k = 4;
    constexpr float top_p = 0.85f;
    constexpr float temperature = 0.75f;
    constexpr float penalty = 1.1f;
    uint64_t reference_seed = 77;
    uint64_t fused_seed = reference_seed;

    celeg::DeviceBuffer<__nv_bfloat16> dlogits(logits.size());
    celeg::DeviceBuffer<uint8_t> dseen(seen.size());
    celeg::DeviceBuffer<float> reference_scores(logits.size());
    celeg::DeviceBuffer<float> fused_scores(logits.size());
    celeg::DeviceBuffer<float> reference_values(top_k), fused_values(top_k);
    celeg::DeviceBuffer<int32_t> reference_indices(top_k), fused_indices(top_k);
    celeg::DeviceBuffer<int32_t> reference_result(1), fused_result(1);
    celeg::DeviceBuffer<uint64_t> reference_rng(1), fused_rng(1);
    CELEG_CUDA(cudaMemcpy(dlogits.data(), logits.data(), dlogits.bytes(),
                        cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dseen.data(), seen.data(), dseen.bytes(),
                        cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(reference_rng.data(), &reference_seed, sizeof(reference_seed),
                        cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(fused_rng.data(), &fused_seed, sizeof(fused_seed),
                        cudaMemcpyHostToDevice));

    celeg::launch_prepare_sampling_scores(
        dlogits.data(), dseen.data(), reference_scores.data(),
        static_cast<int>(logits.size()), temperature, penalty, stream.get());
    for (int rank = 0; rank < top_k; ++rank) {
        celeg::launch_select_topk(reference_scores.data(), reference_values.data(),
                                reference_indices.data(), rank,
                                static_cast<int>(logits.size()), stream.get());
    }
    celeg::launch_sample_topk(reference_values.data(), reference_indices.data(),
                            top_k, top_p, reference_rng.data(),
                            reference_result.data(), stream.get());

    celeg::launch_fused_sample_topk(
        dlogits.data(), dseen.data(), fused_scores.data(),
        fused_values.data(), fused_indices.data(),
        static_cast<int>(logits.size()), temperature, penalty,
        top_k, top_p, fused_rng.data(), fused_result.data(), stream.get());

    int32_t reference_token = -1;
    int32_t fused_token = -1;
    CELEG_CUDA(cudaMemcpyAsync(&reference_token, reference_result.data(), sizeof(reference_token),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(&fused_token, fused_result.data(), sizeof(fused_token),
                             cudaMemcpyDeviceToHost, stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));
    CELEG_TEST_CHECK(reference_token == fused_token);
}

// Fused sampler produces the exact ordered top-k at realistic vocabulary
// size, including the tie-break rule.
//
// The check above only compares the finally sampled token on a 6-entry
// vocabulary, which cannot detect a mis-ordered top-k array. The selection
// is a block-parallel argmax drain, so it has to reproduce the ordering the
// original single-threaded insertion sort produced -- descending score, and
// on an exact tie the *lower* vocabulary index first. Duplicated logits below
// force many exact ties (bf16 has only 8 mantissa bits, so ties are common in
// practice, not a synthetic worry), and they are seeded across the whole
// vocabulary so ties land in different threads' strided slices.
{
    constexpr int vocab = 65536;
    constexpr int top_k = 50;
    std::vector<__nv_bfloat16> logits(vocab);
    std::vector<uint8_t> seen(vocab, 0);
    std::mt19937 rng(20260729);
    std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
    for (int i = 0; i < vocab; ++i) logits[i] = to_bf16(dist(rng));
    // Plant a plateau of exactly-equal top values spread far apart, so the
    // correct answer is "these indices, in ascending order".
    const int tie_positions[] = {5, 999, 1024, 1025, 40000, 65535, 257, 258};
    for (int p : tie_positions) logits[p] = to_bf16(9.5f);
    for (int i = 0; i < vocab; i += 977) seen[i] = 1;

    constexpr float temperature = 0.75f;
    constexpr float penalty = 1.1f;

    // CPU reference: the exact semantics of the replaced insertion sort.
    std::vector<float> ref_scores(vocab);
    for (int i = 0; i < vocab; ++i) {
        float v = __bfloat162float(logits[i]);
        if (seen[i]) v = v < 0.0f ? v * penalty : v / penalty;
        ref_scores[i] = v / temperature;
    }
    std::vector<int> order(vocab);
    for (int i = 0; i < vocab; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (ref_scores[a] != ref_scores[b]) return ref_scores[a] > ref_scores[b];
        return a < b;
    });

    celeg::DeviceBuffer<__nv_bfloat16> dlogits(vocab);
    celeg::DeviceBuffer<uint8_t> dseen(vocab);
    celeg::DeviceBuffer<float> dscores(vocab);
    celeg::DeviceBuffer<float> dvalues(top_k);
    celeg::DeviceBuffer<int32_t> dindices(top_k);
    celeg::DeviceBuffer<int32_t> dresult(1);
    celeg::DeviceBuffer<uint64_t> drng(1);
    uint64_t seed = 4242;
    CELEG_CUDA(cudaMemcpy(dlogits.data(), logits.data(), dlogits.bytes(),
                        cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(dseen.data(), seen.data(), dseen.bytes(),
                        cudaMemcpyHostToDevice));
    CELEG_CUDA(cudaMemcpy(drng.data(), &seed, sizeof(seed), cudaMemcpyHostToDevice));

    celeg::launch_fused_sample_topk(
        dlogits.data(), dseen.data(), dscores.data(), dvalues.data(),
        dindices.data(), vocab, temperature, penalty, top_k, 1.0f,
        drng.data(), dresult.data(), stream.get());

    std::vector<int32_t> got_indices(top_k);
    std::vector<float> got_values(top_k);
    CELEG_CUDA(cudaMemcpyAsync(got_indices.data(), dindices.data(),
                             top_k * sizeof(int32_t), cudaMemcpyDeviceToHost,
                             stream.get()));
    CELEG_CUDA(cudaMemcpyAsync(got_values.data(), dvalues.data(),
                             top_k * sizeof(float), cudaMemcpyDeviceToHost,
                             stream.get()));
    CELEG_CUDA(cudaStreamSynchronize(stream.get()));

    for (int r = 0; r < top_k; ++r) {
        CELEG_TEST_CHECK(got_indices[r] == order[r]);
        CELEG_TEST_CHECK(got_values[r] == ref_scores[order[r]]);
    }
    // The planted plateau must come out first, in ascending index order.
    std::vector<int> ties(std::begin(tie_positions), std::end(tie_positions));
    std::sort(ties.begin(), ties.end());
    for (size_t t = 0; t < ties.size(); ++t) {
        CELEG_TEST_CHECK(got_indices[t] == ties[t]);
    }
}


}

} // namespace celeg::cuda_test
