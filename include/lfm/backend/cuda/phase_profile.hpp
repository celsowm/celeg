#pragma once

// Coarse GPU phase profiler for the CUDA forward passes.
//
// Why this exists: CUDA decode is captured into a single CUDA graph, and Nsight
// Compute needs elevated permissions (ERR_NVGPUCTRPERM) that are often not
// available on developer Windows boxes. Without in-tree attribution it is very
// easy to optimize the wrong thing -- this profiler was added after a plausible
// bandwidth argument pointed at the GEMV path, and immediately showed that 66%
// of decode was actually a single-threaded top-k loop in the sampler.
//
// Usage:
//   set LFM_PROFILE_DECODE=1  (any value) and run with --no-cuda-graph
//   e.g.  python scripts/profile_decode.py --model model.gguf
//
// The graph must be disabled because capture defers execution: the events would
// record capture order, not execution time. The profiler prints a warning and
// produces no output rather than reporting nonsense if that is forgotten.
//
// Cost when disabled is one predictable branch per phase boundary, so the
// instrumentation can stay in the hot path permanently.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace lfm {

// Fixed phase list. Keep names short -- they are column headers in the report.
// Order follows the decode step so the report reads top-to-bottom.
//   - AttnOut: the attention output projection (line 191 in execution.cu),
//     previously unattributed between the Attention and Mlp regions.
//   - Conv: the conv_in / conv_decode / conv_out block (conv branch of the
//     per-layer loop), previously unattributed entirely.
//   - Other: residual/unattributed bucket -- launch_increment_position and the
//     optional launch_residual_add when !options_.fused_residuals. Lets the
//     report expose work that would otherwise be hidden between regions.
enum class DecodePhase : int {
    Sampling = 0,
    Embed,
    Norm,
    Projection,
    RopeKv,
    Attention,
    AttnOut,
    Conv,
    Mlp,
    Logits,
    Other,
    kCount
};

class PhaseProfile {
public:
    PhaseProfile() {
        const char* flag = std::getenv("LFM_PROFILE_DECODE");
        enabled_ = flag != nullptr && flag[0] != '\0' && flag[0] != '0';
        if (enabled_) {
            cudaEventCreate(&begin_);
            cudaEventCreate(&end_);
        }
    }

    ~PhaseProfile() {
        if (enabled_) {
            report();
            cudaEventDestroy(begin_);
            cudaEventDestroy(end_);
        }
    }

    PhaseProfile(const PhaseProfile&) = delete;
    PhaseProfile& operator=(const PhaseProfile&) = delete;

    bool enabled() const { return enabled_; }
    void count_step() { if (enabled_) ++steps_; }

    void begin(cudaStream_t stream) {
        if (enabled_) cudaEventRecord(begin_, stream);
    }

    // Closes the region opened by begin() and attributes it to `phase`.
    // Synchronizes, so this is a measurement mode, not a production path.
    void end(DecodePhase phase, cudaStream_t stream) {
        if (!enabled_) return;
        cudaEventRecord(end_, stream);
        cudaEventSynchronize(end_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, begin_, end_);
        totals_[static_cast<int>(phase)] += ms;
    }

    void report() const {
        if (!enabled_ || steps_ == 0) {
            if (enabled_) {
                std::fprintf(stderr,
                             "[phase-profile] no steps recorded -- decode is CUDA-graph "
                             "captured; re-run with --no-cuda-graph\n");
            }
            return;
        }
        static const char* kNames[] = {"sampling", "embed", "rmsnorm", "qkv+proj",
                                       "rope+kv", "attention", "attn_out", "conv",
                                       "mlp", "logits", "other"};
        double total = 0.0;
        for (int i = 0; i < static_cast<int>(DecodePhase::kCount); ++i) total += totals_[i];
        std::fprintf(stderr, "\n=== decode phase profile (%lld steps) ===\n", steps_);
        for (int i = 0; i < static_cast<int>(DecodePhase::kCount); ++i) {
            std::fprintf(stderr, "  %-10s %8.4f ms/token  %5.1f%%\n", kNames[i],
                         totals_[i] / static_cast<double>(steps_),
                         total > 0.0 ? 100.0 * totals_[i] / total : 0.0);
        }
        std::fprintf(stderr, "  %-10s %8.4f ms/token\n", "TOTAL",
                     total / static_cast<double>(steps_));
        std::fprintf(stderr,
                     "  (phase sum excludes tiny inter-region work such as the position\n"
                     "   increment; the residual-add and conv block are now attributed.\n"
                     "   Percentages are of the phase sum, not the end-to-end benchmark\n"
                     "   ms/token, so they sum to ~100%%)\n");
    }

private:
    bool enabled_ = false;
    long long steps_ = 0;
    double totals_[static_cast<int>(DecodePhase::kCount)] = {};
    cudaEvent_t begin_{};
    cudaEvent_t end_{};
};

// Process-wide instance. Defined in execution.cu.
PhaseProfile& decode_phase_profile();

// Prefill phase profiler. Same pattern as PhaseProfile but for prefill_batched.
// Activated by LFM_PROFILE_PREFILL=1.
enum class PrefillPhase : int {
    Embed = 0,
    Norm,
    QkvProj,
    RopeKv,
    Attention,
    AttnOut,
    Conv,
    Mlp,
    Logits,
    Other,
    kCount
};

class PrefillPhaseProfile {
public:
    PrefillPhaseProfile() {
        const char* flag = std::getenv("LFM_PROFILE_PREFILL");
        enabled_ = flag != nullptr && flag[0] != '\0' && flag[0] != '0';
        if (enabled_) {
            cudaEventCreate(&begin_);
            cudaEventCreate(&end_);
        }
    }

    ~PrefillPhaseProfile() {
        if (enabled_) {
            report();
            cudaEventDestroy(begin_);
            cudaEventDestroy(end_);
        }
    }

    PrefillPhaseProfile(const PrefillPhaseProfile&) = delete;
    PrefillPhaseProfile& operator=(const PrefillPhaseProfile&) = delete;

    bool enabled() const { return enabled_; }
    void count_step() { if (enabled_) ++steps_; }

    void begin(cudaStream_t stream) {
        if (enabled_) cudaEventRecord(begin_, stream);
    }

    void end(PrefillPhase phase, cudaStream_t stream) {
        if (!enabled_) return;
        cudaEventRecord(end_, stream);
        cudaEventSynchronize(end_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, begin_, end_);
        totals_[static_cast<int>(phase)] += ms;
    }

    void report() const {
        if (!enabled_ || steps_ == 0) {
            if (enabled_) {
                std::fprintf(stderr,
                             "[prefill-profile] no prefill steps recorded\n");
            }
            return;
        }
        static const char* kNames[] = {"embed", "rmsnorm", "qkv+proj", "rope+kv",
                                       "attention", "attn_out", "conv",
                                       "mlp", "logits", "other"};
        double total = 0.0;
        for (int i = 0; i < static_cast<int>(PrefillPhase::kCount); ++i) total += totals_[i];
        std::fprintf(stderr, "\n=== prefill phase profile (%lld steps) ===\n", steps_);
        for (int i = 0; i < static_cast<int>(PrefillPhase::kCount); ++i) {
            std::fprintf(stderr, "  %-10s %8.4f ms/step  %5.1f%%\n", kNames[i],
                         totals_[i] / static_cast<double>(steps_),
                         total > 0.0 ? 100.0 * totals_[i] / total : 0.0);
        }
        std::fprintf(stderr, "  %-10s %8.4f ms/step\n", "TOTAL",
                     total / static_cast<double>(steps_));
    }

private:
    bool enabled_ = false;
    long long steps_ = 0;
    double totals_[static_cast<int>(PrefillPhase::kCount)] = {};
    cudaEvent_t begin_{};
    cudaEvent_t end_{};
};

PrefillPhaseProfile& prefill_phase_profile();

} // namespace lfm
