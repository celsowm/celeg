#pragma once


#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace celeg {

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
        const char* flag = std::getenv("CELEG_PROFILE_DECODE");
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

PhaseProfile& decode_phase_profile();

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
        const char* flag = std::getenv("CELEG_PROFILE_PREFILL");
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

}
