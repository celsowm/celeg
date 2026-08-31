#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>

namespace celeg {

namespace detail {

inline bool phase_profile_enabled(const char* environment_variable) {
    const char* flag = std::getenv(environment_variable);
    return flag != nullptr && flag[0] != '\0' && flag[0] != '0';
}

template <typename Phase, size_t PhaseCount>
class CudaPhaseAccumulator {
public:
    explicit CudaPhaseAccumulator(const char* environment_variable)
        : enabled_(phase_profile_enabled(environment_variable)) {
        if (enabled_) {
            cudaEventCreate(&begin_);
            cudaEventCreate(&end_);
        }
    }

    ~CudaPhaseAccumulator() {
        if (enabled_) {
            cudaEventDestroy(begin_);
            cudaEventDestroy(end_);
        }
    }

    CudaPhaseAccumulator(const CudaPhaseAccumulator&) = delete;
    CudaPhaseAccumulator& operator=(const CudaPhaseAccumulator&) = delete;

    bool enabled() const { return enabled_; }

    void count_step() {
        if (enabled_) ++steps_;
    }

    void begin(cudaStream_t stream) {
        if (enabled_) cudaEventRecord(begin_, stream);
    }

    void end(Phase phase, cudaStream_t stream) {
        if (!enabled_) return;
        cudaEventRecord(end_, stream);
        cudaEventSynchronize(end_);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, begin_, end_);
        totals_[static_cast<size_t>(phase)] += milliseconds;
    }

    long long steps() const { return steps_; }
    const std::array<double, PhaseCount>& totals() const { return totals_; }

    double total() const {
        double result = 0.0;
        for (double value : totals_) result += value;
        return result;
    }

private:
    bool enabled_ = false;
    long long steps_ = 0;
    std::array<double, PhaseCount> totals_{};
    cudaEvent_t begin_{};
    cudaEvent_t end_{};
};

template <typename Phase, size_t PhaseCount>
void report_phase_profile(
    const CudaPhaseAccumulator<Phase, PhaseCount>& accumulator,
    const char* title,
    const char* unit,
    const std::array<const char*, PhaseCount>& names,
    const char* no_steps_message) {
    if (!accumulator.enabled() || accumulator.steps() == 0) {
        if (accumulator.enabled()) {
            std::fprintf(stderr, "%s", no_steps_message);
        }
        return;
    }

    const auto& totals = accumulator.totals();
    const double total = accumulator.total();
    const double steps = static_cast<double>(accumulator.steps());

    std::fprintf(stderr, "\n=== %s (%lld steps) ===\n", title, accumulator.steps());
    for (size_t i = 0; i < names.size(); ++i) {
        std::fprintf(stderr, "  %-10s %8.4f %s  %5.1f%%\n", names[i],
                     totals[i] / steps, unit,
                     total > 0.0 ? 100.0 * totals[i] / total : 0.0);
    }
    std::fprintf(stderr, "  %-10s %8.4f %s\n", "TOTAL", total / steps, unit);
}

}  // namespace detail

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
    PhaseProfile() : accumulator_("CELEG_PROFILE_DECODE") {}

    ~PhaseProfile() {
        if (enabled()) report();
    }

    PhaseProfile(const PhaseProfile&) = delete;
    PhaseProfile& operator=(const PhaseProfile&) = delete;

    bool enabled() const { return accumulator_.enabled(); }
    void count_step() { accumulator_.count_step(); }
    void begin(cudaStream_t stream) { accumulator_.begin(stream); }
    void end(DecodePhase phase, cudaStream_t stream) {
        accumulator_.end(phase, stream);
    }

    void report() const {
        static constexpr std::array<const char*,
                                    static_cast<size_t>(DecodePhase::kCount)>
            kNames{"sampling", "embed", "rmsnorm", "qkv+proj", "rope+kv",
                   "attention", "attn_out", "conv", "mlp", "logits", "other"};
        detail::report_phase_profile(
            accumulator_, "decode phase profile", "ms/token", kNames,
            "[phase-profile] no steps recorded -- decode is CUDA-graph captured; "
            "re-run with --no-cuda-graph\n");
        if (enabled() && accumulator_.steps() != 0) {
            std::fprintf(stderr,
                         "  (phase sum excludes tiny inter-region work such as the position\n"
                         "   increment; the residual-add and conv block are now attributed.\n"
                         "   Percentages are of the phase sum, not the end-to-end benchmark\n"
                         "   ms/token, so they sum to ~100%%)\n");
        }
    }

private:
    detail::CudaPhaseAccumulator<DecodePhase,
                                 static_cast<size_t>(DecodePhase::kCount)>
        accumulator_;
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
    PrefillPhaseProfile() : accumulator_("CELEG_PROFILE_PREFILL") {}

    ~PrefillPhaseProfile() {
        if (enabled()) report();
    }

    PrefillPhaseProfile(const PrefillPhaseProfile&) = delete;
    PrefillPhaseProfile& operator=(const PrefillPhaseProfile&) = delete;

    bool enabled() const { return accumulator_.enabled(); }
    void count_step() { accumulator_.count_step(); }
    void begin(cudaStream_t stream) { accumulator_.begin(stream); }
    void end(PrefillPhase phase, cudaStream_t stream) {
        accumulator_.end(phase, stream);
    }

    void report() const {
        static constexpr std::array<const char*,
                                    static_cast<size_t>(PrefillPhase::kCount)>
            kNames{"embed", "rmsnorm", "qkv+proj", "rope+kv", "attention",
                   "attn_out", "conv", "mlp", "logits", "other"};
        detail::report_phase_profile(
            accumulator_, "prefill phase profile", "ms/step", kNames,
            "[prefill-profile] no prefill steps recorded\n");
    }

private:
    detail::CudaPhaseAccumulator<PrefillPhase,
                                 static_cast<size_t>(PrefillPhase::kCount)>
        accumulator_;
};

PrefillPhaseProfile& prefill_phase_profile();

}  // namespace celeg
