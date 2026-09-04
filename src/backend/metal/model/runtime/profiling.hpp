#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace celeg::metal_model_detail {

enum class DispatchProfileMode : uint8_t {
    Off,
    Counts,
    GpuStage,
};

struct DispatchKernelProfile {
    std::string kernel;
    uint64_t count = 0;
    std::vector<double> samples_ms;
};

struct DispatchProfileReport {
    DispatchProfileMode mode = DispatchProfileMode::Off;
    std::vector<DispatchKernelProfile> kernels;
    double gpu_execution_ms = 0.0;
};

void set_dispatch_profile_mode(DispatchProfileMode mode);
DispatchProfileMode dispatch_profile_mode();
void reset_dispatch_profile();
void record_dispatch_count(std::string_view kernel);
void record_dispatch_gpu_time(std::string_view kernel, double milliseconds);
DispatchProfileReport take_dispatch_profile(double gpu_execution_ms);

}
