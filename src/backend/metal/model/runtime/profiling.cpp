#include "profiling.hpp"

#include <algorithm>
#include <unordered_map>

namespace celeg::metal_model_detail {
namespace {

struct DispatchProfileState {
    DispatchProfileMode mode = DispatchProfileMode::Off;
    std::unordered_map<std::string, DispatchKernelProfile> kernels;
};

thread_local DispatchProfileState gDispatchProfile;

}

void set_dispatch_profile_mode(DispatchProfileMode mode) {
    gDispatchProfile.mode = mode;
    gDispatchProfile.kernels.clear();
}

DispatchProfileMode dispatch_profile_mode() {
    return gDispatchProfile.mode;
}

void reset_dispatch_profile() {
    gDispatchProfile.kernels.clear();
}

void record_dispatch_count(std::string_view kernel) {
    if (gDispatchProfile.mode == DispatchProfileMode::Off) return;
    auto [entry, inserted] = gDispatchProfile.kernels.try_emplace(std::string(kernel));
    if (inserted) entry->second.kernel = entry->first;
    ++entry->second.count;
}

void record_dispatch_gpu_time(std::string_view kernel, double milliseconds) {
    if (gDispatchProfile.mode != DispatchProfileMode::GpuStage) return;
    auto [entry, inserted] = gDispatchProfile.kernels.try_emplace(std::string(kernel));
    if (inserted) entry->second.kernel = entry->first;
    entry->second.samples_ms.push_back(milliseconds);
}

DispatchProfileReport take_dispatch_profile(double gpu_execution_ms) {
    DispatchProfileReport result;
    result.mode = gDispatchProfile.mode;
    result.gpu_execution_ms = gpu_execution_ms;
    result.kernels.reserve(gDispatchProfile.kernels.size());
    for (auto& [name, profile] : gDispatchProfile.kernels) {
        static_cast<void>(name);
        result.kernels.push_back(std::move(profile));
    }
    std::sort(result.kernels.begin(), result.kernels.end(),
              [](const DispatchKernelProfile& left, const DispatchKernelProfile& right) {
                  return left.kernel < right.kernel;
              });
    gDispatchProfile.kernels.clear();
    return result;
}

}
