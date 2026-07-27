#include "lfm/runtime/moe/expert_residency.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lfm {

namespace {

// Rounds `value` down / up to a page-aligned boundary. cudaHostRegister
// requires page-aligned ranges; we register the enclosing aligned range and
// offset the returned device pointer back to the requested address.
constexpr std::size_t kPageSize = 4096;

std::uintptr_t align_down(std::uintptr_t v) { return v & ~(kPageSize - 1); }
std::size_t align_up(std::size_t v) {
    return (v + kPageSize - 1) & ~(kPageSize - 1);
}

// Device kernel: for each selected expert, check if it is GPU-resident.
// Outputs a compact list of cold experts that need promotion.
//   sel_dev[i]          = selected expert index (rows * K total)
//   expert_slot_dev[e]  = cache slot for expert e, or -1 if host-resident
//   cold_list_dev[out]  = compact list of cold expert indices
//   cold_count_dev[out] = number of cold experts (atomic, initialized to 0)
__global__ void resolve_residency_kernel(const int* sel_dev,
                                         const int* expert_slot_dev,
                                         int* cold_flags_dev,
                                         int* cold_list_dev,
                                         int* cold_count_dev,
                                         int total) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int e = sel_dev[idx];
    if (expert_slot_dev[e] < 0) {
        if (atomicCAS(&cold_flags_dev[e], 0, 1) == 0) {
            const int pos = atomicAdd(cold_count_dev, 1);
            cold_list_dev[pos] = e;
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// HostExpertStore
// ---------------------------------------------------------------------------

HostExpertStore::HostExpertStore(HostExpertStore&& other) noexcept {
    *this = std::move(other);
}

HostExpertStore& HostExpertStore::operator=(HostExpertStore&& other) noexcept {
    if (this != &other) {
        release();
        registrations_ = std::move(other.registrations_);
        registered_bytes_ = std::exchange(other.registered_bytes_, 0);
        pinned_bytes_ = std::exchange(other.pinned_bytes_, 0);
        other.registrations_.clear();
    }
    return *this;
}

HostExpertStore::~HostExpertStore() { release(); }

void HostExpertStore::release() {
    for (Registration& reg : registrations_) {
        if (reg.host_ptr == nullptr) continue;
        if (reg.mapped) {
            cudaHostUnregister(reg.host_ptr);
        } else {
            cudaFreeHost(reg.host_ptr);
        }
        reg.host_ptr = nullptr;
    }
    registrations_.clear();
    registered_bytes_ = 0;
    pinned_bytes_ = 0;
}

const void* HostExpertStore::register_mapped(const void* host_ptr,
                                             std::size_t bytes) {
    if (host_ptr == nullptr || bytes == 0) {
        throw std::invalid_argument("register_mapped: empty range");
    }
    const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(host_ptr);
    const std::uintptr_t base = align_down(addr);
    const std::size_t offset = static_cast<std::size_t>(addr - base);
    const std::size_t aligned_bytes = align_up(offset + bytes);

    void* base_ptr = reinterpret_cast<void*>(base);
    LFM_CUDA(cudaHostRegister(base_ptr, aligned_bytes,
                              cudaHostRegisterMapped | cudaHostRegisterPortable));
    void* dev_base = nullptr;
    LFM_CUDA(cudaHostGetDevicePointer(&dev_base, base_ptr, 0));

    registrations_.push_back(Registration{base_ptr, /*owned=*/false});
    registered_bytes_ += aligned_bytes;
    return reinterpret_cast<const void*>(
        reinterpret_cast<std::uintptr_t>(dev_base) + offset);
}

const void* HostExpertStore::store_pinned_copy(const void* src,
                                                std::size_t bytes) {
    if (src == nullptr || bytes == 0) {
        throw std::invalid_argument("store_pinned_copy: empty range");
    }
    void* host = nullptr;
    LFM_CUDA(cudaHostAlloc(&host, bytes,
                            cudaHostAllocMapped | cudaHostAllocPortable));
    std::memcpy(host, src, bytes);
    void* dev = nullptr;
    LFM_CUDA(cudaHostGetDevicePointer(&dev, host, 0));

    registrations_.push_back(Registration{host, /*owned=*/true, /*mapped=*/false});
    pinned_bytes_ += bytes;
    return dev;
}

const void* HostExpertStore::alloc_mapped(std::size_t bytes) {
    if (bytes == 0) {
        throw std::invalid_argument("alloc_mapped: empty range");
    }
    // A single pinned, mapped host allocation serving as the whole-layer arena.
    // cudaHostAlloc with Mapped|Portable yields a device-visible pointer, so no
    // separate cudaHostRegister call is needed (registering CUDA-allocated
    // memory is invalid).
    void* host = nullptr;
    LFM_CUDA(cudaHostAlloc(&host, bytes,
                            cudaHostAllocMapped | cudaHostAllocPortable));
    void* dev = nullptr;
    LFM_CUDA(cudaHostGetDevicePointer(&dev, host, 0));
    registrations_.push_back(Registration{host, /*owned=*/true, /*mapped=*/false});
    pinned_bytes_ += bytes;
    return dev;
}

// ---------------------------------------------------------------------------

} // namespace lfm

