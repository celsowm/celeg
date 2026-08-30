#pragma once

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <stdexcept>
#include <string>
#include <utility>

namespace celeg {

enum class CudaMemoryKind : std::uint8_t {
    Device,
    Managed,
};

inline thread_local bool g_cuda_managed_weight_allocations = false;

class CudaManagedWeightAllocationScope {
public:
    explicit CudaManagedWeightAllocationScope(bool enabled)
        : previous_(g_cuda_managed_weight_allocations) {
        g_cuda_managed_weight_allocations = enabled;
    }
    ~CudaManagedWeightAllocationScope() {
        g_cuda_managed_weight_allocations = previous_;
    }

private:
    bool previous_;
};

inline CudaMemoryKind cuda_current_memory_kind() noexcept {
    return g_cuda_managed_weight_allocations
        ? CudaMemoryKind::Managed
        : CudaMemoryKind::Device;
}

struct CudaAllocationSnapshot {
    size_t device_allocations = 0;
    size_t device_bytes = 0;
    size_t host_allocations = 0;
    size_t host_bytes = 0;
};

namespace detail {
inline std::atomic<size_t> cuda_device_allocations{0};
inline std::atomic<size_t> cuda_device_bytes{0};
inline std::atomic<size_t> cuda_host_allocations{0};
inline std::atomic<size_t> cuda_host_bytes{0};
}

inline CudaAllocationSnapshot cuda_allocation_snapshot() {
    return {
        detail::cuda_device_allocations.load(std::memory_order_relaxed),
        detail::cuda_device_bytes.load(std::memory_order_relaxed),
        detail::cuda_host_allocations.load(std::memory_order_relaxed),
        detail::cuda_host_bytes.load(std::memory_order_relaxed),
    };
}

class CudaAllocationScope {
public:
    CudaAllocationScope() : begin_(cuda_allocation_snapshot()) {}

    CudaAllocationSnapshot delta() const {
        const CudaAllocationSnapshot end = cuda_allocation_snapshot();
        return {
            end.device_allocations - begin_.device_allocations,
            end.device_bytes - begin_.device_bytes,
            end.host_allocations - begin_.host_allocations,
            end.host_bytes - begin_.host_bytes,
        };
    }

private:
    CudaAllocationSnapshot begin_;
};

inline void record_cuda_device_allocation(size_t bytes) {
    detail::cuda_device_allocations.fetch_add(1, std::memory_order_relaxed);
    detail::cuda_device_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

inline void record_cuda_host_allocation(size_t bytes) {
    detail::cuda_host_allocations.fetch_add(1, std::memory_order_relaxed);
    detail::cuda_host_bytes.fetch_add(bytes, std::memory_order_relaxed);
}

inline void check_cuda(cudaError_t status, const char* call) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
    }
}

inline void check_cublas(cublasStatus_t status, const char* call) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(call) + " failed with status " + std::to_string(status));
    }
}

#define CELEG_CUDA(call) ::celeg::check_cuda((call), #call)
#define CELEG_CUBLAS(call) ::celeg::check_cublas((call), #call)
#define CELEG_CUDA_STRINGIFY_IMPL(value) #value
#define CELEG_CUDA_STRINGIFY(value) CELEG_CUDA_STRINGIFY_IMPL(value)
#define CELEG_KERNEL_CHECK() ::celeg::check_cuda(cudaPeekAtLastError(), "CUDA kernel launch at " __FILE__ ":" CELEG_CUDA_STRINGIFY(__LINE__))

inline void cuda_prefer_managed_host(const void* pointer, size_t bytes) {
    cudaMemLocation location{};
    location.type = cudaMemLocationTypeHost;
    location.id = 0;
    CELEG_CUDA(cudaMemAdvise(pointer, bytes, cudaMemAdviseSetPreferredLocation, location));
}

#if defined(CELEG_DEBUG_SYNC) && CELEG_DEBUG_SYNC
#define CELEG_KERNEL_DEBUG_SYNC(stream)                    \
    do {                                                 \
        CELEG_KERNEL_CHECK();                              \
        ::celeg::check_cuda(cudaStreamSynchronize(stream), \
                          "cudaStreamSynchronize");     \
    } while (false)
#else
#define CELEG_KERNEL_DEBUG_SYNC(stream) \
    do {                              \
        (void)(stream);               \
        CELEG_KERNEL_CHECK();           \
    } while (false)
#endif

class CudaStream {
public:
    CudaStream() { CELEG_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~CudaStream() {
        if (stream_) cudaStreamDestroy(stream_);
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept : stream_(std::exchange(other.stream_, nullptr)) {}
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) cudaStreamDestroy(stream_);
            stream_ = std::exchange(other.stream_, nullptr);
        }
        return *this;
    }

    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

class CudaEvent {
public:
    explicit CudaEvent(unsigned flags = cudaEventDefault) {
        CELEG_CUDA(cudaEventCreateWithFlags(&event_, flags));
    }
    ~CudaEvent() {
        if (event_) cudaEventDestroy(event_);
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    CudaEvent(CudaEvent&& other) noexcept
        : event_(std::exchange(other.event_, nullptr)) {}
    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) cudaEventDestroy(event_);
            event_ = std::exchange(other.event_, nullptr);
        }
        return *this;
    }

    void record(cudaStream_t stream) { CELEG_CUDA(cudaEventRecord(event_, stream)); }
    void synchronize() { CELEG_CUDA(cudaEventSynchronize(event_)); }
    cudaEvent_t get() const { return event_; }

    static float elapsed_ms(const CudaEvent& start, const CudaEvent& stop) {
        float result = 0.0f;
        CELEG_CUDA(cudaEventElapsedTime(&result, start.get(), stop.get()));
        return result;
    }

private:
    cudaEvent_t event_ = nullptr;
};

class CublasLtHandle {
public:
    CublasLtHandle() { CELEG_CUBLAS(cublasLtCreate(&handle_)); }
    ~CublasLtHandle() {
        if (handle_) cublasLtDestroy(handle_);
    }

    CublasLtHandle(const CublasLtHandle&) = delete;
    CublasLtHandle& operator=(const CublasLtHandle&) = delete;
    CublasLtHandle(CublasLtHandle&& other) noexcept
        : handle_(std::exchange(other.handle_, nullptr)) {}
    CublasLtHandle& operator=(CublasLtHandle&& other) noexcept {
        if (this != &other) {
            if (handle_) cublasLtDestroy(handle_);
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    cublasLtHandle_t get() const { return handle_; }
    operator cublasLtHandle_t() const { return handle_; }

private:
    cublasLtHandle_t handle_ = nullptr;
};

class CublasHandle {
public:
    explicit CublasHandle(cudaStream_t stream) {
        CELEG_CUBLAS(cublasCreate(&handle_));
        try {
            CELEG_CUBLAS(cublasSetStream(handle_, stream));
            CELEG_CUBLAS(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH));
        } catch (...) {
            cublasDestroy(handle_);
            handle_ = nullptr;
            throw;
        }
    }

    ~CublasHandle() {
        if (handle_) cublasDestroy(handle_);
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    CublasHandle(CublasHandle&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    CublasHandle& operator=(CublasHandle&& other) noexcept {
        if (this != &other) {
            if (handle_) cublasDestroy(handle_);
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    cublasHandle_t get() const { return handle_; }
    operator cublasHandle_t() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : memory_kind_(cuda_current_memory_kind()) {}
    explicit DeviceBuffer(CudaMemoryKind memory_kind) : memory_kind_(memory_kind) {}
    explicit DeviceBuffer(size_t count) : DeviceBuffer() { reset(count); }
    DeviceBuffer(size_t count, CudaMemoryKind memory_kind)
        : memory_kind_(memory_kind) {
        reset(count);
    }
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)),
          count_(std::exchange(other.count_, 0)),
          memory_kind_(other.memory_kind_) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = std::exchange(other.ptr_, nullptr);
            count_ = std::exchange(other.count_, 0);
            memory_kind_ = other.memory_kind_;
        }
        return *this;
    }

    void reset(size_t count) {
        if (count == count_) return;
        if (ptr_) CELEG_CUDA(cudaFree(ptr_));
        ptr_ = nullptr;
        count_ = count;
        if (count_) {
            if (memory_kind_ == CudaMemoryKind::Managed) {
                CELEG_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&ptr_),
                    count_ * sizeof(T), cudaMemAttachGlobal));
                cuda_prefer_managed_host(ptr_, count_ * sizeof(T));
            } else {
                CELEG_CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
            }
            record_cuda_device_allocation(count_ * sizeof(T));
        }
    }

    void reserve(size_t count) {
        if (count <= count_) return;
        if (ptr_) CELEG_CUDA(cudaFree(ptr_));
        ptr_ = nullptr;
        count_ = count;
        if (count_) {
            if (memory_kind_ == CudaMemoryKind::Managed) {
                CELEG_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&ptr_),
                    count_ * sizeof(T), cudaMemAttachGlobal));
                cuda_prefer_managed_host(ptr_, count_ * sizeof(T));
            } else {
                CELEG_CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
            }
            record_cuda_device_allocation(count_ * sizeof(T));
        }
    }

    void zero_async(cudaStream_t stream) {
        if (ptr_ && count_) CELEG_CUDA(cudaMemsetAsync(ptr_, 0, bytes(), stream));
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return count_; }
    size_t bytes() const { return count_ * sizeof(T); }
    bool empty() const { return count_ == 0; }
    CudaMemoryKind memory_kind() const noexcept { return memory_kind_; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
    CudaMemoryKind memory_kind_ = CudaMemoryKind::Device;
};

template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    explicit PinnedBuffer(size_t count) { reset(count); }
    ~PinnedBuffer() {
        if (ptr_) cudaFreeHost(ptr_);
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)), count_(std::exchange(other.count_, 0)) {}
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFreeHost(ptr_);
            ptr_ = std::exchange(other.ptr_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    void reset(size_t count) {
        if (count == count_) return;
        if (ptr_) CELEG_CUDA(cudaFreeHost(ptr_));
        ptr_ = nullptr;
        count_ = count;
        if (count_) {
            CELEG_CUDA(cudaMallocHost(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
            record_cuda_host_allocation(count_ * sizeof(T));
        }
    }

    void reserve(size_t count) {
        if (count <= count_) return;
        if (ptr_) CELEG_CUDA(cudaFreeHost(ptr_));
        ptr_ = nullptr;
        count_ = count;
        if (count_) {
            CELEG_CUDA(cudaMallocHost(reinterpret_cast<void**>(&ptr_), count_ * sizeof(T)));
            record_cuda_host_allocation(count_ * sizeof(T));
        }
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return count_; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

class CudaGraphExec {
public:
    CudaGraphExec() = default;
    ~CudaGraphExec() {
        reset();
    }

    CudaGraphExec(const CudaGraphExec&) = delete;
    CudaGraphExec& operator=(const CudaGraphExec&) = delete;

    void reset() noexcept {
        if (exec_) cudaGraphExecDestroy(exec_);
        if (graph_) cudaGraphDestroy(graph_);
        exec_ = nullptr;
        graph_ = nullptr;
    }

    void abort_capture(cudaStream_t stream) noexcept {
        if (!capturing_) return;
        cudaGraph_t discarded = nullptr;
        (void)cudaStreamEndCapture(stream, &discarded);
        if (discarded) cudaGraphDestroy(discarded);
        (void)cudaGetLastError();
        capturing_ = false;
    }

    void capture_begin(cudaStream_t stream) {
        if (capturing_) throw std::runtime_error("CUDA graph capture is already active");
        reset();
        CELEG_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        capturing_ = true;
    }

    void capture_end(cudaStream_t stream) {
        cudaGraph_t captured = nullptr;
        const cudaError_t status = cudaStreamEndCapture(stream, &captured);
        capturing_ = false;
        check_cuda(status, "cudaStreamEndCapture");
        graph_ = captured;
        try {
            CELEG_CUDA(cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0));
        } catch (...) {
            reset();
            throw;
        }
    }

    void launch(cudaStream_t stream) const {
        if (!exec_) throw std::runtime_error("CUDA graph has not been instantiated");
        CELEG_CUDA(cudaGraphLaunch(exec_, stream));
    }

    bool ready() const { return exec_ != nullptr; }

private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    bool capturing_ = false;
};

}
