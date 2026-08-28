#pragma once

#include "celeg/backend/metal/runtime_types.hpp"

#include <memory>

namespace celeg {

class MetalDevice {
public:
    MetalDevice();
    ~MetalDevice();

    MetalDevice(const MetalDevice&) = delete;
    MetalDevice& operator=(const MetalDevice&) = delete;
    MetalDevice(MetalDevice&&) noexcept;
    MetalDevice& operator=(MetalDevice&&) noexcept;

    const MetalCapabilities& capabilities() const noexcept;
    void run_probe() const;

    static bool available() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
