#include "celeg/backend/metal/device.hpp"

namespace celeg {

bool metal_backend_is_ready() noexcept {
    return MetalDevice::available();
}

}
