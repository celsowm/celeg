#include "celeg/backend/metal/device.hpp"

#include <iostream>

int main() {
    try {
        celeg::MetalDevice device;
        device.run_probe();
        std::cout << device.capabilities().summary() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
