#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]


SOURCE = r'''
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <iostream>

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "no Metal device\n";
            return 1;
        }
        std::cout << "device=" << device.name.UTF8String << "\n";
        std::cout << "stage="
                  << ([device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary] ? "yes" : "no")
                  << "\n";
        std::cout << "draw="
                  << ([device supportsCounterSampling:MTLCounterSamplingPointAtDrawBoundary] ? "yes" : "no")
                  << "\n";
        std::cout << "blit="
                  << ([device supportsCounterSampling:MTLCounterSamplingPointAtBlitBoundary] ? "yes" : "no")
                  << "\n";
        std::cout << "dispatch="
                  << ([device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary] ? "yes" : "no")
                  << "\n";
        std::cout << "tile_dispatch="
                  << ([device supportsCounterSampling:MTLCounterSamplingPointAtTileDispatchBoundary] ? "yes" : "no")
                  << "\n";
        for (id<MTLCounterSet> set in device.counterSets) {
            std::cout << "counter_set=" << set.name.UTF8String << "\n";
        }
    }
    return 0;
}
'''


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="celeg-metal-counters-") as directory:
        root = pathlib.Path(directory)
        source = root / "counter_capabilities.mm"
        binary = root / "counter-capabilities"
        source.write_text(SOURCE, encoding="utf-8")
        subprocess.run([
            "xcrun", "--sdk", "macosx", "clang++",
            "-std=c++20", "-fobjc-arc", str(source),
            "-framework", "Foundation", "-framework", "Metal",
            "-o", str(binary),
        ], cwd=ROOT, check=True)
        subprocess.run([str(binary)], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
