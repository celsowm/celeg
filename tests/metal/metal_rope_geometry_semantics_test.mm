#include "celeg/model/rope_geometry.hpp"
#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

struct Case {
    uint32_t pair;
    uint32_t pair_count;
    celeg::RopePairingKind pairing;
    uint32_t section0;
    uint32_t section1;
    bool interleaved;
};

void run_case(id<MTLDevice> device,
              id<MTLComputePipelineState> state,
              id<MTLCommandQueue> queue,
              const Case& test_case) {
    id<MTLBuffer> components = [device newBufferWithLength:2 * sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];
    id<MTLBuffer> axis = [device newBufferWithLength:sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];
    const uint32_t pairing_mode =
        test_case.pairing == celeg::RopePairingKind::AdjacentPairs ? 1u : 0u;
    const uint32_t interleaved = test_case.interleaved ? 1u : 0u;

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:components offset:0 atIndex:0];
    [encoder setBuffer:axis offset:0 atIndex:1];
    [encoder setBytes:&test_case.pair length:sizeof(test_case.pair) atIndex:2];
    [encoder setBytes:&test_case.pair_count length:sizeof(test_case.pair_count) atIndex:3];
    [encoder setBytes:&pairing_mode length:sizeof(pairing_mode) atIndex:4];
    [encoder setBytes:&test_case.section0 length:sizeof(test_case.section0) atIndex:5];
    [encoder setBytes:&test_case.section1 length:sizeof(test_case.section1) atIndex:6];
    [encoder setBytes:&interleaved length:sizeof(interleaved) atIndex:7];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal RoPE geometry dispatch failed");
    }

    const auto expected_components = celeg::rope_geometry::pair_components(
        static_cast<int>(test_case.pair),
        static_cast<int>(test_case.pair_count),
        test_case.pairing);
    const int expected_axis = celeg::rope_geometry::mrope_axis_for_pair(
        static_cast<int>(test_case.pair),
        static_cast<int>(test_case.section0),
        static_cast<int>(test_case.section1),
        test_case.interleaved);
    const auto* actual_components = static_cast<const uint32_t*>(components.contents);
    const auto actual_axis = *static_cast<const uint32_t*>(axis.contents);

    if (actual_components[0] != static_cast<uint32_t>(expected_components.first) ||
        actual_components[1] != static_cast<uint32_t>(expected_components.second) ||
        actual_axis != static_cast<uint32_t>(expected_axis)) {
        throw std::runtime_error("Metal RoPE geometry semantics drifted");
    }
}

}

int main() {
    try {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("no default Metal device is available");
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:celeg::metal_detail::kInferenceShader];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library) throw std::runtime_error("Metal RoPE geometry shader compilation failed");
        id<MTLFunction> function =
            [library newFunctionWithName:@"celeg_rope_geometry_semantics_probe"];
        if (!function) throw std::runtime_error("missing Metal RoPE geometry probe");
        id<MTLComputePipelineState> state =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!state) throw std::runtime_error("failed Metal RoPE geometry pipeline");
        id<MTLCommandQueue> queue = [device newCommandQueue];

        const std::array<Case, 12> cases{{
            {0, 4, celeg::RopePairingKind::SplitHalf, 2, 1, true},
            {2, 4, celeg::RopePairingKind::SplitHalf, 2, 1, true},
            {3, 4, celeg::RopePairingKind::SplitHalf, 2, 1, true},
            {0, 4, celeg::RopePairingKind::AdjacentPairs, 2, 1, true},
            {2, 4, celeg::RopePairingKind::AdjacentPairs, 2, 1, true},
            {3, 4, celeg::RopePairingKind::AdjacentPairs, 2, 1, true},
            {0, 6, celeg::RopePairingKind::SplitHalf, 2, 3, false},
            {1, 6, celeg::RopePairingKind::SplitHalf, 2, 3, false},
            {2, 6, celeg::RopePairingKind::SplitHalf, 2, 3, false},
            {4, 6, celeg::RopePairingKind::SplitHalf, 2, 3, false},
            {5, 6, celeg::RopePairingKind::SplitHalf, 2, 3, false},
            {5, 6, celeg::RopePairingKind::AdjacentPairs, 2, 3, false},
        }};
        for (const Case& test_case : cases) run_case(device, state, queue, test_case);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
