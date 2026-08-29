#include "celeg/backend/metal/device.hpp"

#include "metal_probe_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace celeg {

namespace {

std::string ns_string(NSString* value) {
    return value ? std::string(value.UTF8String) : std::string{};
}

}

struct MetalDevice::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> probe_pipeline = nil;
    MetalCapabilities capabilities;
};

std::string MetalCapabilities::summary() const {
    std::ostringstream output;
    output << device_name
           << " max_buffer=" << max_buffer_length
           << " max_threads=" << max_threads_per_threadgroup
           << " family8=" << (apple_gpu_family_8 ? "yes" : "no")
           << " family9=" << (apple_gpu_family_9 ? "yes" : "no")
           << " family10=" << (apple_gpu_family_10 ? "yes" : "no")
           << " runtime_shader_compilation="
           << (runtime_shader_compilation ? "yes" : "no");
    return output.str();
}

MetalDevice::MetalDevice() : impl_(std::make_unique<Impl>()) {
    (*impl_).device = MTLCreateSystemDefaultDevice();
    if (!(*impl_).device) throw std::runtime_error("no default Metal device is available");
    (*impl_).queue = [(*impl_).device newCommandQueue];
    if (!(*impl_).queue) throw std::runtime_error("Metal command queue creation failed");

    (*impl_).capabilities.device_name = ns_string((*impl_).device.name);
    (*impl_).capabilities.registry_id = std::to_string((*impl_).device.registryID);
    (*impl_).capabilities.max_buffer_length = (*impl_).device.maxBufferLength;
    (*impl_).capabilities.max_threads_per_threadgroup =
        (*impl_).device.maxThreadsPerThreadgroup.width;
    (*impl_).capabilities.apple_gpu_family_8 =
        [(*impl_).device supportsFamily:MTLGPUFamilyApple8];
    (*impl_).capabilities.apple_gpu_family_9 =
        [(*impl_).device supportsFamily:MTLGPUFamilyApple9];
    (*impl_).capabilities.apple_gpu_family_10 =
        [(*impl_).device supportsFamily:static_cast<MTLGPUFamily>(1010)];

    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:metal_detail::kProbeShader];
    id<MTLLibrary> library = [(*impl_).device newLibraryWithSource:source
                                                           options:nil
                                                             error:&error];
    if (!library) {
        const std::string message = error ? ns_string(error.localizedDescription)
                                          : "unknown Metal shader compilation error";
        throw std::runtime_error("Metal probe shader compilation failed: " + message);
    }
    id<MTLFunction> function = [library newFunctionWithName:@"celeg_probe"];
    if (!function) throw std::runtime_error("Metal probe function is missing");
    (*impl_).probe_pipeline = [(*impl_).device newComputePipelineStateWithFunction:function
                                                                            error:&error];
    if (!(*impl_).probe_pipeline) {
        const std::string message = error ? ns_string(error.localizedDescription)
                                          : "unknown Metal pipeline creation error";
        throw std::runtime_error("Metal probe pipeline creation failed: " + message);
    }
    (*impl_).capabilities.runtime_shader_compilation = true;
}

MetalDevice::~MetalDevice() = default;

MetalDevice::MetalDevice(MetalDevice&&) noexcept = default;

MetalDevice& MetalDevice::operator=(MetalDevice&&) noexcept = default;

const MetalCapabilities& MetalDevice::capabilities() const noexcept {
    return (*impl_).capabilities;
}

void MetalDevice::run_probe() const {
    id<MTLBuffer> output = [(*impl_).device newBufferWithLength:sizeof(uint32_t)
                                                       options:MTLResourceStorageModeShared];
    if (!output) throw std::runtime_error("Metal probe buffer allocation failed");
    std::memset(output.contents, 0, sizeof(uint32_t));
    id<MTLCommandBuffer> command_buffer = [(*impl_).queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    [encoder setComputePipelineState:(*impl_).probe_pipeline];
    [encoder setBuffer:output offset:0 atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) {
        const std::string message = command_buffer.error
            ? ns_string(command_buffer.error.localizedDescription)
            : "unknown Metal command-buffer error";
        throw std::runtime_error("Metal probe dispatch failed: " + message);
    }
    if (*static_cast<const uint32_t*>(output.contents) != 0xCE1E9u) {
        throw std::runtime_error("Metal probe returned an unexpected value");
    }
}

bool MetalDevice::available() noexcept {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device != nil;
}

}
