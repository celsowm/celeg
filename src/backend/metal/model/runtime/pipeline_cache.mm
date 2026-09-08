#include "pipeline_cache.hpp"
#include "quant_registry.hpp"

#import <Metal/Metal.h>

#include <stdexcept>
#include <string>

namespace celeg {

id<MTLComputePipelineState> MetalPipelineCache::pipeline(std::string_view name) {
    const std::string key = std::string(name);
    const auto found = pipelines.find(key);
    if (found != pipelines.end()) return found->second;
    NSString* function_name = [NSString stringWithUTF8String:std::string(name).c_str()];
    id<MTLFunction> function = [library newFunctionWithName:function_name];
    if (!function) throw std::runtime_error("Metal inference function is missing: " + std::string(name));
    NSError* error = nil;
    id<MTLComputePipelineState> result = [device newComputePipelineStateWithFunction:function error:&error];
    if (!result) {
        const std::string message = error ? std::string(error.localizedDescription.UTF8String) : "unknown Metal pipeline error";
        throw std::runtime_error("Metal inference pipeline failed: " + message);
    }
    pipelines.emplace(key, result);
    return result;
}

id<MTLComputePipelineState> MetalPipelineCache::tensor_pipeline(std::string_view name) {
    const std::string key = "tensor:" + std::string(name);
    const auto found = pipelines.find(key);
    if (found != pipelines.end()) return found->second;
    id<MTLLibrary> selected_library = quant_tensor_library_for(name, *this);
    if (!selected_library) throw std::runtime_error("Metal tensor library is unavailable");
    NSString* function_name = [NSString stringWithUTF8String:std::string(name).c_str()];
    id<MTLFunction> function = [selected_library newFunctionWithName:function_name];
    if (!function) throw std::runtime_error("Metal tensor function is missing: " + std::string(name));
    NSError* error = nil;
    id<MTLComputePipelineState> result = [device newComputePipelineStateWithFunction:function error:&error];
    if (!result) {
        const std::string message = error ? std::string(error.localizedDescription.UTF8String) : "unknown Metal tensor pipeline error";
        throw std::runtime_error("Metal tensor pipeline failed: " + message);
    }
    pipelines.emplace(key, result);
    return result;
}

}
