#include "celeg/backend/cpu/rope.hpp"
#include "metal_inference_source.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kHeadDim = 12, kHeads = 1, kCachePosition = 2, kPageTokens = 16;
constexpr float kTheta = 1000000.0f, kQueryScale = 0.625f;
constexpr float kQueryEpsilon = 1.0e-5f, kKeyEpsilon = 2.0e-5f, kTolerance = 2.0e-4f;
constexpr std::array<uint32_t, 3> kMetalSections{2, 3, 1};
constexpr std::array<int, 3> kCpuSections{2, 3, 1};
constexpr std::array<int32_t, 3> kPosition{3, 9, 17};

std::string ns_string(NSString* value) { return value ? std::string(value.UTF8String) : std::string{}; }
id<MTLBuffer> buffer(id<MTLDevice> d, const std::vector<float>& v) {
    id<MTLBuffer> b = [d newBufferWithBytes:v.data() length:v.size()*sizeof(float) options:MTLResourceStorageModeShared];
    if (!b) throw std::runtime_error("MRoPE fused theta allocation failed"); return b;
}
std::vector<float> values(float p) {
    std::vector<float> v(kHeadDim); for (size_t i=0;i<v.size();++i) v[i]=std::sin(p+i*0.37f)*0.8f+std::cos(p*0.5f+i*0.11f)*0.2f; return v;
}
std::vector<float> weights(float b,float s) {
    std::vector<float> v(kHeadDim); for(uint32_t i=0;i<kHeadDim;++i)v[i]=b+s*i; return v;
}
void check(const float* a,const std::vector<float>& e,const char* label) {
    for(size_t i=0;i<e.size();++i) if(std::abs(a[i]-e[i])>kTolerance)
        throw std::runtime_error(std::string(label)+" differs at "+std::to_string(i));
}
void run(id<MTLDevice> device,id<MTLLibrary> library,id<MTLCommandQueue> queue,bool interleaved) {
    celeg::RopePositionSpec rope; rope.theta=kTheta; rope.rotary_fraction=1.0; rope.pairing=celeg::RopePairingKind::SplitHalf; rope.scaling=celeg::NoRopeScaling{};
    auto q=values(0.4f), k=values(1.1f); const auto v=values(1.9f); const auto qw=weights(0.8f,0.025f), kw=weights(0.95f,-0.018f);
    auto eq=q, ek=k;
    celeg::cpu_qk_norm_rope_mrope(eq.data(),qw.data(),1,kHeadDim,kPosition,kCpuSections,interleaved,rope,kQueryEpsilon);
    celeg::cpu_qk_norm_rope_mrope(ek.data(),kw.data(),1,kHeadDim,kPosition,kCpuSections,interleaved,rope,kKeyEpsilon);
    for(float& x:eq)x*=kQueryScale;
    id<MTLBuffer> qb=buffer(device,q), kb=buffer(device,k), vb=buffer(device,v), qwb=buffer(device,qw), kwb=buffer(device,kw);
    const size_t n=(kCachePosition+1u)*kHeadDim; id<MTLBuffer> kc=buffer(device,std::vector<float>(n,0)), vc=buffer(device,std::vector<float>(n,0));
    NSError* error=nil; id<MTLFunction> fn=[library newFunctionWithName:@"celeg_qk_norm_mrope_store_kv"];
    id<MTLComputePipelineState> state=[device newComputePipelineStateWithFunction:fn error:&error];
    if(!state) throw std::runtime_error("MRoPE fused theta pipeline failed: "+(error?ns_string(error.localizedDescription):"unknown"));
    uint32_t inter=interleaved?1u:0u; id<MTLCommandBuffer> cmd=[queue commandBuffer]; id<MTLComputeCommandEncoder> enc=[cmd computeCommandEncoder]; [enc setComputePipelineState:state];
    [enc setBuffer:qb offset:0 atIndex:0]; [enc setBuffer:qwb offset:0 atIndex:1]; [enc setBuffer:kb offset:0 atIndex:2]; [enc setBuffer:kwb offset:0 atIndex:3]; [enc setBuffer:vb offset:0 atIndex:4]; [enc setBuffer:kc offset:0 atIndex:5]; [enc setBuffer:vc offset:0 atIndex:6];
    [enc setBytes:&kHeads length:sizeof(kHeads) atIndex:7]; [enc setBytes:&kHeads length:sizeof(kHeads) atIndex:8]; [enc setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:9]; [enc setBytes:&kCachePosition length:sizeof(kCachePosition) atIndex:10]; [enc setBytes:kPosition.data() length:sizeof(kPosition) atIndex:11]; [enc setBytes:kMetalSections.data() length:sizeof(kMetalSections) atIndex:12]; [enc setBytes:&kTheta length:sizeof(kTheta) atIndex:13]; [enc setBytes:&kQueryScale length:sizeof(kQueryScale) atIndex:14]; [enc setBytes:&kQueryEpsilon length:sizeof(kQueryEpsilon) atIndex:15]; [enc setBytes:&kKeyEpsilon length:sizeof(kKeyEpsilon) atIndex:16]; [enc setBytes:&kPageTokens length:sizeof(kPageTokens) atIndex:17]; [enc setBytes:&inter length:sizeof(inter) atIndex:18];
    [enc dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    if(cmd.status!=MTLCommandBufferStatusCompleted) throw std::runtime_error("MRoPE fused theta dispatch failed");
    check(static_cast<const float*>(qb.contents),eq,"query"); check(static_cast<const float*>(kb.contents),ek,"key"); check(static_cast<const float*>(kc.contents)+kCachePosition*kHeadDim,ek,"key cache"); check(static_cast<const float*>(vc.contents)+kCachePosition*kHeadDim,v,"value cache");
}
}
int main(){try{id<MTLDevice>d=MTLCreateSystemDefaultDevice();NSError*e=nil;NSString*s=[NSString stringWithUTF8String:celeg::metal_detail::kInferenceShader];id<MTLLibrary>l=[d newLibraryWithSource:s options:nil error:&e];if(!l)throw std::runtime_error("shader compile failed");id<MTLCommandQueue>q=[d newCommandQueue];run(d,l,q,false);run(d,l,q,true);return 0;}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<'\n';return 1;}}
