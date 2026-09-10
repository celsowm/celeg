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
constexpr uint32_t kRows=2,kHeads=1,kHeadDim=12;
constexpr float kTheta=1000000.0f,kQueryScale=0.875f,kTolerance=8.0e-5f;
constexpr std::array<uint32_t,3> kMetalSections{2,3,1};
constexpr std::array<int,3> kCpuSections{2,3,1};
constexpr std::array<std::array<int32_t,3>,kRows> kPositions{{{2,5,11},{4,13,19}}};

std::string ns_string(NSString* v){return v?std::string(v.UTF8String):std::string{};}
id<MTLBuffer> fbuf(id<MTLDevice>d,const std::vector<float>&v){id<MTLBuffer>b=[d newBufferWithBytes:v.data() length:v.size()*sizeof(float) options:MTLResourceStorageModeShared];if(!b)throw std::runtime_error("MRoPE batch theta float allocation failed");return b;}
id<MTLBuffer> ibuf(id<MTLDevice>d,const std::vector<int32_t>&v){id<MTLBuffer>b=[d newBufferWithBytes:v.data() length:v.size()*sizeof(int32_t) options:MTLResourceStorageModeShared];if(!b)throw std::runtime_error("MRoPE batch theta position allocation failed");return b;}
std::vector<float> values(float p){std::vector<float>v(kRows*kHeadDim);for(size_t i=0;i<v.size();++i)v[i]=std::sin(p+i*0.29f)*0.75f+std::cos(p*0.4f+i*0.15f)*0.25f;return v;}
void check(const float*a,const std::vector<float>&e,const char*l){for(size_t i=0;i<e.size();++i)if(std::abs(a[i]-e[i])>kTolerance)throw std::runtime_error(std::string(l)+" differs at "+std::to_string(i));}
void run(id<MTLDevice>d,id<MTLLibrary>lib,id<MTLCommandQueue>q,bool interleaved){
 celeg::RopePositionSpec rope;rope.theta=kTheta;rope.rotary_fraction=1.0;rope.pairing=celeg::RopePairingKind::SplitHalf;rope.scaling=celeg::NoRopeScaling{};
 auto query=values(0.2f),key=values(1.0f),eq=query,ek=key;std::vector<int32_t>flat;for(auto&p:kPositions)flat.insert(flat.end(),p.begin(),p.end());
 for(uint32_t r=0;r<kRows;++r){float*qr=eq.data()+r*kHeadDim;float*kr=ek.data()+r*kHeadDim;celeg::cpu_rope_mrope(qr,1,kHeadDim,kPositions[r],kCpuSections,interleaved,rope);celeg::cpu_rope_mrope(kr,1,kHeadDim,kPositions[r],kCpuSections,interleaved,rope);for(uint32_t x=0;x<kHeadDim;++x)qr[x]*=kQueryScale;}
 id<MTLBuffer>qb=fbuf(d,query),kb=fbuf(d,key),pb=ibuf(d,flat);NSError*err=nil;id<MTLFunction>fn=[lib newFunctionWithName:@"celeg_qk_mrope_position_batch"];id<MTLComputePipelineState>state=[d newComputePipelineStateWithFunction:fn error:&err];if(!state)throw std::runtime_error("MRoPE batch theta pipeline failed: "+(err?ns_string(err.localizedDescription):"unknown"));uint32_t inter=interleaved?1u:0u;
 id<MTLCommandBuffer>cmd=[q commandBuffer];id<MTLComputeCommandEncoder>enc=[cmd computeCommandEncoder];[enc setComputePipelineState:state];[enc setBuffer:qb offset:0 atIndex:0];[enc setBuffer:kb offset:0 atIndex:1];[enc setBytes:&kRows length:sizeof(kRows) atIndex:2];[enc setBytes:&kHeads length:sizeof(kHeads) atIndex:3];[enc setBytes:&kHeads length:sizeof(kHeads) atIndex:4];[enc setBytes:&kHeadDim length:sizeof(kHeadDim) atIndex:5];[enc setBuffer:pb offset:0 atIndex:6];[enc setBytes:kMetalSections.data() length:sizeof(kMetalSections) atIndex:7];[enc setBytes:&kTheta length:sizeof(kTheta) atIndex:8];[enc setBytes:&kQueryScale length:sizeof(kQueryScale) atIndex:9];[enc setBytes:&inter length:sizeof(inter) atIndex:10];[enc dispatchThreads:MTLSizeMake(kRows,1,1) threadsPerThreadgroup:MTLSizeMake(kRows,1,1)];[enc endEncoding];[cmd commit];[cmd waitUntilCompleted];if(cmd.status!=MTLCommandBufferStatusCompleted)throw std::runtime_error("MRoPE batch theta dispatch failed");check(static_cast<const float*>(qb.contents),eq,"query");check(static_cast<const float*>(kb.contents),ek,"key");
}
}
int main(){try{id<MTLDevice>d=MTLCreateSystemDefaultDevice();NSError*e=nil;NSString*s=[NSString stringWithUTF8String:celeg::metal_detail::kInferenceShader];id<MTLLibrary>l=[d newLibraryWithSource:s options:nil error:&e];if(!l)throw std::runtime_error("shader compile failed");id<MTLCommandQueue>q=[d newCommandQueue];run(d,l,q,false);run(d,l,q,true);return 0;}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<'\n';return 1;}}
