#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr uint32_t QH = 16, KH = 8, HD = 64, PAGE = 16;
constexpr uint32_t ROWS[] = {128, 256, 512};
constexpr float SCALE = 1.0f / 8.0f;
constexpr NSUInteger BASE_SG = 8, BASE_THREADS = 256;
constexpr NSUInteger BASE_SHARED = (2 * BASE_SG + BASE_SG * HD) * sizeof(float);
constexpr NSUInteger TILED_THREADS = 128;
constexpr NSUInteger TILED_SHARED = 2400 * sizeof(float);

struct Stats { double mean = 0.0, rmse = 0.0; float maximum = 0.0f; };

std::string text(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error(std::string("cannot read ") + path);
    std::ostringstream s; s << f.rdbuf(); return s.str();
}
std::string ns(NSString* s) { return s ? std::string(s.UTF8String) : std::string{}; }

id<MTLBuffer> buffer(id<MTLDevice> d, size_t n, uint32_t seed, float lo, float hi) {
    id<MTLBuffer> b = [d newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
    if (!b) throw std::runtime_error("buffer allocation failed");
    auto* p = static_cast<float*>(b.contents);
    std::mt19937 g(seed); std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < n; ++i) p[i] = dist(g);
    return b;
}
id<MTLBuffer> zeros(id<MTLDevice> d, size_t n) {
    id<MTLBuffer> b = [d newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
    if (!b) throw std::runtime_error("buffer allocation failed");
    std::memset(b.contents, 0, n * sizeof(float)); return b;
}
id<MTLComputePipelineState> pipeline(id<MTLDevice> d, id<MTLLibrary> l, const char* name) {
    id<MTLFunction> f = [l newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!f) throw std::runtime_error(std::string("missing kernel: ") + name);
    NSError* e = nil; id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:f error:&e];
    if (!p) throw std::runtime_error(std::string("pipeline failed: ") + name + ": " + (e ? ns(e.localizedDescription) : "unknown"));
    return p;
}
void base(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p,
          id<MTLBuffer> q, id<MTLBuffer> k, id<MTLBuffer> v, id<MTLBuffer> o, uint32_t rows) {
    const uint32_t pos = 0;
    [e setComputePipelineState:p];
    [e setBuffer:q offset:0 atIndex:0]; [e setBuffer:k offset:0 atIndex:1];
    [e setBuffer:v offset:0 atIndex:2]; [e setBuffer:o offset:0 atIndex:3];
    [e setBytes:&rows length:4 atIndex:4]; [e setBytes:&pos length:4 atIndex:5];
    [e setBytes:&QH length:4 atIndex:6]; [e setBytes:&KH length:4 atIndex:7];
    [e setBytes:&HD length:4 atIndex:8]; [e setBytes:&SCALE length:4 atIndex:9];
    [e setBytes:&PAGE length:4 atIndex:10]; [e setThreadgroupMemoryLength:BASE_SHARED atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(QH, rows, 1) threadsPerThreadgroup:MTLSizeMake(BASE_THREADS,1,1)];
}
void tiled(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p,
           id<MTLBuffer> q, id<MTLBuffer> k, id<MTLBuffer> v, id<MTLBuffer> o, uint32_t rows) {
    [e setComputePipelineState:p];
    [e setBuffer:q offset:0 atIndex:0]; [e setBuffer:k offset:0 atIndex:1];
    [e setBuffer:v offset:0 atIndex:2]; [e setBuffer:o offset:0 atIndex:3];
    [e setBytes:&rows length:4 atIndex:4]; [e setBytes:&QH length:4 atIndex:5];
    [e setBytes:&KH length:4 atIndex:6]; [e setBytes:&HD length:4 atIndex:7];
    [e setBytes:&SCALE length:4 atIndex:8]; [e setThreadgroupMemoryLength:TILED_SHARED atIndex:0];
    [e dispatchThreadgroups:MTLSizeMake(QH, (rows + 31) / 32, 1) threadsPerThreadgroup:MTLSizeMake(TILED_THREADS,1,1)];
}
void wait(id<MTLCommandBuffer> c) {
    [c commit]; [c waitUntilCompleted];
    if (c.status != MTLCommandBufferStatusCompleted) throw std::runtime_error(c.error ? ns(c.error.localizedDescription) : "Metal failure");
}
Stats compare(id<MTLBuffer> a, id<MTLBuffer> b, size_t n) {
    auto* x = static_cast<const float*>(a.contents); auto* y = static_cast<const float*>(b.contents);
    Stats s; double sa = 0.0, ss = 0.0;
    for (size_t i = 0; i < n; ++i) { float d = std::abs(x[i]-y[i]); s.maximum = std::max(s.maximum,d); sa += d; ss += double(d)*d; }
    s.mean = sa/n; s.rmse = std::sqrt(ss/n); return s;
}
template<class F> double time(id<MTLCommandQueue> q, F f, int iters) {
    std::vector<double> samples;
    for (int r = 0; r < 6; ++r) { @autoreleasepool {
        id<MTLCommandBuffer> c = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [c computeCommandEncoder];
        for (int i=0;i<iters;++i) f(e); [e endEncoding]; wait(c);
        if (r) samples.push_back((c.GPUEndTime-c.GPUStartTime)*1000.0/iters);
    }}
    std::sort(samples.begin(), samples.end()); return samples[samples.size()/2];
}
}

int main() {
    try {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice(); if (!d) throw std::runtime_error("no Metal device");
        id<MTLCommandQueue> q = [d newCommandQueue];
        std::string src = text("src/backend/metal/kernels/inference/common.metal") + "\n" +
                          text("src/backend/metal/kernels/inference/attention_one_exp.metal") + "\n" +
                          text("apps/benchmark/metal/attention_tiled_simdgroup.metal");
        NSError* err=nil; id<MTLLibrary> lib=[d newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:nil error:&err];
        if (!lib) throw std::runtime_error("shader failed: " + (err ? ns(err.localizedDescription) : std::string("unknown")));
        auto bp=pipeline(d,lib,"celeg_attention_batch"); auto tp=pipeline(d,lib,"celeg_attention_tiled_simdgroup");
        std::cout << "Metal LFM2.5 tiled simdgroup attention A/B on " << ns(d.name) << "\n";
        std::cout << "heads=16 kv_heads=8 head_dim=64 candidate=float8x8 tiled-online fast-mode shared=9.375KiB\n\n";
        std::cout << std::left << std::setw(8) << "rows" << std::right << std::setw(14) << "one-exp ms" << std::setw(14) << "tiled ms" << std::setw(11) << "speedup" << std::setw(14) << "mean abs" << std::setw(14) << "max abs" << std::setw(14) << "rmse" << "\n";
        for (uint32_t rows: ROWS) {
            size_t qn=size_t(rows)*QH*HD, kvn=size_t(rows)*KH*HD;
            auto qb=buffer(d,qn,0x5100+rows,-0.4f,0.4f), kb=buffer(d,kvn,0x5200+rows,-0.4f,0.4f), vb=buffer(d,kvn,0x5300+rows,-1.0f,1.0f);
            auto bo=zeros(d,qn), to=zeros(d,qn);
            id<MTLCommandBuffer> c=[q commandBuffer]; id<MTLComputeCommandEncoder> e=[c computeCommandEncoder];
            base(e,bp,qb,kb,vb,bo,rows); tiled(e,tp,qb,kb,vb,to,rows); [e endEncoding]; wait(c);
            Stats s=compare(bo,to,qn); int it=rows>=512?2:3;
            double bm=time(q,^(id<MTLComputeCommandEncoder> x){base(x,bp,qb,kb,vb,bo,rows);},it);
            double tm=time(q,^(id<MTLComputeCommandEncoder> x){tiled(x,tp,qb,kb,vb,to,rows);},it);
            std::cout<<std::left<<std::setw(8)<<rows<<std::right<<std::fixed<<std::setprecision(3)<<std::setw(14)<<bm<<std::setw(14)<<tm<<std::setw(10)<<(bm/tm)<<"x"<<std::scientific<<std::setprecision(3)<<std::setw(14)<<s.mean<<std::setw(14)<<s.maximum<<std::setw(14)<<s.rmse<<"\n";
        }
        return 0;
    } catch(const std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 1; }
}
