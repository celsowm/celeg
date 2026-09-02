// Benchmark-only production-geometry Q4_K decoder using vector loads/stores.
// The TensorOps geometry, K64 accumulation partition, strict precision,
// threadgroup size and barriers are identical to production. Only the Q4_K
// 32-value decoder is vectorized, making this the lowest-risk performance A/B.

struct CelegTensorQ4KVector {
    void store(threadgroup half* destination, device const uchar* row_data,
               uint column) const {
        const device uchar* block = row_data + static_cast<size_t>(column / 256u) * 144u;
        const uint within = column & 255u;
        const uint sub = within >> 5;
        uchar scale = 0;
        uchar minimum = 0;
        celeg_tensor_scale_min(block + 4, sub, scale, minimum);
        const ushort d_bits = static_cast<ushort>(
            static_cast<uint>(block[0]) | (static_cast<uint>(block[1]) << 8));
        const float d = static_cast<float>(as_type<half>(d_bits));
        const ushort dmin_bits = static_cast<ushort>(
            static_cast<uint>(block[2]) | (static_cast<uint>(block[3]) << 8));
        const float dmin = static_cast<float>(as_type<half>(dmin_bits));
        const float factor = d * static_cast<float>(scale);
        const float bias = dmin * static_cast<float>(minimum);
        const device uchar* qs = block + 16 + (sub >> 1) * 32;
        const bool high = (sub & 1u) != 0u;

        #pragma unroll
        for (uint group = 0; group < 8u; ++group) {
            const uchar4 packed =
                *reinterpret_cast<device const uchar4*>(qs + group * 4u);
            const uint4 raw = uint4(packed);
            const uint4 quantized = high ? (raw >> 4u) : (raw & uint4(0x0fu));
            const float4 decoded = factor * float4(quantized) - float4(bias);
            *reinterpret_cast<threadgroup half4*>(destination + group * 4u) =
                half4(decoded);
        }
    }
};

kernel void celeg_matmul_tensor_q4k_k64_vector(
        device const uchar* weights [[buffer(0)]],
        device float* input [[buffer(1)]],
        device float* output [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        constant uint& output_rows [[buffer(5)]],
        constant uint& output_stride [[buffer(6)]],
        constant uint& row_bytes [[buffer(7)]],
        threadgroup half* weights_tile [[threadgroup(0)]],
        uint thread_index [[thread_index_in_threadgroup]],
        uint2 grid [[threadgroup_position_in_grid]]) {
    celeg_matmul_tensor_quantized(
        weights, input, output, rows, cols, output_rows, output_stride, row_bytes,
        CelegTensorQ4KVector{}, weights_tile, thread_index, grid);
}
