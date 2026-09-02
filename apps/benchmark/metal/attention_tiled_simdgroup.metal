constant uint kTQ = 8;
constant uint kTS = 4;
constant uint kHD = 64;
constant uint kScore = kTS * kTQ * 8;
constant uint kOut = kTS * kTQ * kHD;
constant uint kRow = kTS * kTQ;
constant uint kShared = kScore + kOut + 3 * kRow;

kernel void celeg_attention_tiled_simdgroup(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& qh [[buffer(5)]],
    constant uint& kh [[buffer(6)]],
    constant uint& hd [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    threadgroup float* sh [[threadgroup(0)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint2 grid [[threadgroup_position_in_grid]]) {
    const uint head = grid.x;
    const uint qb = grid.y * (kTQ * kTS);
    const uint qr = qb + simd * kTQ;
    if (head >= qh || simd >= kTS || hd != kHD || qr >= rows) return;

    const uint qw = qh * hd;
    const uint kw = kh * hd;
    const uint key_head = head / (qh / kh);
    threadgroup float* score = sh + simd * (kTQ * 8);
    threadgroup float* num = sh + kScore + simd * (kTQ * kHD);
    threadgroup float* mx = sh + kScore + kOut + simd * kTQ;
    threadgroup float* den = sh + kScore + kOut + kRow + simd * kTQ;
    threadgroup float* corr = sh + kScore + kOut + 2 * kRow + simd * kTQ;

    for (uint i = lane; i < kTQ * kHD; i += 32) num[i] = 0.0f;
    if (lane < kTQ) {
        mx[lane] = -INFINITY;
        den[lane] = 0.0f;
        corr[lane] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint lastq = min(rows - 1u, qr + kTQ - 1u);
    for (uint kr = 0; kr <= lastq; kr += 8) {
        simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint d = 0; d < hd; d += 8) {
            simdgroup_float8x8 qm;
            simdgroup_float8x8 km;
            simdgroup_load(qm, q + (size_t)qr * qw + (size_t)head * hd + d,
                           qw, 0, false);
            simdgroup_load(km, k + (size_t)kr * kw + (size_t)key_head * hd + d,
                           kw, 0, true);
            simdgroup_multiply_accumulate(s, qm, km, s);
        }
        simdgroup_store(s, score, 8, 0, false);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < kTQ) {
            const uint gq = qr + lane;
            float bm = -INFINITY;
            for (uint j = 0; j < 8; ++j) {
                const uint gk = kr + j;
                if (gk < rows && gk <= gq) bm = max(bm, score[lane * 8 + j] * scale);
            }
            const float old = mx[lane];
            const float updated = max(old, bm);
            const float c = old == -INFINITY ? 0.0f : exp(old - updated);
            float bs = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                const uint gk = kr + j;
                float p = 0.0f;
                if (gk < rows && gk <= gq) {
                    p = exp(score[lane * 8 + j] * scale - updated);
                    bs += p;
                }
                score[lane * 8 + j] = p;
            }
            corr[lane] = c;
            mx[lane] = updated;
            den[lane] = den[lane] * c + bs;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = lane; i < kTQ * kHD; i += 32) num[i] *= corr[i / hd];
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 pm;
        simdgroup_load(pm, score, 8, 0, false);
        for (uint d = 0; d < hd; d += 8) {
            simdgroup_float8x8 vm;
            simdgroup_float8x8 om;
            simdgroup_load(vm, v + (size_t)kr * kw + (size_t)key_head * hd + d,
                           kw, 0, false);
            simdgroup_load(om, num + d, hd, 0, false);
            simdgroup_multiply_accumulate(om, pm, vm, om);
            simdgroup_store(om, num + d, hd, 0, false);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = lane; i < kTQ * kHD; i += 32) {
        const uint lq = i / hd;
        const uint d = i - lq * hd;
        const uint gq = qr + lq;
        if (gq < rows) {
            out[(size_t)gq * qw + (size_t)head * hd + d] = num[i] / den[lq];
        }
    }
}
