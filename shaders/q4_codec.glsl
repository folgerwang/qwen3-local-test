// GGUF weight codecs for the q4_* matmul shaders (one per pipeline). All rows are a whole number of blocks.
//   C_Q8   planar Q8_0: int8 [rows][cols] at w, one f32 scale per 32 weights at s
//   C_Q4K  native Q4_K super-blocks (144 bytes / 256 weights)
//   C_Q5K  native Q5_K super-blocks (176 bytes / 256 weights)
//   C_Q51  native Q5_1 blocks (24 bytes / 32 weights)
//   C_F32  plain floats
// decode8(w, s, cols, row, u, a0, a1) returns weights [8u, 8u + 8) of `row`.

// four 4-bit fields of w, `sh` bits into each byte
vec4 nib4(uint w, uint sh) { return vec4((uvec4(w) >> (uvec4(0u, 8u, 16u, 24u) + sh)) & 15u); }
// bits [b, b + 4) of w as 0.0 / 16.0 (the fifth bit of a 5-bit weight)
vec4 hi4(uint w, uint b) { return vec4((uvec4(w) >> (uvec4(0u, 1u, 2u, 3u) + b)) & 1u) * 16.0; }
vec4 i8x4(uint w) { return vec4(ivec4(uvec4(w << 24, w << 16, w << 8, w)) >> ivec4(24)); }

// K-quant 6-bit (scale, min) pair j from the 12 packed bytes in words sc[0..2]
uint kbyte(uvec3 sc, uint i) { return (sc[i >> 2] >> ((i & 3u) * 8u)) & 0xFFu; }
uvec2 scale_min(uvec3 sc, uint j) {
    if (j < 4u) return uvec2(kbyte(sc, j) & 63u, kbyte(sc, j + 4u) & 63u);
    const uint b = kbyte(sc, j + 4u);
    return uvec2((b & 15u) | ((kbyte(sc, j - 4u) >> 6) << 4), (b >> 4) | ((kbyte(sc, j) >> 6) << 4));
}

void decode8(uint64_t w, uint64_t s, uint cols, uint row, uint u, out vec4 a0, out vec4 a1) {
#if defined(C_Q8)
    const uvec2 wv = U2(w).v[row * (cols >> 3) + u];
    const float sc = F32(s).v[row * (cols >> 5) + (u >> 2)];
    a0 = i8x4(wv.x) * sc;
    a1 = i8x4(wv.y) * sc;
#elif defined(C_F32)
    F4 W = F4(w);
    const uint b = row * (cols >> 2) + u * 2u;
    a0 = W.v[b];
    a1 = W.v[b + 1u];
#elif defined(C_Q51)
    // block: d, m (f16) | qh (u32) | 16 bytes of nibbles; weights 0..15 are the low nibbles, 16..31 the high ones
    U32 W = U32(w);
    const uint b = (row * (cols >> 5) + (u >> 2)) * 6u, h = (u >> 1) & 1u, part = u & 1u;
    const vec2 dm = unpackHalf2x16(W.v[b]);
    const uint qh = W.v[b + 1u], hb = h * 16u + part * 8u;
    a0 = (nib4(W.v[b + 2u + part * 2u], h * 4u) + hi4(qh, hb)) * dm.x + dm.y;
    a1 = (nib4(W.v[b + 3u + part * 2u], h * 4u) + hi4(qh, hb + 4u)) * dm.x + dm.y;
#else
    // super-block: d, dmin (f16) | 12 bytes of 6-bit scales/mins | [Q5_K: 32 bytes of high bits] | 128 bytes
    // of nibbles. Sub-block j (32 weights) uses the low (j even) or high (j odd) nibbles of bytes [32 (j/2), +32).
#if defined(C_Q5K)
    const uint WORDS = 44u, QS = 12u;
#else
    const uint WORDS = 36u, QS = 4u;
#endif
    U32 W = U32(w);
    const uint b = (row * (cols >> 8) + (u >> 5)) * WORDS, j = (u >> 2) & 7u, half_ = ((u >> 1) & 1u) * 4u + (u & 1u) * 2u;
    const vec2 dd = unpackHalf2x16(W.v[b]);
    const uvec2 sm = scale_min(uvec3(W.v[b + 1u], W.v[b + 2u], W.v[b + 3u]), j);
    const float A = dd.x * float(sm.x), O = -dd.y * float(sm.y);
    const uint q = b + QS + (j >> 1) * 8u + half_, sh = (j & 1u) * 4u;
    a0 = nib4(W.v[q], sh);
    a1 = nib4(W.v[q + 1u], sh);
#if defined(C_Q5K)
    // bit j of high-bit byte l belongs to weight l of sub-block j
    const uvec4 bit = uvec4(0u, 8u, 16u, 24u) + j;
    a0 += vec4((uvec4(W.v[b + 4u + half_]) >> bit) & 1u) * 16.0;
    a1 += vec4((uvec4(W.v[b + 5u + half_]) >> bit) & 1u) * 16.0;
#endif
    a0 = a0 * A + O;
    a1 = a1 * A + O;
#endif
}
