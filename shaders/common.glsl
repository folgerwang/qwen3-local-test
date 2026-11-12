// Shared definitions. All buffers are accessed through 64-bit device addresses (GL_EXT_buffer_reference).
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require

layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer U4 { uvec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer U32 { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32W { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer U2 { uvec2 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer F32 { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer F4 { vec4 v[]; };

float bf16_lo(uint w) { return uintBitsToFloat(w << 16); }
float bf16_hi(uint w) { return uintBitsToFloat(w & 0xFFFF0000u); }
float bf16_at(uint64_t base, uint i) {
    uint w = U32(base).v[i >> 1];
    return (i & 1u) == 0u ? bf16_lo(w) : bf16_hi(w);
}

// Four e4m3fn bytes -> vec4. Each byte is re-biased into an fp16 bit pattern, giving value * 2^-8
// (the 256 factor is folded into the block scale).
vec4 fp8x4(uint w) {
    uint e = w & 0x00FF00FFu, o = (w >> 8) & 0x00FF00FFu;
    vec2 a = unpackHalf2x16(((e & 0x00800080u) << 8) | ((e & 0x007F007Fu) << 7));
    vec2 b = unpackHalf2x16(((o & 0x00800080u) << 8) | ((o & 0x007F007Fu) << 7));
    return vec4(a.x, b.x, a.y, b.y);
}

vec4 bf16x2x2(uvec2 w) { return vec4(bf16_lo(w.x), bf16_hi(w.x), bf16_lo(w.y), bf16_hi(w.y)); }

float silu(float x) { return x / (1.0 + exp(-x)); }
float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
// log1p(exp(x)); the series keeps full precision where 1 + exp(x) would round the small term away (the
// deltanet decay of long-memory heads lives there, and its error would accumulate over the sequence)
float softplus(float x) {
    if (x > 20.0) return x;
    const float e = exp(x);
    return e < 0.01 ? e * (1.0 - e * (0.5 - e * (1.0 / 3.0))) : log(1.0 + e);
}
