#include "quant.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "cpu_kernels.h"

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
#include <immintrin.h>
#define QW_AVX512 1
#endif

namespace gq {

namespace {

inline float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
#ifdef QW_AVX512
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
#else
    return ck::half_to_f32(h);
#endif
}

const int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// 6-bit (scale, min) pair `j` of a K-quant super-block.
inline void scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

#ifdef QW_AVX512
// 16 unsigned bytes -> 16 floats * a + b
inline void put16(float* out, __m128i v, __m512 a, __m512 b) {
    _mm512_storeu_ps(out, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v)), a, b));
}
// 32 unsigned bytes -> 32 floats * a + b
inline void put32(float* out, __m256i v, __m512 a, __m512 b) {
    put16(out, _mm256_castsi256_si128(v), a, b);
    put16(out + 16, _mm256_extracti128_si256(v, 1), a, b);
}
#endif

void dq_q8_0(const uint8_t* p, int cols, float* out) {
    for (int j = 0; j < cols; j += 32, p += 34) {
        const float d = h2f(p);
        const int8_t* q = (const int8_t*)p + 2;
#ifdef QW_AVX512
        const __m512 dv = _mm512_set1_ps(d);
        _mm512_storeu_ps(out + j, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)q))), dv));
        _mm512_storeu_ps(out + j + 16, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(q + 16)))), dv));
#else
        for (int i = 0; i < 32; i++) out[j + i] = d * q[i];
#endif
    }
}

void dq_q5_1(const uint8_t* p, int cols, float* out) {
    for (int j = 0; j < cols; j += 32, p += 24) {
        const float d = h2f(p), m = h2f(p + 2);
        uint32_t qh;
        std::memcpy(&qh, p + 4, 4);
        const uint8_t* qs = p + 8;
#ifdef QW_AVX512
        const __m128i q = _mm_loadu_si128((const __m128i*)qs), m4 = _mm_set1_epi8(0x0F);
        const __m128i lo = _mm_or_si128(_mm_and_si128(q, m4), _mm_maskz_set1_epi8((__mmask16)(qh & 0xFFFF), 16));
        const __m128i hi = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q, 4), m4), _mm_maskz_set1_epi8((__mmask16)(qh >> 16), 16));
        const __m512 dv = _mm512_set1_ps(d), mv = _mm512_set1_ps(m);
        put16(out + j, lo, dv, mv);
        put16(out + j + 16, hi, dv, mv);
#else
        for (int i = 0; i < 16; i++) {
            const int x0 = (qs[i] & 0x0F) | (((qh >> i) & 1) << 4);
            const int x1 = (qs[i] >> 4) | (((qh >> (i + 16)) & 1) << 4);
            out[j + i] = x0 * d + m;
            out[j + i + 16] = x1 * d + m;
        }
#endif
    }
}

void dq_iq4_nl(const uint8_t* p, int cols, float* out) {
    for (int j = 0; j < cols; j += 32, p += 18) {
        const float d = h2f(p);
        const uint8_t* qs = p + 2;
        for (int i = 0; i < 16; i++) {
            out[j + i] = d * kIq4nl[qs[i] & 0xF];
            out[j + i + 16] = d * kIq4nl[qs[i] >> 4];
        }
    }
}

// Q4_K (hi == nullptr inside) and Q5_K share the layout apart from the 32 high-bit bytes of Q5_K.
void dq_q45_k(const uint8_t* p, int cols, float* out, bool q5) {
    const int bytes = q5 ? 176 : 144;
    for (int j = 0; j < cols; j += 256, p += bytes) {
        const float d = h2f(p), dmin = h2f(p + 2);
        const uint8_t* sc = p + 4;
        const uint8_t* qh = p + 16;
        const uint8_t* ql = p + (q5 ? 48 : 16);
        float* o = out + j;
        for (int g = 0; g < 4; g++, ql += 32, o += 64) {
            uint8_t s1, m1, s2, m2;
            scale_min_k4(2 * g, sc, s1, m1);
            scale_min_k4(2 * g + 1, sc, s2, m2);
            const float d1 = d * s1, o1 = -dmin * m1, d2 = d * s2, o2 = -dmin * m2;
#ifdef QW_AVX512
            const __m256i q = _mm256_loadu_si256((const __m256i*)ql), m4 = _mm256_set1_epi8(0x0F);
            __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            if (q5) {
                const __m256i h = _mm256_loadu_si256((const __m256i*)qh);
                lo = _mm256_or_si256(lo, _mm256_maskz_set1_epi8(_mm256_test_epi8_mask(h, _mm256_set1_epi8((char)(1 << (2 * g)))), 16));
                hi = _mm256_or_si256(hi, _mm256_maskz_set1_epi8(_mm256_test_epi8_mask(h, _mm256_set1_epi8((char)(2 << (2 * g)))), 16));
            }
            put32(o, lo, _mm512_set1_ps(d1), _mm512_set1_ps(o1));
            put32(o + 32, hi, _mm512_set1_ps(d2), _mm512_set1_ps(o2));
#else
            for (int l = 0; l < 32; l++) {
                int a = ql[l] & 0xF, b = ql[l] >> 4;
                if (q5) {
                    a += (qh[l] & (1 << (2 * g))) ? 16 : 0;
                    b += (qh[l] & (2 << (2 * g))) ? 16 : 0;
                }
                o[l] = d1 * a + o1;
                o[l + 32] = d2 * b + o2;
            }
#endif
        }
    }
}

void dq_q6_k(const uint8_t* p, int cols, float* out) {
    for (int j = 0; j < cols; j += 256, p += 210) {
        const uint8_t* ql = p;
        const uint8_t* qh = p + 128;
        const int8_t* sc = (const int8_t*)p + 192;
        const float d = h2f(p + 208);
        float* y = out + j;
        for (int n = 0; n < 256; n += 128, y += 128, ql += 64, qh += 32, sc += 8) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l] = d * sc[is] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
        }
    }
}

float dot_q8_0(const uint8_t* p, const float* x, int cols) {
#ifdef QW_AVX512
    __m512 total = _mm512_setzero_ps();
    for (int j = 0; j < cols; j += 32, p += 34) {
        const __m128i* q = (const __m128i*)(p + 2);
        __m512 a = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(q))), _mm512_loadu_ps(x + j));
        a = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(q + 1))), _mm512_loadu_ps(x + j + 16), a);
        total = _mm512_fmadd_ps(a, _mm512_set1_ps(h2f(p)), total);
    }
    return _mm512_reduce_add_ps(total);
#else
    float total = 0;
    for (int j = 0; j < cols; j += 32, p += 34) {
        const int8_t* q = (const int8_t*)p + 2;
        float a = 0;
        for (int i = 0; i < 32; i++) a += q[i] * x[j + i];
        total += a * h2f(p);
    }
    return total;
#endif
}

#ifdef QW_AVX512
inline __m512 u8x16_ps(__m128i v) { return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(v)); }

// Fused dots for the 4/5-bit codecs. A block's weights are a * q + o with integer q, so the row dot is
// sum_b a_b * <q_b, x_b> + o_b * sum(x_b); `xs` holds sum(x_b) per 32 inputs (computed once per input vector).
float dot_q45_k(const uint8_t* p, const float* x, const float* xs, int cols, bool q5) {
    const int bytes = q5 ? 176 : 144;
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m512 total = _mm512_setzero_ps();
    float off = 0;
    for (int j = 0; j < cols; j += 256, p += bytes, x += 256, xs += 8) {
        const float d = h2f(p), dmin = h2f(p + 2);
        const uint8_t* sc = p + 4;
        const uint8_t* qh = p + 16;
        const uint8_t* ql = p + (q5 ? 48 : 16);
        float mo = 0;
        for (int g = 0; g < 4; g++, ql += 32) {
            uint8_t s1, m1, s2, m2;
            scale_min_k4(2 * g, sc, s1, m1);
            scale_min_k4(2 * g + 1, sc, s2, m2);
            mo += m1 * xs[2 * g] + m2 * xs[2 * g + 1];
            const __m256i q = _mm256_loadu_si256((const __m256i*)ql);
            __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            if (q5) {
                const __m256i h = _mm256_loadu_si256((const __m256i*)qh);
                lo = _mm256_or_si256(lo, _mm256_maskz_set1_epi8(_mm256_test_epi8_mask(h, _mm256_set1_epi8((char)(1 << (2 * g)))), 16));
                hi = _mm256_or_si256(hi, _mm256_maskz_set1_epi8(_mm256_test_epi8_mask(h, _mm256_set1_epi8((char)(2 << (2 * g)))), 16));
            }
            const float* xg = x + 64 * g;
            __m512 a = _mm512_mul_ps(u8x16_ps(_mm256_castsi256_si128(lo)), _mm512_loadu_ps(xg));
            a = _mm512_fmadd_ps(u8x16_ps(_mm256_extracti128_si256(lo, 1)), _mm512_loadu_ps(xg + 16), a);
            __m512 b = _mm512_mul_ps(u8x16_ps(_mm256_castsi256_si128(hi)), _mm512_loadu_ps(xg + 32));
            b = _mm512_fmadd_ps(u8x16_ps(_mm256_extracti128_si256(hi, 1)), _mm512_loadu_ps(xg + 48), b);
            total = _mm512_fmadd_ps(a, _mm512_set1_ps(d * s1), total);
            total = _mm512_fmadd_ps(b, _mm512_set1_ps(d * s2), total);
        }
        off -= dmin * mo;
    }
    return _mm512_reduce_add_ps(total) + off;
}

float dot_q5_1(const uint8_t* p, const float* x, const float* xs, int cols) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m512 total = _mm512_setzero_ps();
    float off = 0;
    for (int j = 0; j < cols; j += 32, p += 24) {
        uint32_t qh;
        std::memcpy(&qh, p + 4, 4);
        const __m128i q = _mm_loadu_si128((const __m128i*)(p + 8));
        const __m128i lo = _mm_or_si128(_mm_and_si128(q, m4), _mm_maskz_set1_epi8((__mmask16)(qh & 0xFFFF), 16));
        const __m128i hi = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q, 4), m4), _mm_maskz_set1_epi8((__mmask16)(qh >> 16), 16));
        __m512 a = _mm512_mul_ps(u8x16_ps(lo), _mm512_loadu_ps(x + j));
        a = _mm512_fmadd_ps(u8x16_ps(hi), _mm512_loadu_ps(x + j + 16), a);
        total = _mm512_fmadd_ps(a, _mm512_set1_ps(h2f(p)), total);
        off += h2f(p + 2) * xs[j / 32];
    }
    return _mm512_reduce_add_ps(total) + off;
}
#endif

constexpr int RB = 4;  // rows per block

void run_block(const Job& j, int r0, std::vector<float>& buf) {
    const GMat& m = *j.m;
    const int cols = m.cols, rows = m.rows, nr = std::min(RB, rows - r0);
    if (j.n <= 3) {  // few tokens (decode, sparse experts): the decoded row stays in L1
#ifdef QW_AVX512
        if (j.xsum && (m.type == GType::Q4_K || m.type == GType::Q5_K || m.type == GType::Q5_1)) {
            const int nb = cols / 32;
            for (int r = r0; r < r0 + nr; r++) {
                const uint8_t* src = m.row(r);
                for (int t = 0; t < j.n; t++) {
                    const float* x = j.x + (size_t)t * cols;
                    const float* xs = j.xsum + (size_t)t * nb;
                    j.y[(size_t)t * rows + r] = m.type == GType::Q5_1 ? dot_q5_1(src, x, xs, cols)
                                                                       : dot_q45_k(src, x, xs, cols, m.type == GType::Q5_K);
                }
            }
            return;
        }
#endif
        for (int r = r0; r < r0 + nr; r++) {
            const uint8_t* src = m.row(r);
            const float* w = nullptr;
            if (m.type == GType::F32) {
                w = (const float*)src;
            } else if (m.type != GType::Q8_0) {
                buf.resize((size_t)RB * cols);
                dequant_row(m.type, src, cols, buf.data());
                w = buf.data();
            }
            for (int t = 0; t < j.n; t++) {
                const float* x = j.x + (size_t)t * cols;
                j.y[(size_t)t * rows + r] = w ? ck::dot_f32(w, x, cols) : dot_q8_0(src, x, cols);
            }
        }
        return;
    }
    buf.resize((size_t)RB * cols);
    for (int i = 0; i < nr; i++) dequant_row(m.type, m.row(r0 + i), cols, &buf[(size_t)i * cols]);
    ck::gemm_block(buf.data(), nr, j.x, j.n, cols, j.y, rows, r0);
}

}  // namespace

void dequant_row(GType type, const uint8_t* src, int cols, float* out) {
    switch (type) {
        case GType::F32: std::memcpy(out, src, (size_t)cols * 4); break;
        case GType::F16: for (int i = 0; i < cols; i++) out[i] = h2f(src + 2 * i); break;
        case GType::BF16:
            for (int i = 0; i < cols; i++) {
                uint16_t v;
                std::memcpy(&v, src + 2 * i, 2);
                out[i] = bf16_to_f32(v);
            }
            break;
        case GType::Q8_0: dq_q8_0(src, cols, out); break;
        case GType::Q5_1: dq_q5_1(src, cols, out); break;
        case GType::IQ4_NL: dq_iq4_nl(src, cols, out); break;
        case GType::Q4_K: dq_q45_k(src, cols, out, false); break;
        case GType::Q5_K: dq_q45_k(src, cols, out, true); break;
        case GType::Q6_K: dq_q6_k(src, cols, out); break;
        default: throw std::runtime_error("unsupported GGUF tensor type");
    }
}

GMat make_mat(const GTensor& t, const std::string& name, int rows, int cols, int count) {
    const std::vector<int64_t> want = count == 1 ? std::vector<int64_t>{cols, rows} : std::vector<int64_t>{cols, rows, count};
    std::vector<int64_t> dims = t.dims;
    if (count == 1 && rows == 1 && dims.size() == 1) dims.push_back(1);
    if (dims != want) {
        std::string s = "shape mismatch for " + name + ": got [";
        for (auto d : t.dims) s += std::to_string(d) + ",";
        throw std::runtime_error(s + "]");
    }
    const int blk = gtype_block(t.type);
    if (blk == 0) throw std::runtime_error("unsupported tensor type for " + name);
    GMat m;
    m.type = t.type;
    m.data = t.data;
    m.rows = rows;
    m.cols = cols;
    m.row_bytes = (size_t)(cols / blk) * gtype_block_bytes(t.type);
    return m;
}

std::vector<float> to_floats(const GTensor& t, const std::string& name, size_t n) {
    size_t numel = 1;
    for (auto d : t.dims) numel *= (size_t)d;
    if (numel != n) throw std::runtime_error("size mismatch: " + name);
    if (t.type != GType::F32 && t.type != GType::F16 && t.type != GType::BF16)
        throw std::runtime_error("expected a float tensor: " + name);
    std::vector<float> out(n);
    dequant_row(t.type, t.data, (int)n, out.data());
    return out;
}

void run_jobs(ThreadPool& pool, const std::vector<Job>& jobs_in) {
    // per-32 input sums for the fused 4/5-bit dots (few-token jobs only); jobs sharing an input share them
    std::vector<Job> jobs = jobs_in;
    std::vector<std::vector<float>> sums;
    sums.reserve(jobs.size());
    for (size_t i = 0; i < jobs.size(); i++) {
        Job& j = jobs[i];
        const GType t = j.m->type;
        if (j.n > 3 || (t != GType::Q4_K && t != GType::Q5_K && t != GType::Q5_1)) continue;
        if (i > 0 && jobs[i - 1].xsum && jobs[i - 1].x == j.x && jobs[i - 1].n == j.n && jobs[i - 1].m->cols == j.m->cols) {
            j.xsum = jobs[i - 1].xsum;
            continue;
        }
        const int nb = j.m->cols / 32;
        sums.emplace_back((size_t)j.n * nb);
        float* s = sums.back().data();
        for (int k = 0; k < j.n * nb; k++) {
            float a = 0;
            for (int c = 0; c < 32; c++) a += j.x[(size_t)k * 32 + c];
            s[k] = a;
        }
        j.xsum = s;
    }
    std::vector<int64_t> start(jobs.size() + 1, 0);  // in row blocks
    for (size_t i = 0; i < jobs.size(); i++) start[i + 1] = start[i] + (jobs[i].m->rows + RB - 1) / RB;
    pool.parallel_for(start.back(), [&](int64_t b, int64_t e) {
        thread_local std::vector<float> buf;
        size_t j = std::upper_bound(start.begin(), start.end(), b) - start.begin() - 1;
        for (int64_t blk = b; blk < e;) {
            while (blk >= start[j + 1]) j++;
            const Job& job = jobs[j];
            const int64_t end = std::min(e, start[j + 1]);
            if (job.n <= 64) {
                for (; blk < end; blk++) run_block(job, (int)(blk - start[j]) * RB, buf);
                continue;
            }
            // Many tokens: sweep a 64-token chunk (kept in L2) across all of this thread's row blocks.
            const int cols = job.m->cols, rows = job.m->rows;
            for (int t0 = 0; t0 < job.n; t0 += 64) {
                Job sub = job;
                sub.n = std::min(64, job.n - t0);
                sub.x = job.x + (size_t)t0 * cols;
                sub.y = job.y + (size_t)t0 * rows;
                for (int64_t k = blk; k < end; k++) run_block(sub, (int)(k - start[j]) * RB, buf);
            }
            blk = end;
        }
    }, 4);
}

}  // namespace gq
