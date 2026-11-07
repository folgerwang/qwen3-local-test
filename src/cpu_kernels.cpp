#include "cpu_kernels.h"

#include <algorithm>
#include <cstring>

#if defined(__AVX512F__)
#include <immintrin.h>
#define QW_AVX512 1
#endif

namespace ck {

// ---------------------------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------------------------

// e4m3fn byte -> fp16 bits whose value equals fp8_value * 2^-8 (exact incl. subnormals).
inline uint16_t fp8_to_half_bits(uint8_t b) { return (uint16_t)(((b & 0x80) << 8) | ((b & 0x7F) << 7)); }

float half_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
    float f;
    if (exp == 0) {
        f = std::ldexp((float)man, -24);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        u |= sign;
        std::memcpy(&f, &u, 4);
        return f;
    }
    uint32_t u = sign | ((exp + 112) << 23) | (man << 13);
    std::memcpy(&f, &u, 4);
    return f;
}

struct Fp8Lut {
    float v[256];
    Fp8Lut() {
        for (int i = 0; i < 256; i++) v[i] = half_to_f32(fp8_to_half_bits((uint8_t)i)) * 256.0f;
    }
};
const Fp8Lut g_fp8;

float fp8_to_f32(uint8_t b) { return g_fp8.v[b]; }


// ---------------------------------------------------------------------------------------------
// Dot-product kernels. `scale_row` points to cols/128 bf16 block scales for this row.
// ---------------------------------------------------------------------------------------------

#ifdef QW_AVX512
inline __m512 fp8x16_to_f32(const uint8_t* p) {
    __m256i h = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)p));
    __m256i s = _mm256_slli_epi16(_mm256_and_si256(h, _mm256_set1_epi16(0x80)), 8);
    __m256i m = _mm256_slli_epi16(_mm256_and_si256(h, _mm256_set1_epi16(0x7F)), 7);
    return _mm512_cvtph_ps(_mm256_or_si256(s, m));  // value * 2^-8
}
inline __m512 bf16x16_to_f32(const uint16_t* p) {
    __m512i v = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)p));
    return _mm512_castsi512_ps(_mm512_slli_epi32(v, 16));
}
#endif

float dot_q8(const uint8_t* w, const uint16_t* scale_row, const float* x, int cols) {
#ifdef QW_AVX512
    __m512 total = _mm512_setzero_ps();
    for (int j = 0; j < cols; j += 128) {
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        for (int i = 0; i < 128; i += 32) {
            a0 = _mm512_fmadd_ps(fp8x16_to_f32(w + j + i), _mm512_loadu_ps(x + j + i), a0);
            a1 = _mm512_fmadd_ps(fp8x16_to_f32(w + j + i + 16), _mm512_loadu_ps(x + j + i + 16), a1);
        }
        total = _mm512_fmadd_ps(_mm512_add_ps(a0, a1), _mm512_set1_ps(bf16_to_f32(scale_row[j / 128]) * 256.0f), total);
    }
    return _mm512_reduce_add_ps(total);
#else
    float total = 0;
    for (int j = 0; j < cols; j += 128) {
        float acc = 0;
        for (int i = 0; i < 128; i++) acc += g_fp8.v[w[j + i]] * x[j + i];
        total += acc * bf16_to_f32(scale_row[j / 128]);
    }
    return total;
#endif
}

float dot_bf16(const uint16_t* w, const float* x, int cols) {
#ifdef QW_AVX512
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 32 <= cols; i += 32) {
        a0 = _mm512_fmadd_ps(bf16x16_to_f32(w + i), _mm512_loadu_ps(x + i), a0);
        a1 = _mm512_fmadd_ps(bf16x16_to_f32(w + i + 16), _mm512_loadu_ps(x + i + 16), a1);
    }
    float s = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    for (; i < cols; i++) s += bf16_to_f32(w[i]) * x[i];
    return s;
#else
    float s = 0;
    for (int i = 0; i < cols; i++) s += bf16_to_f32(w[i]) * x[i];
    return s;
#endif
}

float dot_f32(const float* a, const float* b, int n) {
#ifdef QW_AVX512
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), a1);
    }
    float s = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
#else
    float s = 0;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
#endif
}

void dequant_q8_row(const uint8_t* w, const uint16_t* scale_row, int cols, float* out) {
    for (int j = 0; j < cols; j += 128) {
#ifdef QW_AVX512
        const __m512 s = _mm512_set1_ps(bf16_to_f32(scale_row[j / 128]) * 256.0f);
        for (int i = 0; i < 128; i += 16) _mm512_storeu_ps(out + j + i, _mm512_mul_ps(fp8x16_to_f32(w + j + i), s));
#else
        const float s = bf16_to_f32(scale_row[j / 128]);
        for (int i = 0; i < 128; i++) out[j + i] = g_fp8.v[w[j + i]] * s;
#endif
    }
}

void dequant_bf16_row(const uint16_t* w, int cols, float* out) {
    int i = 0;
#ifdef QW_AVX512
    for (; i + 16 <= cols; i += 16) _mm512_storeu_ps(out + i, bf16x16_to_f32(w + i));
#endif
    for (; i < cols; i++) out[i] = bf16_to_f32(w[i]);
}

// ---------------------------------------------------------------------------------------------
// Matrix job: Y[n, rows] = X[n, cols] * W^T. Many jobs are fused into one parallel dispatch.
// ---------------------------------------------------------------------------------------------

constexpr int RB = 4;  // rows per block
constexpr int TB = 4;  // tokens per register tile

// Y[t, r0 + i] for i < nr, all tokens, from nr dequantized rows w[i * cols].
void gemm_block(const float* w, int nr, const float* x, int n, int cols, float* y, int rows, int r0) {
    int t0 = 0;
#ifdef QW_AVX512
    if (nr == RB && cols % 16 == 0) {
        for (; t0 + TB <= n; t0 += TB) {
            const float* xp = x + (size_t)t0 * cols;
            __m512 a00 = _mm512_setzero_ps(), a01 = _mm512_setzero_ps(), a02 = _mm512_setzero_ps(), a03 = _mm512_setzero_ps(), a10 = _mm512_setzero_ps(), a11 = _mm512_setzero_ps(), a12 = _mm512_setzero_ps(), a13 = _mm512_setzero_ps(), a20 = _mm512_setzero_ps(), a21 = _mm512_setzero_ps(), a22 = _mm512_setzero_ps(), a23 = _mm512_setzero_ps(), a30 = _mm512_setzero_ps(), a31 = _mm512_setzero_ps(), a32 = _mm512_setzero_ps(), a33 = _mm512_setzero_ps();
            for (int c = 0; c < cols; c += 16) {  // explicitly unrolled 4x4 tile (MSVC won't unroll arrays)
                const __m512 v0 = _mm512_loadu_ps(xp + 0 * (size_t)cols + c), v1 = _mm512_loadu_ps(xp + 1 * (size_t)cols + c), v2 = _mm512_loadu_ps(xp + 2 * (size_t)cols + c), v3 = _mm512_loadu_ps(xp + 3 * (size_t)cols + c);
                { const __m512 wv = _mm512_loadu_ps(w + 0 * (size_t)cols + c);
                  a00 = _mm512_fmadd_ps(wv, v0, a00); a01 = _mm512_fmadd_ps(wv, v1, a01); a02 = _mm512_fmadd_ps(wv, v2, a02); a03 = _mm512_fmadd_ps(wv, v3, a03); }
                { const __m512 wv = _mm512_loadu_ps(w + 1 * (size_t)cols + c);
                  a10 = _mm512_fmadd_ps(wv, v0, a10); a11 = _mm512_fmadd_ps(wv, v1, a11); a12 = _mm512_fmadd_ps(wv, v2, a12); a13 = _mm512_fmadd_ps(wv, v3, a13); }
                { const __m512 wv = _mm512_loadu_ps(w + 2 * (size_t)cols + c);
                  a20 = _mm512_fmadd_ps(wv, v0, a20); a21 = _mm512_fmadd_ps(wv, v1, a21); a22 = _mm512_fmadd_ps(wv, v2, a22); a23 = _mm512_fmadd_ps(wv, v3, a23); }
                { const __m512 wv = _mm512_loadu_ps(w + 3 * (size_t)cols + c);
                  a30 = _mm512_fmadd_ps(wv, v0, a30); a31 = _mm512_fmadd_ps(wv, v1, a31); a32 = _mm512_fmadd_ps(wv, v2, a32); a33 = _mm512_fmadd_ps(wv, v3, a33); }
            }
            float* yp = y + (size_t)t0 * rows + r0;
            yp[0 * (size_t)rows + 0] = _mm512_reduce_add_ps(a00); yp[1 * (size_t)rows + 0] = _mm512_reduce_add_ps(a01); yp[2 * (size_t)rows + 0] = _mm512_reduce_add_ps(a02); yp[3 * (size_t)rows + 0] = _mm512_reduce_add_ps(a03);
            yp[0 * (size_t)rows + 1] = _mm512_reduce_add_ps(a10); yp[1 * (size_t)rows + 1] = _mm512_reduce_add_ps(a11); yp[2 * (size_t)rows + 1] = _mm512_reduce_add_ps(a12); yp[3 * (size_t)rows + 1] = _mm512_reduce_add_ps(a13);
            yp[0 * (size_t)rows + 2] = _mm512_reduce_add_ps(a20); yp[1 * (size_t)rows + 2] = _mm512_reduce_add_ps(a21); yp[2 * (size_t)rows + 2] = _mm512_reduce_add_ps(a22); yp[3 * (size_t)rows + 2] = _mm512_reduce_add_ps(a23);
            yp[0 * (size_t)rows + 3] = _mm512_reduce_add_ps(a30); yp[1 * (size_t)rows + 3] = _mm512_reduce_add_ps(a31); yp[2 * (size_t)rows + 3] = _mm512_reduce_add_ps(a32); yp[3 * (size_t)rows + 3] = _mm512_reduce_add_ps(a33);
        }
    }
#endif
#ifdef QW_AVX512
    if (nr == RB && cols % 16 == 0) {
        for (; t0 < n; t0++) {  // leftover tokens: 4 rows x 1 token
            const float* xp = x + (size_t)t0 * cols;
            __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
            for (int c = 0; c < cols; c += 16) {
                const __m512 v = _mm512_loadu_ps(xp + c);
                a0 = _mm512_fmadd_ps(_mm512_loadu_ps(w + c), v, a0);
                a1 = _mm512_fmadd_ps(_mm512_loadu_ps(w + (size_t)cols + c), v, a1);
                a2 = _mm512_fmadd_ps(_mm512_loadu_ps(w + 2 * (size_t)cols + c), v, a2);
                a3 = _mm512_fmadd_ps(_mm512_loadu_ps(w + 3 * (size_t)cols + c), v, a3);
            }
            float* yp = y + (size_t)t0 * rows + r0;
            yp[0] = _mm512_reduce_add_ps(a0); yp[1] = _mm512_reduce_add_ps(a1);
            yp[2] = _mm512_reduce_add_ps(a2); yp[3] = _mm512_reduce_add_ps(a3);
        }
    }
#endif
    for (; t0 < n; t0++)
        for (int i = 0; i < nr; i++)
            y[(size_t)t0 * rows + r0 + i] = dot_f32(w + (size_t)i * cols, x + (size_t)t0 * cols, cols);
}

void run_block(const MatJob& j, int r0, std::vector<float>& buf) {
    const int cols = j.cols(), rows = j.rows(), nr = std::min(RB, rows - r0);
    if (j.n <= 3) {  // few tokens (decode, sparse experts): fused decode + dot, row stays in L1
        for (int r = r0; r < r0 + nr; r++)
            for (int t = 0; t < j.n; t++) {
                const float* x = j.x + (size_t)t * cols;
                j.y[(size_t)t * rows + r] =
                    j.q ? dot_q8(j.q->w + (size_t)r * cols, j.q->scale + (size_t)(r / 128) * (cols / 128), x, cols)
                        : dot_bf16(j.b->w + (size_t)r * cols, x, cols);
            }
        return;
    }
    buf.resize((size_t)RB * cols);
    for (int i = 0; i < nr; i++) {
        const int r = r0 + i;
        if (j.q) dequant_q8_row(j.q->w + (size_t)r * cols, j.q->scale + (size_t)(r / 128) * (cols / 128), cols, &buf[(size_t)i * cols]);
        else dequant_bf16_row(j.b->w + (size_t)r * cols, cols, &buf[(size_t)i * cols]);
    }
    gemm_block(buf.data(), nr, j.x, j.n, cols, j.y, rows, r0);
}

void run_jobs(ThreadPool& pool, const std::vector<MatJob>& jobs) {
    std::vector<int64_t> start(jobs.size() + 1, 0);  // in row blocks
    for (size_t i = 0; i < jobs.size(); i++) start[i + 1] = start[i] + (jobs[i].rows() + RB - 1) / RB;
    pool.parallel_for(start.back(), [&](int64_t b, int64_t e) {
        thread_local std::vector<float> buf;
        size_t j = std::upper_bound(start.begin(), start.end(), b) - start.begin() - 1;
        for (int64_t blk = b; blk < e;) {
            while (blk >= start[j + 1]) j++;
            const MatJob& job = jobs[j];
            const int64_t end = std::min(e, start[j + 1]);
            if (job.n <= 64) {
                for (; blk < end; blk++) run_block(job, (int)(blk - start[j]) * RB, buf);
                continue;
            }
            // Many tokens: sweep a 64-token chunk (kept in L2) across all of this thread's row blocks.
            const int cols = job.cols(), rows = job.rows();
            for (int t0 = 0; t0 < job.n; t0 += 64) {
                MatJob sub = job;
                sub.n = std::min(64, job.n - t0);
                sub.x = job.x + (size_t)t0 * cols;
                sub.y = job.y + (size_t)t0 * rows;
                for (int64_t k = blk; k < end; k++) run_block(sub, (int)(k - start[j]) * RB, buf);
            }
            blk = end;
        }
    }, 4);
}

// One gated-delta-rule step for a head with state S[KD][DV]:
//   S *= decay; kv = S^T k; d = (v - kv) * beta; S += k d^T; o = S^T q
void delta_step(float* S, const float* k, const float* q, const float* v, float beta, float decay, float* o, int KD,
                int DV) {
#ifdef QW_AVX512
    if (DV == 128) {
        __m512 acc[8], dv[8];
        const __m512 dec = _mm512_set1_ps(decay);
        for (int j = 0; j < 8; j++) acc[j] = _mm512_setzero_ps();
        for (int i = 0; i < KD; i++) {
            float* Si = S + (size_t)i * 128;
            const __m512 ki = _mm512_set1_ps(k[i]);
            for (int j = 0; j < 8; j++) {
                __m512 sv = _mm512_mul_ps(_mm512_loadu_ps(Si + 16 * j), dec);
                _mm512_storeu_ps(Si + 16 * j, sv);
                acc[j] = _mm512_fmadd_ps(sv, ki, acc[j]);
            }
        }
        const __m512 bt = _mm512_set1_ps(beta);
        for (int j = 0; j < 8; j++) {
            dv[j] = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(v + 16 * j), acc[j]), bt);
            acc[j] = _mm512_setzero_ps();
        }
        for (int i = 0; i < KD; i++) {
            float* Si = S + (size_t)i * 128;
            const __m512 ki = _mm512_set1_ps(k[i]), qi = _mm512_set1_ps(q[i]);
            for (int j = 0; j < 8; j++) {
                __m512 sv = _mm512_fmadd_ps(ki, dv[j], _mm512_loadu_ps(Si + 16 * j));
                _mm512_storeu_ps(Si + 16 * j, sv);
                acc[j] = _mm512_fmadd_ps(sv, qi, acc[j]);
            }
        }
        for (int j = 0; j < 8; j++) _mm512_storeu_ps(o + 16 * j, acc[j]);
        return;
    }
#endif
    std::vector<float> kv(DV, 0.f), d(DV);
    for (int i = 0; i < KD; i++) {
        float* Si = S + (size_t)i * DV;
        for (int j = 0; j < DV; j++) { Si[j] *= decay; kv[j] += Si[j] * k[i]; }
    }
    for (int j = 0; j < DV; j++) d[j] = (v[j] - kv[j]) * beta;
    std::fill(o, o + DV, 0.f);
    for (int i = 0; i < KD; i++) {
        float* Si = S + (size_t)i * DV;
        for (int j = 0; j < DV; j++) { Si[j] += k[i] * d[j]; o[j] += Si[j] * q[i]; }
    }
}

// out[d] += p * v[d]
void axpy(float* out, float p, const float* v, int n) {
    int i = 0;
#ifdef QW_AVX512
    const __m512 pv = _mm512_set1_ps(p);
    for (; i + 16 <= n; i += 16) _mm512_storeu_ps(out + i, _mm512_fmadd_ps(pv, _mm512_loadu_ps(v + i), _mm512_loadu_ps(out + i)));
#endif
    for (; i < n; i++) out[i] += p * v[i];
}

void rmsnorm(const float* x, const BMat& w, float* out, int n, float eps, bool one_plus, int woff) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float r = 1.0f / std::sqrt((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) {
        float wi = w.at(woff + i);
        out[i] = x[i] * r * (one_plus ? 1.0f + wi : wi);
    }
}

void ple_embed(ThreadPool& pool, const Config& c, const Ple& P, const std::vector<int>& hist, const int* tokens, int n,
               float* emb) {
    const int NG = c.ngram, C = NG - 1, HPN = c.ngram_heads / C, D = c.ple_dim, eos = c.ple_eos;
    std::vector<int> h(hist);  // previous C tokens, then this chunk
    h.insert(h.end(), tokens, tokens + n);
    pool.parallel_for(n, [&](int64_t b, int64_t e) {
        for (int64_t t = b; t < e; t++) {
            const int i = C + (int)t;
            // shifted[s] = token i-s if no eos lies in [i-s, i-1] (same segment), else eos
            int64_t sh[4];
            sh[0] = h[i];
            bool ok = true;
            for (int s = 1; s < NG; s++) {
                ok = ok && h[i - s] != eos;
                sh[s] = ok ? h[i - s] : eos;
            }
            for (int g = 2; g <= NG; g++) {
                uint64_t mixed = (uint64_t)(sh[0] * P.mult[0]);
                for (int p = 1; p < g; p++) mixed ^= (uint64_t)(sh[p] * P.mult[p]);
                for (int k = 0; k < HPN; k++) {
                    const int hd = (g - 2) * HPN + k;
                    const int64_t row = (int64_t)(mixed % (uint64_t)P.sizes[hd]) + P.offsets[hd];
                    const uint8_t* src = P.shards[row / P.shard_rows] + (size_t)(row % P.shard_rows) * P.dim;
                    float* dst = &emb[(size_t)t * D + hd * P.dim];
                    for (int j = 0; j < P.dim; j++) dst[j] = fp8_to_f32(src[j]) * P.scale;
                }
            }
        }
    });
}

}  // namespace ck
