// CPU kernels shared by the CPU backend and the Vulkan backend's CPU expert path: FP8/BF16 dot products,
// fused multi-matrix jobs (Y = X W^T) on a thread pool, and small vector helpers.
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

#include "model.h"
#include "threadpool.h"

namespace ck {

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

// Y[n, rows] = X[n, cols] * W^T for an FP8 (q) or BF16 (b) weight. Many jobs run as one parallel dispatch.
struct MatJob {
    const QMat* q = nullptr;
    const BMat* b = nullptr;
    const float* x = nullptr;
    float* y = nullptr;
    int n = 1;
    int rows() const { return q ? q->rows : b->rows; }
    int cols() const { return q ? q->cols : b->cols; }
};
void run_jobs(ThreadPool& pool, const std::vector<MatJob>& jobs);

float dot_f32(const float* a, const float* b, int n);
void axpy(float* out, float p, const float* v, int n);  // out += p * v
void rmsnorm(const float* x, const BMat& w, float* out, int n, float eps, bool one_plus);
// One gated-delta-rule step for a head with state S[KD][DV].
void delta_step(float* S, const float* k, const float* q, const float* v, float beta, float decay, float* o, int KD, int DV);

}  // namespace ck
