// GGUF (ggml) quantized matrices on the CPU: row dequantization for Q8_0 / Q4_K / Q5_K / Q5_1 / IQ4_NL /
// BF16 / F16 / F32 and fused multi-matrix jobs (Y = X W^T) on the thread pool.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "gguf.h"
#include "threadpool.h"

namespace gq {

// Weight [rows, cols] stored row by row; every row is a whole number of quantization blocks.
struct GMat {
    GType type = GType::Unknown;
    const uint8_t* data = nullptr;
    int rows = 0, cols = 0;
    size_t row_bytes = 0;
    const uint8_t* row(int64_t r) const { return data + (size_t)r * row_bytes; }
    size_t nbytes() const { return (size_t)rows * row_bytes; }
    // Matrix `i` of a stack of equally shaped matrices (the experts of a layer).
    GMat slice(int i) const {
        GMat m = *this;
        m.data = data + (size_t)i * nbytes();
        return m;
    }
};

// Matrix view of a GGUF tensor with dims [cols, rows] or [cols, rows, count] (count matrices, see slice()).
GMat make_mat(const GTensor& t, const std::string& name, int rows, int cols, int count = 1);
// Small float tensor (F32/F16/BF16) as an owned float vector of `n` elements.
std::vector<float> to_floats(const GTensor& t, const std::string& name, size_t n);

void dequant_row(GType type, const uint8_t* src, int cols, float* out);

// Y[n, rows] = X[n, cols] * W^T. Many jobs run as one parallel dispatch.
struct Job {
    const GMat* m = nullptr;
    const float* x = nullptr;
    float* y = nullptr;
    int n = 1;
    const float* xsum = nullptr;  // filled by run_jobs
};
void run_jobs(ThreadPool& pool, const std::vector<Job>& jobs);

}  // namespace gq
