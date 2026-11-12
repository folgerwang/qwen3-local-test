// QSA indexer (CPU side, shared by the CPU and Vulkan backends).
//
// Every full-attention layer with a compress ratio CR scores blocks of CR consecutive tokens with one pooled
// key: the mean of the tokens' raw indexer keys, normed and rotated to the block's first position. A token
// attends to its `budget / CR` best complete blocks plus the incomplete tail block, so selection only
// matters once more than `budget / CR` blocks are complete; before that attention is dense.
#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "cpu_kernels.h"
#include "qwen4.h"
#include "threadpool.h"

class QsaIndexer {
public:
    QsaIndexer(const Q4Config& c, const Q4Weights& w, int max_ctx, ThreadPool& pool)
        : c_(c), w_(w), max_ctx_(max_ctx), pool_(pool) {
        int n = 0;
        slot_.assign(w.layers.size(), -1);
        for (size_t l = 0; l < w.layers.size(); l++)
            if (w.layers[l].full_attn && w.layers[l].compress > 0) slot_[l] = n++;
        raw_.assign((size_t)n * max_ctx * c.idx_dim, 0.f);
        pooled_.assign((size_t)n * (max_ctx / 2 + 1) * c.idx_dim, 0.f);
        for (int i = 0; i < c.rope_dim / 2; i++)
            inv_freq_.push_back((float)(1.0 / std::pow((double)c.rope_theta, (2.0 * i) / c.rope_dim)));
    }

    bool active(int l) const { return slot_[l] >= 0; }
    int top_blocks(int l) const { return c_.idx_budget / w_.layers[l].compress; }
    // True when some token of [pos, pos + n) sees more complete blocks than it may attend to.
    bool needs_select(int l, int pos, int n) const { return active(l) && (pos + n) / w_.layers[l].compress > top_blocks(l); }

    // Caches the raw keys of tokens [pos, pos + n) from the layer's block input xn[n][hidden] and pools the
    // blocks they complete. Positions below `pos` must have been fed before (they are never rewritten).
    void update(int l, const float* xn, int n, int pos) {
        const Q4Layer& L = w_.layers[l];
        const int ID = c_.idx_dim, CR = L.compress;
        float* raw = &raw_[(size_t)slot_[l] * max_ctx_ * ID];
        float* pk0 = &pooled_[(size_t)slot_[l] * (max_ctx_ / 2 + 1) * ID];
        gq::run_jobs(pool_, {gq::Job{&L.idx_k, xn, raw + (size_t)pos * ID, n}});
        const int b0 = pos / CR, b1 = (pos + n) / CR;  // blocks [b0, b1) end inside this call
        pool_.parallel_for(b1 - b0, [&](int64_t bb, int64_t be) {
            for (int64_t blk = b0 + bb; blk < b0 + be; blk++) {
                float* pk = pk0 + (size_t)blk * ID;
                std::fill(pk, pk + ID, 0.f);
                for (int i = 0; i < CR; i++) ck::axpy(pk, 1.0f / CR, raw + ((size_t)blk * CR + i) * ID, ID);
                norm(pk, L.idx_k_norm.data());
                rope(pk, (int)blk * CR);
            }
        }, 16);
    }

    // After update(): blocks[t] = sorted ids of the complete blocks token pos + t attends to, or empty when
    // it sees all of them (dense). The tail block (keys from (len / CR) * CR on) is always visible.
    void select(int l, const float* xn, int n, int pos, std::vector<std::vector<int>>& blocks) {
        const Q4Layer& L = w_.layers[l];
        const int ID = c_.idx_dim, IH = c_.idx_heads, CR = L.compress, top = top_blocks(l);
        const float* pk0 = &pooled_[(size_t)slot_[l] * (max_ctx_ / 2 + 1) * ID];
        blocks.assign(n, {});
        iq_.resize((size_t)n * IH * ID);
        gq::run_jobs(pool_, {gq::Job{&L.idx_q, xn, iq_.data(), n}});
        pool_.parallel_for(n, [&](int64_t tb, int64_t te) {
            std::vector<float> score;
            std::vector<int> order;
            for (int64_t t = tb; t < te; t++) {
                const int p = pos + (int)t, nblk = (p + 1) / CR;
                if (nblk <= top) continue;
                float* qt = &iq_[(size_t)t * IH * ID];
                for (int h = 0; h < IH; h++) {
                    norm(qt + (size_t)h * ID, L.idx_q_norm.data());
                    rope(qt + (size_t)h * ID, p);
                }
                // score = sum over heads of relu(q_h . pooled); the reference's 1/sqrt(dim) keeps the order
                score.resize(nblk);
                order.resize(nblk);
                for (int b = 0; b < nblk; b++) {
                    float s = 0;
                    for (int h = 0; h < IH; h++) s += std::max(0.f, ck::dot_f32(qt + (size_t)h * ID, pk0 + (size_t)b * ID, ID));
                    score[b] = s;
                }
                std::iota(order.begin(), order.end(), 0);
                std::nth_element(order.begin(), order.begin() + top, order.end(),
                                 [&](int x, int y) { return score[x] > score[y] || (score[x] == score[y] && x < y); });
                std::sort(order.begin(), order.begin() + top);
                blocks[t].assign(order.begin(), order.begin() + top);
            }
        });
    }

private:
    void norm(float* v, const float* w) const {
        const int n = c_.idx_dim;
        const float r = 1.0f / std::sqrt(ck::dot_f32(v, v, n) / n + c_.rms_eps);
        for (int i = 0; i < n; i++) v[i] = v[i] * r * w[i];
    }
    void rope(float* v, int p) const {
        const int half = c_.rope_dim / 2;
        for (int i = 0; i < half; i++) {
            const float ang = (float)p * inv_freq_[i], cs = std::cos(ang), sn = std::sin(ang);
            const float a = v[i], b = v[i + half];
            v[i] = a * cs - b * sn;
            v[i + half] = b * cs + a * sn;
        }
    }

    const Q4Config& c_;
    const Q4Weights& w_;
    int max_ctx_;
    ThreadPool& pool_;
    std::vector<int> slot_;
    std::vector<float> raw_, pooled_, inv_freq_, iq_;
};
