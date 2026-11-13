// A small model, separate from the language model, that is trained while the engine runs to predict how much
// each routed expert will be used in the near future. The VRAM expert cache keeps the experts it scores highest.
//
//   target   log1p of the expert's uses over the next DELAY passes, discounted by GAMMA per pass
//   input    per expert: use counts decayed over 8 / 32 / 128 / 512 passes, the session count, the count from
//            earlier sessions, how recently it was used
//   model    one hidden layer plus a linear path, shared by all experts, and for every expert a linear readout
//            of a sketch of the session's current expert mix (a fixed random projection), which lets it learn
//            "when the session looks like this, this expert is about to be needed"
//   training every pass adds the experts it used (and some random ones) as samples; a sample's label is known
//            DELAY passes later, and it is then used for one Adam / Adagrad step
//
// The weights are saved on exit and loaded at start, so what was learned in one session carries over.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <random>
#include <string>
#include <utility>
#include <vector>

class ExpertPredictor {
public:
    static constexpr int NF = 8, HID = 16, CD = 64, DELAY = 200, NT = 4, MAX_USED = 2048, N_RANDOM = 256;
    static constexpr float GAMMA = 0.99f;
    static constexpr int NP = HID * NF + HID + HID + 1 + NF;  // shared weights: W1, b1, w2, b2, skip

    // prior: use counts from earlier sessions (may be all zero). path: weights file, empty for none.
    void init(size_t n_keys, const std::vector<uint64_t>& prior, const std::string& path) {
        K_ = n_keys;
        path_ = path;
        for (auto& h : heat_) h.assign(K_, 0.f);
        count_.assign(K_, 0.f);
        last_.assign(K_, -1e6f);
        score_.assign(K_, 0.f);
        X_.assign(K_ * NF, 0.f);
        y_.assign(K_, 0.f);
        prior_.assign(K_, 0.f);
        double sum = 0;
        for (size_t k = 0; k < K_ && k < prior.size(); k++) sum += (double)prior[k];
        if (sum > 0)
            for (size_t k = 0; k < K_ && k < prior.size(); k++) prior_[k] = std::log1p((float)(prior[k] * (1e4 / sum)));
        std::mt19937 r(1);
        std::normal_distribution<float> nd(0.f, 1.f);
        P_.resize((size_t)CD * K_);
        const float ps = 1.f / std::sqrt((float)K_);
        for (float& p : P_) p = nd(r) * ps;
        V_.assign(K_ * CD, 0.f);
        b_.assign(K_, 0.f);
        G_.assign(K_, 1e-3f);
        th_.assign(NP, 0.f);
        m_.assign(NP, 0.f);
        v_.assign(NP, 0.f);
        for (int i = 0; i < HID * NF; i++) th_[i] = nd(r) * 0.3f;
        for (int i = 0; i < HID; i++) th_[HID * NF + HID + i] = nd(r) * 0.1f;
        th_[NP - NF + 2] = 1.f;  // before any training: follow the count decayed over 128 passes
        load();
    }

    // Call once after every pass with the experts it used (key, uses; a key may repeat). Updates the scores.
    void step(const std::vector<std::pair<int, uint32_t>>& used) {
        static const float tau[NT] = {8.f, 32.f, 128.f, 512.f};
        for (int i = 0; i < NT; i++) {
            const float k = 1.f - 1.f / tau[i];
            for (float& h : heat_[i]) h *= k;
        }
        for (auto [key, c] : used) {
            for (int i = 0; i < NT; i++) heat_[i][key] += (float)c;
            count_[key] += (float)c;
            last_[key] = (float)t_;
        }
        hist_.push_back(used);
        if ((int)hist_.size() > DELAY + 1) hist_.pop_front();
        t_++;

        // features, context sketch, scores
        float z[CD] = {};
        for (size_t k = 0; k < K_; k++) {
            float* x = &X_[k * NF];
            for (int i = 0; i < NT; i++) x[i] = std::log1p(heat_[i][k]);
            x[4] = std::log1p(count_[k]);
            x[5] = prior_[k];
            x[6] = 1.f / (1.f + ((float)t_ - last_[k]) / 16.f);
            x[7] = 1.f;
        }
        for (int d = 0; d < CD; d++) {
            const float* p = &P_[(size_t)d * K_];
            float s = 0;
            for (size_t k = 0; k < K_; k++) s += p[k] * X_[k * NF + 1];
            z[d] = s;
        }
        float nz = 1e-6f;
        for (float a : z) nz += a * a;
        nz = 1.f / std::sqrt(nz);
        for (float& a : z) a *= nz;
        for (size_t k = 0; k < K_; k++) score_[k] = shared(&X_[k * NF]) + readout(k, z);

        // this pass's samples: the experts it used and some random ones; their labels arrive DELAY passes later
        Sample s;
        s.t = t_;
        std::memcpy(s.z, z, sizeof z);
        for (auto [key, c] : used) s.keys.push_back(key);
        std::sort(s.keys.begin(), s.keys.end());
        s.keys.erase(std::unique(s.keys.begin(), s.keys.end()), s.keys.end());
        if ((int)s.keys.size() > MAX_USED) {
            std::shuffle(s.keys.begin(), s.keys.end(), rng_);
            s.keys.resize(MAX_USED);
        }
        for (int i = 0; i < N_RANDOM; i++) s.keys.push_back((int)(rng_() % K_));
        s.x.resize(s.keys.size() * NF);
        for (size_t i = 0; i < s.keys.size(); i++) std::memcpy(&s.x[i * NF], &X_[(size_t)s.keys[i] * NF], NF * sizeof(float));
        queue_.push_back(std::move(s));
        while (!queue_.empty() && queue_.front().t + DELAY <= t_) {
            train(queue_.front());
            queue_.pop_front();
        }
    }

    float score(int key) const { return score_[key]; }
    uint64_t trained() const { return steps_; }
    float loss() const { return loss_; }  // running mean of the squared error on training samples

    void save() const {
        if (path_.empty() || !steps_) return;
        FILE* f = std::fopen(path_.c_str(), "wb");
        if (!f) return;
        const uint64_t head[4] = {MAGIC, K_, steps_, (uint64_t)NP};
        std::fwrite(head, 8, 4, f);
        std::fwrite(th_.data(), 4, NP, f);
        std::fwrite(m_.data(), 4, NP, f);
        std::fwrite(v_.data(), 4, NP, f);
        std::fwrite(V_.data(), 4, V_.size(), f);
        std::fwrite(b_.data(), 4, K_, f);
        std::fwrite(G_.data(), 4, K_, f);
        std::fclose(f);
    }

private:
    static constexpr uint64_t MAGIC = 0x3144455250455851ull;  // "QXEPRED1"
    struct Sample {
        uint64_t t;
        float z[CD];
        std::vector<int> keys;
        std::vector<float> x;
    };

    void load() {
        if (path_.empty()) return;
        FILE* f = std::fopen(path_.c_str(), "rb");
        if (!f) return;
        uint64_t head[4];
        std::vector<float> th(NP), m(NP), v(NP), V(V_.size()), b(K_), G(K_);
        const bool ok = std::fread(head, 8, 4, f) == 4 && head[0] == MAGIC && head[1] == K_ && head[3] == (uint64_t)NP &&
                        std::fread(th.data(), 4, NP, f) == (size_t)NP && std::fread(m.data(), 4, NP, f) == (size_t)NP &&
                        std::fread(v.data(), 4, NP, f) == (size_t)NP && std::fread(V.data(), 4, V.size(), f) == V.size() &&
                        std::fread(b.data(), 4, K_, f) == K_ && std::fread(G.data(), 4, K_, f) == K_;
        std::fclose(f);
        if (!ok) return;
        th_ = th; m_ = m; v_ = v; V_ = V; b_ = b; G_ = G;
        steps_ = head[2];
    }

    // shared part: w2 . relu(W1 x + b1) + b2 + skip . x; h receives the hidden activations when given
    float shared(const float* x, float* h = nullptr) const {
        const float* W1 = th_.data();
        const float* b1 = W1 + HID * NF;
        const float* w2 = b1 + HID;
        const float* skip = w2 + HID + 1;
        float out = w2[HID];
        for (int j = 0; j < HID; j++) {
            float a = b1[j];
            for (int f = 0; f < NF; f++) a += W1[j * NF + f] * x[f];
            a = a > 0 ? a : 0;
            if (h) h[j] = a;
            out += w2[j] * a;
        }
        for (int f = 0; f < NF; f++) out += skip[f] * x[f];
        return out;
    }
    float readout(size_t k, const float* z) const {
        const float* v = &V_[k * CD];
        float s = b_[k];
        for (int d = 0; d < CD; d++) s += v[d] * z[d];
        return s;
    }

    void train(const Sample& s) {
        // labels: discounted uses over the DELAY passes after the sample was taken
        const size_t n_hist = hist_.size();
        const uint64_t first = t_ - n_hist;  // hist_[i] is pass first + i (0-based); the sample saw passes < s.t
        float w = 1.f;
        for (uint64_t p = s.t; p < t_; p++, w *= GAMMA)
            for (auto [key, c] : hist_[(size_t)(p - first)]) y_[key] += w * (float)c;
        const size_t n = s.keys.size();
        std::vector<float> g(NP, 0.f);
        const float* w2 = th_.data() + HID * NF + HID;
        double se = 0;
        for (size_t i = 0; i < n; i++) {
            const int key = s.keys[i];
            const float* x = &s.x[i * NF];
            float h[HID];
            const float base = shared(x, h);
            const float target = std::log1p(y_[key]);
            const float e = std::clamp(base + readout(key, s.z) - target, -3.f, 3.f);
            se += (double)e * e;
            // per-expert readout: Adagrad
            G_[key] += e * e;
            const float st = 0.3f * e / std::sqrt(G_[key]);
            float* v = &V_[(size_t)key * CD];
            for (int d = 0; d < CD; d++) v[d] -= st * s.z[d];
            b_[key] -= st;
            // shared part: gradient of the squared error of the shared prediction alone
            const float es = std::clamp(base - target, -3.f, 3.f) / (float)n;
            for (int j = 0; j < HID; j++) {
                g[HID * NF + HID + j] += es * h[j];
                if (h[j] > 0) {
                    const float gj = es * w2[j];
                    g[HID * NF + j] += gj;
                    for (int f = 0; f < NF; f++) g[j * NF + f] += gj * x[f];
                }
            }
            g[HID * NF + HID + HID] += es;
            for (int f = 0; f < NF; f++) g[NP - NF + f] += es * x[f];
        }
        for (uint64_t p = s.t; p < t_; p++)
            for (auto [key, c] : hist_[(size_t)(p - first)]) y_[key] = 0;
        steps_++;
        const float c1 = 1.f - std::pow(0.9f, (float)std::min<uint64_t>(steps_, 10000));
        const float c2 = 1.f - std::pow(0.999f, (float)std::min<uint64_t>(steps_, 100000));
        for (int i = 0; i < NP; i++) {
            m_[i] = 0.9f * m_[i] + 0.1f * g[i];
            v_[i] = 0.999f * v_[i] + 0.001f * g[i] * g[i];
            th_[i] -= 0.003f * (m_[i] / c1) / (std::sqrt(v_[i] / c2) + 1e-8f);
        }
        loss_ = loss_ == 0 ? (float)(se / n) : 0.98f * loss_ + 0.02f * (float)(se / n);
    }

    size_t K_ = 0;
    std::string path_;
    uint64_t t_ = 0, steps_ = 0;
    float loss_ = 0;
    std::vector<float> heat_[NT], count_, last_, prior_, score_, X_, y_;
    std::vector<float> P_, V_, b_, G_;  // context projection [CD][K]; per-expert readout [K][CD], bias, Adagrad sum
    std::vector<float> th_, m_, v_;     // shared weights and their Adam moments
    std::deque<std::vector<std::pair<int, uint32_t>>> hist_;
    std::deque<Sample> queue_;
    std::mt19937 rng_{7};
};
