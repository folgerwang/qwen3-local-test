// Fixed-size spinning thread pool for low-latency parallel_for (hundreds of dispatches per token).
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <vector>

class ThreadPool {
public:
    explicit ThreadPool(int n = 0) {
        if (n <= 0) n = (int)std::thread::hardware_concurrency();
        n_ = n;
        for (int i = 1; i < n_; i++) workers_.emplace_back([this, i] { worker(i); });
    }
    ~ThreadPool() {
        stop_ = true;
        gen_.fetch_add(1, std::memory_order_release);
        for (auto& t : workers_) t.join();
    }
    int size() const { return n_; }

    // Runs fn(begin, end) over [0, n) split into contiguous chunks, one per thread.
    void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn, int64_t min_chunk = 1) {
        int nt = (int)std::min<int64_t>(n_, (n + min_chunk - 1) / min_chunk);
        if (nt <= 1) { if (n > 0) fn(0, n); return; }
        job_ = &fn;
        job_n_ = n;
        job_threads_ = nt;
        // Every worker acknowledges every dispatch (even without a chunk), so a late worker can never
        // observe the parameters of a newer dispatch.
        pending_.store(n_ - 1, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        run_chunk(0);
        while (pending_.load(std::memory_order_acquire) != 0) std::this_thread::yield();
    }

private:
    void run_chunk(int i) {
        int64_t b = job_n_ * i / job_threads_, e = job_n_ * (i + 1) / job_threads_;
        if (b < e) (*job_)(b, e);
    }
    void worker(int i) {
        uint64_t seen = 0;
        for (;;) {
            uint64_t g;
            int spins = 0;
            while ((g = gen_.load(std::memory_order_acquire)) == seen) {
                ++spins;
                if (spins > 2000000) std::this_thread::sleep_for(std::chrono::microseconds(200));
                else if (spins > 20000) std::this_thread::yield();
            }
            seen = g;
            if (stop_) return;
            if (i < job_threads_) run_chunk(i);
            pending_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
    int n_;
    std::vector<std::thread> workers_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> pending_{0};
    std::atomic<bool> stop_{false};
    const std::function<void(int64_t, int64_t)>* job_ = nullptr;
    int64_t job_n_ = 0;
    int job_threads_ = 0;
};
