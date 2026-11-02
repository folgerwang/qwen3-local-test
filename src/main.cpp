// qwen: local inference for Qwen3.5/3.6 MoE (FP8) with CPU and Vulkan backends.
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "backend.h"
#include "json.h"
#include "model.h"
#include "safetensors.h"
#include "tokenizer.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace {

struct Options {
    std::string model_dir = "models/Qwen3.6-35B-A3B-FP8";
    std::string backend = "cpu";
    std::string prompt;
    std::string system;
    std::string tokenize;
    std::string tokenize_file;
    std::string eval_ids;
    int eval_split = 0;
    int max_ctx = 32768;
    int max_new = 4096;
    int threads = 0;
    size_t expert_cache_mb = 0;  // 0 = auto
    float temperature = 1.0f, top_p = 0.95f;
    int top_k = 20;
    uint64_t seed = 0;
    bool think = true;
    bool greedy = false;
    bool no_pin = false;
    int hybrid = -1;
    int keep_think = -1;  // -1: model default (qwen4 templates keep it, Qwen3.5/3.6 drop it)
    bool hide_think = false;
    std::string effort = "xhigh";  // qwen4 reasoning effort: xhigh | medium | low
    bool no_stats = false;
};

void usage() {
    std::printf(
        "usage: qwen [options]\n"
        "  --model DIR          model directory (default models/Qwen3.6-35B-A3B-FP8)\n"
        "  --backend cpu|vulkan inference backend (default cpu)\n"
        "  -p, --prompt TEXT    single prompt, then exit (otherwise interactive chat)\n"
        "  -f, --prompt-file F  read the single prompt from a file\n"
        "  --system TEXT        system prompt\n"
        "  --no-think           disable thinking mode\n"
        "  --hide-think         don't print the reasoning, show thinking... (a dot every 2s) instead (/think toggles)\n"
        "  --keep-think         keep earlier turns' reasoning in the history (default: as the model's template)\n"
        "  --drop-think         drop earlier turns' reasoning from the history\n"
        "  --effort LEVEL       reasoning effort for Qwen3.8-Flash-Next: xhigh (default), medium, low\n"
        "  -n N                 max new tokens per reply (default 4096)\n"
        "  --ctx N              context length (default 32768)\n"
        "  --temp T --top-k K --top-p P --seed S --greedy\n"
        "  --threads N          CPU threads (default: all)\n"
        "  --expert-cache MB    GPU expert cache size (vulkan, default: auto)\n"
        "  --tokenize TEXT      print token ids and exit\n"
        "  --eval IDS           comma-separated ids: run and print top-10 logits of last position\n");
}

#ifdef _WIN32
bool read_line(std::string& out) {
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    if (!GetConsoleMode(h, &mode)) return (bool)std::getline(std::cin, out);
    std::wstring w;
    wchar_t buf[1024];
    for (;;) {
        DWORD n = 0;
        if (!ReadConsoleW(h, buf, 1024, &n, nullptr) || n == 0) return false;
        w.append(buf, n);
        if (!w.empty() && w.back() == L'\n') break;
    }
    while (!w.empty() && (w.back() == L'\n' || w.back() == L'\r')) w.pop_back();
    int len = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    out.assign(len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), out.data(), len, nullptr, nullptr);
    return true;
}
#else
bool read_line(std::string& out) { return (bool)std::getline(std::cin, out); }
#endif

class Sampler {
public:
    explicit Sampler(const Options& o) : o_(o), rng_(o.seed ? o.seed : std::random_device{}()) {}
    int sample(const float* logits, int vocab) {
        if (o_.greedy || o_.temperature <= 0) return (int)(std::max_element(logits, logits + vocab) - logits);
        const int k = std::min(o_.top_k > 0 ? o_.top_k : vocab, vocab);
        idx_.resize(vocab);
        for (int i = 0; i < vocab; i++) idx_[i] = i;
        std::partial_sort(idx_.begin(), idx_.begin() + k, idx_.end(), [&](int a, int b) { return logits[a] > logits[b]; });
        std::vector<double> p(k);
        double sum = 0;
        for (int i = 0; i < k; i++) { p[i] = std::exp((logits[idx_[i]] - logits[idx_[0]]) / o_.temperature); sum += p[i]; }
        double cum = 0;
        int cut = k;
        for (int i = 0; i < k; i++) {
            cum += p[i] / sum;
            if (cum >= o_.top_p) { cut = i + 1; break; }
        }
        std::discrete_distribution<int> dist(p.begin(), p.begin() + cut);
        return idx_[dist(rng_)];
    }

private:
    const Options& o_;
    std::mt19937_64 rng_;
    std::vector<int> idx_;
};

// Prints token bytes as they come, holding back incomplete UTF-8 sequences.
class Streamer {
public:
    void put(const std::string& bytes) {
        pending_ += bytes;
        size_t ok = complete_prefix(pending_);
        if (ok) {
            std::fwrite(pending_.data(), 1, ok, stdout);
            std::fflush(stdout);
            pending_.erase(0, ok);
        }
    }
    void flush() {
        std::fwrite(pending_.data(), 1, pending_.size(), stdout);
        std::fflush(stdout);
        pending_.clear();
    }

private:
    static size_t complete_prefix(const std::string& s) {
        size_t n = s.size(), i = n;
        // walk back over at most 3 continuation bytes to find the last lead byte
        size_t back = 0;
        while (i > 0 && back < 4 && ((uint8_t)s[i - 1] & 0xC0) == 0x80) { i--; back++; }
        if (i == 0) return back >= 4 ? n : 0;
        uint8_t lead = (uint8_t)s[i - 1];
        size_t need = lead < 0x80 ? 1 : (lead >> 5) == 6 ? 2 : (lead >> 4) == 14 ? 3 : (lead >> 3) == 30 ? 4 : 1;
        return (n - (i - 1) >= need) ? n : i - 1;
    }
    std::string pending_;
};

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    Options o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(1); }
            return argv[++i];
        };
        if (a == "--model") o.model_dir = next();
        else if (a == "--backend") o.backend = next();
        else if (a == "-p" || a == "--prompt") o.prompt = next();
        else if (a == "-f" || a == "--prompt-file") {
            std::ifstream f(next(), std::ios::binary);
            if (!f) { std::fprintf(stderr, "cannot read prompt file\n"); return 1; }
            o.prompt.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        }
        else if (a == "--system") o.system = next();
        else if (a == "--no-think") o.think = false;
        else if (a == "--keep-think") o.keep_think = 1;
        else if (a == "--drop-think") o.keep_think = 0;
        else if (a == "--effort") o.effort = next();
        else if (a == "--hide-think") o.hide_think = true;
        else if (a == "-n") o.max_new = std::stoi(next());
        else if (a == "--ctx") o.max_ctx = std::stoi(next());
        else if (a == "--temp") o.temperature = std::stof(next());
        else if (a == "--top-k") o.top_k = std::stoi(next());
        else if (a == "--top-p") o.top_p = std::stof(next());
        else if (a == "--seed") o.seed = std::stoull(next());
        else if (a == "--greedy") o.greedy = true;
        else if (a == "--threads") o.threads = std::stoi(next());
        else if (a == "--expert-cache") o.expert_cache_mb = std::stoull(next());
        else if (a == "--no-pin") o.no_pin = true;
        else if (a == "--hybrid") o.hybrid = 1;
        else if (a == "--no-hybrid") o.hybrid = 0;
        else if (a == "--no-stats") o.no_stats = true;
        else if (a == "--tokenize") o.tokenize = next();
        else if (a == "--tokenize-file") o.tokenize_file = next();
        else if (a == "--eval") o.eval_ids = next();
        else if (a == "--eval-split") o.eval_split = std::stoi(next());
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 1; }
    }

    try {
        Tokenizer tok;
        tok.load(o.model_dir);
        if (!o.tokenize.empty()) {
            for (int id : tok.encode(o.tokenize)) std::printf("%d ", id);
            std::printf("\n");
            return 0;
        }
        if (!o.tokenize_file.empty()) {  // JSON array of strings -> one line of ids per string
            std::ifstream f(o.tokenize_file, std::ios::binary);
            std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            for (auto& v : json::parse(text).arr) {
                for (int id : tok.encode(v.as_str())) std::printf("%d ", id);
                std::printf("\n");
            }
            return 0;
        }

        auto t0 = std::chrono::steady_clock::now();
        Config cfg = Config::load(o.model_dir);
        SafeTensors st(o.model_dir);
        ModelWeights w;
        w.load(st, cfg);
        std::unique_ptr<Backend> be;
        if (o.backend == "cpu") be = make_cpu_backend(cfg, w, o.max_ctx, o.threads);
#ifdef QW_VULKAN
        else if (o.backend == "vulkan" || o.backend == "vk") {
            VulkanOptions vo;
            vo.expert_cache_mb = o.expert_cache_mb;
            vo.stats_path = o.no_stats ? "" : o.model_dir + "/expert_usage.bin";
            vo.pin_experts = !o.no_pin;
            vo.hybrid = o.hybrid;
            be = make_vulkan_backend(cfg, w, o.max_ctx, vo);
        }
#endif
        else throw std::runtime_error("unknown/unavailable backend: " + o.backend);
        double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[%s backend, %d layers, %zu tensors, ready in %.1fs]\n", be->name(), cfg.n_layers,
                     st.count(), load_s);

        std::vector<float> logits(cfg.vocab);

        if (!o.eval_ids.empty()) {
            std::vector<int> ids;
            for (size_t p = 0; p < o.eval_ids.size();) {
                size_t q = o.eval_ids.find(',', p);
                if (q == std::string::npos) q = o.eval_ids.size();
                ids.push_back(std::stoi(o.eval_ids.substr(p, q - p)));
                p = q + 1;
            }
            // prefill all but the last `eval_split` tokens, then feed those one at a time (decode path)
            const int split = std::min<int>(o.eval_split, (int)ids.size() - 1);
            be->forward(ids.data(), (int)ids.size() - split, logits.data());
            for (int i = (int)ids.size() - split; i < (int)ids.size(); i++) be->forward(&ids[i], 1, logits.data());
            std::vector<int> idx(cfg.vocab);
            for (int i = 0; i < cfg.vocab; i++) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
            for (int i = 0; i < 10; i++) std::printf("%d %.4f\n", idx[i], logits[idx[i]]);
            return 0;
        }

        Sampler sampler(o);
        const int im_end = tok.token_id("<|im_end|>");
        auto is_eos = [&](int id) { return std::find(cfg.eos_ids.begin(), cfg.eos_ids.end(), id) != cfg.eos_ids.end(); };

        // One chat turn. The history is kept in the form the official chat template renders it: earlier
        // assistant turns appear without their <think> block. The recurrent state is snapshotted right after
        // "<|im_start|>assistant\n"; once the reply is done we roll back to it and feed only the final answer.
        const bool keep_think = o.keep_think < 0 ? cfg.qwen4 : o.keep_think == 1;
        // qwen4 template: with thinking on, a reasoning-effort instruction leads the system message
        std::string instr;
        if (cfg.qwen4 && o.think) {
            if (o.effort == "xhigh")
                instr = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
                        "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
            else if (o.effort == "low")
                instr = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
                        "conclusion without unnecessary elaboration.";
            else if (o.effort != "medium") throw std::runtime_error("--effort must be xhigh, medium or low");
        }
        auto run_turn = [&](const std::string& msg, bool first) {
            std::string head;
            const std::string sys = o.system.empty() ? instr : instr.empty() ? o.system : instr + "\n\n" + o.system;
            if (first && !sys.empty()) head += "<|im_start|>system\n" + sys + "<|im_end|>\n";
            if (!first) head += "\n";  // newline after the previous <|im_end|>
            head += "<|im_start|>user\n" + msg + "<|im_end|>\n<|im_start|>assistant\n";
            std::vector<int> ids = tok.encode(head);
            const std::vector<int> gen_prompt = tok.encode(o.think ? "<think>\n" : "<think>\n\n</think>\n\n");
            if (be->pos() + (int)(ids.size() + gen_prompt.size()) >= be->max_ctx()) throw std::runtime_error("context full");

            auto ts = std::chrono::steady_clock::now();
            be->forward(ids.data(), (int)ids.size(), logits.data());
            if (!keep_think) be->save_state();
            be->forward(gen_prompt.data(), (int)gen_prompt.size(), logits.data());
            const size_t n_prefill = ids.size() + gen_prompt.size();
            double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
            // With hide_think, reasoning is not printed: "thinking" gets a dot every 2 seconds until </think>.
            bool hiding = o.think && o.hide_think;
            if (o.think && !hiding) std::printf("<think>\n");
            int dots = 0;
            bool skip_nl = false;
            auto last_dot = std::chrono::steady_clock::now();
            if (hiding) { std::printf("thinking"); std::fflush(stdout); }

            Streamer out;
            std::string text;  // generated bytes (excluding the EOS token)
            int n_gen = 0;
            bool ended = false;
            auto tg = std::chrono::steady_clock::now();
            for (; n_gen < o.max_new; n_gen++) {
                int next = sampler.sample(logits.data(), cfg.vocab);
                if (is_eos(next)) { ended = next == im_end; break; }
                const std::string piece = tok.decode(next);
                text += piece;
                if (!hiding) {
                    if (skip_nl) {  // drop the newlines between </think> and the answer
                        const size_t b = piece.find_first_not_of('\n');
                        if (b != std::string::npos) {
                            skip_nl = false;
                            out.put(piece.substr(b));
                        }
                    } else {
                        out.put(piece);
                    }
                } else {
                    const size_t p = text.find("</think>");
                    if (p != std::string::npos) {
                        hiding = false;
                        std::printf("\r%*s\r", 8 + dots, "");  // clear the "thinking..." line
                        std::string answer = text.substr(p + 8);
                        const size_t b = answer.find_first_not_of('\n');
                        if (b == std::string::npos) skip_nl = true;
                        else out.put(answer.substr(b));
                    } else if (std::chrono::steady_clock::now() - last_dot >= std::chrono::seconds(2)) {
                        last_dot = std::chrono::steady_clock::now();
                        dots++;
                        std::printf(".");
                        std::fflush(stdout);
                    }
                }
                if (be->pos() + 2 >= be->max_ctx()) { std::printf("\n[context full]"); break; }
                be->forward(&next, 1, logits.data());
            }
            if (hiding) std::printf("\n[reasoning did not finish within %d tokens; raise -n]", o.max_new);
            out.flush();
            double gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tg).count();

            if (keep_think) {
                // history keeps the reasoning; just close the turn if the model did not
                if (!ended) be->forward(&im_end, 1, logits.data());
            } else {
                std::string content = text;
                size_t p = content.rfind("</think>");
                if (p != std::string::npos) content = content.substr(p + 8);
                else if (o.think) content.clear();  // reasoning never finished: no answer to keep
                size_t b = content.find_first_not_of("\n");
                content = b == std::string::npos ? "" : content.substr(b);
                while (!content.empty() && std::isspace((unsigned char)content.back())) content.pop_back();
                be->restore_state();
                std::vector<int> tail = tok.encode(content + "<|im_end|>");
                be->forward(tail.data(), (int)tail.size(), logits.data());
            }
            std::fprintf(stderr, "\n[prefill %zu tok %.1f tok/s | gen %d tok %.2f tok/s | ctx %d]\n", n_prefill,
                         n_prefill / prefill_s, n_gen, n_gen / std::max(gen_s, 1e-9), be->pos());
        };

        if (!o.prompt.empty()) {
            run_turn(o.prompt, true);
            return 0;
        }
        std::printf("Interactive chat. Commands: /reset, /think (show/hide reasoning), /exit\n");
        bool first = true;
        for (;;) {
            std::printf("\n> ");
            std::fflush(stdout);
            std::string line;
            if (!read_line(line) || line == "/exit") break;
            if (line.empty()) continue;
            if (line == "/reset") { be->reset(); first = true; std::printf("[reset]\n"); continue; }
            if (line == "/think") {
                o.hide_think = !o.hide_think;
                std::printf("[reasoning %s]\n", o.hide_think ? "hidden" : "shown");
                continue;
            }
            run_turn(line, first);
            first = false;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
