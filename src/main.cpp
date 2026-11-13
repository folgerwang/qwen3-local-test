// qwen: local inference for Qwen MoE models with CPU and Vulkan backends: Qwen3.8-Flash-Next from a GGUF
// (UD-Q4_K_XL by default) and Qwen3.5/3.6 from FP8 safetensors.
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "backend.h"
#include "gguf.h"
#include "json.h"
#include "model.h"
#include "qwen4.h"
#include "safetensors.h"
#include "tokenizer.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace {

struct Options {
    std::string model_dir = "models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL";
    std::string backend = "cpu";
    std::string prompt;
    std::string system;
    std::string tokenize;
    std::string tokenize_file;
    std::string eval_ids;
    int eval_split = 0;
    int max_ctx = 0;  // 0 = the model's default: 262144 for the GGUF model, 32768 for FP8
    int max_new = 4096;
    int threads = 0;
    size_t expert_cache_mb = 0;  // 0 = auto
    bool expert_cache_set = false;
    float temperature = 1.0f, top_p = 0.95f;
    int top_k = 20;
    uint64_t seed = 0;
    bool think = true;
    bool greedy = false;
    bool no_pin = false;
    int hybrid = -1;
    bool keep_think = false;
    bool drop_think = false;
    int draft = 3;  // tokens drafted per pass with the MTP head (GGUF + vulkan), 0 = off
    std::string mtp_path;
    bool hide_think = false;
    bool no_stats = false;
    bool pipe = false;
    bool stream = false;         // pipe mode: write {"delta":...} lines while generating
    std::string interrupt_file;  // pipe mode: generation stops when this file appears
};

void usage() {
    std::printf(
        "usage: qwen [options]\n"
        "  --model PATH         directory of *.gguf shards (or one .gguf file), or an FP8 safetensors model\n"
        "                       directory (default models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL)\n"
        "  --backend cpu|vulkan inference backend (default cpu)\n"
        "  -p, --prompt TEXT    single prompt, then exit (otherwise interactive chat)\n"
        "  -f, --prompt-file F  read the single prompt from a file\n"
        "  --system TEXT        system prompt\n"
        "  --no-think           disable thinking mode\n"
        "  --hide-think         don't print the reasoning, show thinking... (a dot every 2s) instead (/think toggles)\n"
        "  --keep-think         keep earlier turns' reasoning in the history\n"
        "  --drop-think         drop it (default: what the model's chat template does: keep for\n"
        "                       Qwen3.8-Flash-Next, drop for Qwen3.5/3.6)\n"
        "  -n N                 max new tokens per reply (default 4096)\n"
        "  --ctx N              context length (default 262144 for the GGUF model, 32768 for FP8)\n"
        "  --temp T --top-k K --top-p P --seed S --greedy\n"
        "  --threads N          CPU threads (default: all)\n"
        "  --expert-cache MB    GPU expert cache size (vulkan, default: auto)\n"
        "  --draft N            tokens drafted per pass by the MTP draft head (GGUF model on vulkan, default 3,\n"
        "                       0 = off); needs mtp-*.gguf next to or one directory above the model\n"
        "  --mtp FILE           the draft head GGUF, if it is somewhere else\n"
        "  --pipe               serve chat turns over stdin/stdout, one JSON object per line (see sicode/README.md)\n"
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

    // Length of the longest prefix of s that does not end inside a UTF-8 sequence.
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

private:
    std::string pending_;
};

// A JSON string literal. Bytes >= 0x80 pass through (the text is UTF-8).
std::string json_quote(const std::string& s) {
    std::string r = "\"";
    for (unsigned char c : s) {
        if (c == '"') r += "\\\"";
        else if (c == '\\') r += "\\\\";
        else if (c == '\n') r += "\\n";
        else if (c == '\r') r += "\\r";
        else if (c == '\t') r += "\\t";
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); r += b; }
        else r += (char)c;
    }
    return r + "\"";
}

struct Turn {
    std::string text;  // everything generated, reasoning included
    int n_gen = 0;
    size_t n_prefill = 0;
    bool ended = false;  // the model closed its turn (false: token limit or context full)
    double prefill_s = 0, gen_s = 0;
    bool interrupted = false;
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
        else if (a == "--keep-think") o.keep_think = true;
        else if (a == "--drop-think") o.drop_think = true;
        else if (a == "--hide-think") o.hide_think = true;
        else if (a == "-n") o.max_new = std::stoi(next());
        else if (a == "--ctx") o.max_ctx = std::stoi(next());
        else if (a == "--temp") o.temperature = std::stof(next());
        else if (a == "--top-k") o.top_k = std::stoi(next());
        else if (a == "--top-p") o.top_p = std::stof(next());
        else if (a == "--seed") o.seed = std::stoull(next());
        else if (a == "--greedy") o.greedy = true;
        else if (a == "--threads") o.threads = std::stoi(next());
        else if (a == "--expert-cache") { o.expert_cache_mb = std::stoull(next()); o.expert_cache_set = true; }
        else if (a == "--draft") o.draft = std::stoi(next());
        else if (a == "--mtp") o.mtp_path = next();
        else if (a == "--no-pin") o.no_pin = true;
        else if (a == "--hybrid") o.hybrid = 1;
        else if (a == "--no-hybrid") o.hybrid = 0;
        else if (a == "--no-stats") o.no_stats = true;
        else if (a == "--pipe") o.pipe = true;
        else if (a == "--tokenize") o.tokenize = next();
        else if (a == "--tokenize-file") o.tokenize_file = next();
        else if (a == "--eval") o.eval_ids = next();
        else if (a == "--eval-split") o.eval_split = std::stoi(next());
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 1; }
    }

    try {
        // A GGUF (file or directory of shards) selects the qwen4exp path; otherwise FP8 safetensors.
        std::unique_ptr<Gguf> gguf;
        if (Gguf::is_gguf_path(o.model_dir)) gguf = std::make_unique<Gguf>(o.model_dir);
        const std::string state_dir = gguf ? gguf->dir() : o.model_dir;
        // The Qwen3.8 chat template renders earlier assistant turns with their reasoning (preserve_thinking).
        if (gguf) o.keep_think = true;
        if (o.drop_think) o.keep_think = false;
        Tokenizer tok;
        if (gguf)
            tok.load_gguf(gguf->kv("tokenizer.ggml.tokens").strs, gguf->kv("tokenizer.ggml.token_type").ints,
                          gguf->kv("tokenizer.ggml.merges").strs);
        else
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
        // The configs and weight views must outlive the backend, which keeps references to them.
        Config fcfg;
        ModelWeights fw;
        std::unique_ptr<SafeTensors> st;
        std::unique_ptr<Gguf> mtp;
        Q4Config qcfg;
        Q4Weights qw;
        struct { int vocab = 0, n_layers = 0; size_t n_tensors = 0; std::vector<int> eos_ids; } cfg;
        std::unique_ptr<Backend> be;
        if (gguf) {
            gguf->check_complete();
            qcfg = Q4Config::load(*gguf);
            qw.load(*gguf, qcfg);
            // MTP draft head for speculative decoding (vulkan): mtp-*.gguf in the model directory's parent (the
            // Hugging Face layout) or wherever --mtp points
            if (o.draft > 0 && o.backend != "cpu") {
                namespace fs = std::filesystem;
                std::string mp = o.mtp_path;
                std::error_code ec;
                if (mp.empty())
                    for (auto& e : fs::directory_iterator(fs::path(gguf->dir()).parent_path(), ec)) {
                        const std::string n = e.path().filename().string();
                        if (n.rfind("mtp-", 0) == 0 && e.path().extension() == ".gguf") mp = e.path().string();
                    }
                if (!mp.empty()) {
                    mtp = std::make_unique<Gguf>(mp);
                    qw.load_mtp(*mtp, qcfg);
                } else {
                    std::fprintf(stderr, "[no mtp-*.gguf draft head found: speculative decoding is off]\n");
                }
            }
            cfg = {qcfg.vocab, qcfg.n_layers, gguf->count(), qcfg.eos_ids};
            const int ctx = o.max_ctx ? o.max_ctx : 262144;
            if (o.backend == "cpu") be = make_qwen4_cpu_backend(qcfg, qw, ctx, o.threads);
#ifdef QW_VULKAN
            else if (o.backend == "vulkan" || o.backend == "vk") {
                VulkanOptions vo;
                vo.expert_cache_mb = o.expert_cache_mb;
                vo.stats_path = o.no_stats ? "" : state_dir + "/expert_usage.bin";
                vo.pin_experts = !o.no_pin;
                vo.hybrid = o.hybrid;
                vo.expert_cache_set = o.expert_cache_set;
                vo.threads = o.threads;
                vo.draft = o.draft;
                be = make_qwen4_vulkan_backend(qcfg, qw, ctx, vo);
            }
#endif
            else throw std::runtime_error("unknown/unavailable backend: " + o.backend);
        } else {
            fcfg = Config::load(o.model_dir);
            st = std::make_unique<SafeTensors>(o.model_dir);
            fw.load(*st, fcfg);
            cfg = {fcfg.vocab, fcfg.n_layers, st->count(), fcfg.eos_ids};
            const int ctx = o.max_ctx ? o.max_ctx : 32768;
            if (o.backend == "cpu") be = make_cpu_backend(fcfg, fw, ctx, o.threads);
#ifdef QW_VULKAN
            else if (o.backend == "vulkan" || o.backend == "vk") {
                VulkanOptions vo;
                vo.expert_cache_mb = o.expert_cache_mb;
                vo.stats_path = o.no_stats ? "" : state_dir + "/expert_usage.bin";
                vo.pin_experts = !o.no_pin;
                vo.hybrid = o.hybrid;
                be = make_vulkan_backend(fcfg, fw, ctx, vo);
            }
#endif
            else throw std::runtime_error("unknown/unavailable backend: " + o.backend);
        }
        double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::fprintf(stderr, "[%s backend, %d layers, %zu tensors, ready in %.1fs]\n", be->name(), cfg.n_layers,
                     cfg.n_tensors, load_s);

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
        // `quiet` prints nothing to stdout (the pipe mode returns the text instead).
        auto run_turn = [&](const std::string& msg, bool first, bool quiet = false) -> Turn {
            std::string head;
            if (first && !o.system.empty()) head += "<|im_start|>system\n" + o.system + "<|im_end|>\n";
            if (!first) head += "\n";  // newline after the previous <|im_end|>
            head += "<|im_start|>user\n" + msg + "<|im_end|>\n<|im_start|>assistant\n";
            std::vector<int> ids = tok.encode(head);
            const std::vector<int> gen_prompt = tok.encode(o.think ? "<think>\n" : "<think>\n\n</think>\n\n");
            if (be->pos() + (int)(ids.size() + gen_prompt.size()) >= be->max_ctx()) throw std::runtime_error("context full");

            auto ts = std::chrono::steady_clock::now();
            be->forward(ids.data(), (int)ids.size(), logits.data());
            if (!o.keep_think) be->save_state();
            be->forward(gen_prompt.data(), (int)gen_prompt.size(), logits.data());
            const size_t n_prefill = ids.size() + gen_prompt.size();
            double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
            // With hide_think, reasoning is not printed: "thinking" gets a dot every 2 seconds until </think>.
            bool hiding = o.think && o.hide_think && !quiet;
            if (o.think && !hiding && !quiet) std::printf("<think>\n");
            int dots = 0;
            bool skip_nl = false;
            auto last_dot = std::chrono::steady_clock::now();
            if (hiding) { std::printf("thinking"); std::fflush(stdout); }

            Streamer out;
            std::string text;  // generated bytes (excluding the EOS token)
            int n_gen = 0, n_passes = 0, n_drafted = 0;
            bool ended = false, interrupted = false;
            std::string spend;  // streamed bytes waiting for the rest of a UTF-8 sequence
            std::vector<float> all_logits;
            double t_draft = 0, t_verify = 0, t_accept = 0;  // seconds per stage of the speculative passes
            auto tg = std::chrono::steady_clock::now();
            // prints one generated token
            auto emit = [&](int id) {
                const std::string piece = tok.decode(id);
                text += piece;
                if (quiet) {
                    if (o.stream) {
                        spend += piece;
                        const size_t ok = Streamer::complete_prefix(spend);
                        if (ok) {
                            const std::string line = "{\"delta\":" + json_quote(spend.substr(0, ok)) + "}\n";
                            std::fwrite(line.data(), 1, line.size(), stdout);
                            std::fflush(stdout);
                            spend.erase(0, ok);
                        }
                    }
                    return;
                }
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
            };
            // `next` is always a token sampled from the model's own distribution at the current position. With a
            // draft head, the tokens it proposes after `next` are fed in the same pass; each one is kept exactly
            // when it equals the token sampled from the model's logits at its position, so the output is
            // distributed as without drafting.
            std::vector<int> seq(be->max_draft() + 1);
            int next = sampler.sample(logits.data(), cfg.vocab);
            while (n_gen < o.max_new) {
                if (is_eos(next)) { ended = next == im_end; break; }
                if (!o.interrupt_file.empty() && std::filesystem::exists(o.interrupt_file)) { interrupted = true; break; }
                emit(next);
                n_gen++;
                if (be->pos() + 2 >= be->max_ctx()) { if (!quiet) std::printf("\n[context full]"); break; }
                seq[0] = next;
                const int room = std::min(o.max_new - n_gen, be->max_ctx() - be->pos() - 3);
                auto td0 = std::chrono::steady_clock::now();
                const int nd = be->max_draft() > 0 && room > 0 ? be->draft(next, std::min(be->max_draft(), room), seq.data() + 1) : 0;
                auto td1 = std::chrono::steady_clock::now();
                t_draft += std::chrono::duration<double>(td1 - td0).count();
                if (nd == 0) {
                    be->forward(&next, 1, logits.data());
                    next = sampler.sample(logits.data(), cfg.vocab);
                    continue;
                }
                n_passes++;
                all_logits.resize((size_t)(nd + 1) * cfg.vocab);
                be->forward_all(seq.data(), nd + 1, all_logits.data());
                auto td2 = std::chrono::steady_clock::now();
                t_verify += std::chrono::duration<double>(td2 - td1).count();
                int keep = 1;
                for (;; keep++) {
                    next = sampler.sample(&all_logits[(size_t)(keep - 1) * cfg.vocab], cfg.vocab);
                    if (keep > nd || next != seq[keep] || is_eos(next)) break;
                    emit(next);
                    n_gen++;
                    n_drafted++;
                }
                be->accept(keep);
                t_accept += std::chrono::duration<double>(std::chrono::steady_clock::now() - td2).count();
            }
            if (hiding) std::printf("\n[reasoning did not finish within %d tokens; raise -n]", o.max_new);
            if (!quiet) out.flush();
            double gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tg).count();

            if (o.keep_think) {
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
            std::fprintf(stderr, "\n[prefill %zu tok %.1f tok/s | gen %d tok %.2f tok/s | ctx %d", n_prefill,
                         n_prefill / prefill_s, n_gen, n_gen / std::max(gen_s, 1e-9), be->pos());
            if (n_passes) std::fprintf(stderr, " | %d of %d tokens from accepted drafts, %d draft passes", n_drafted, n_gen, n_passes);
            if (n_passes && std::getenv("QW_PROFILE")) std::fprintf(stderr, " (draft %.2fs, verify %.2fs, accept %.2fs)", t_draft, t_verify, t_accept);
            std::fprintf(stderr, "]\n");
            return {text, n_gen, n_prefill, ended, prefill_s, gen_s, interrupted};
        };

        if (!o.prompt.empty()) {
            run_turn(o.prompt, true);
            return 0;
        }
        if (o.pipe) {
            // One request per stdin line, one reply per stdout line. {"op":"reset"} clears the conversation;
            // {"user":TEXT, "system":TEXT, "max_tokens":N, "think":BOOL, "stream":BOOL, "interrupt_file":PATH}
            // appends a user turn and generates the reply (system only counts on the first turn after a reset).
            // With stream, {"delta":TEXT} lines precede the reply; generation stops early once interrupt_file exists.
            std::printf("{\"ready\":true,\"ctx\":%d}\n", be->max_ctx());
            std::fflush(stdout);
            const Options base = o;
            bool first = true;
            std::string line;
            while (std::getline(std::cin, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty()) continue;
                std::string reply;
                try {
                    const json::Value req = json::parse(line);
                    const json::Value* op = req.find("op");
                    if (op && op->as_str() == "reset") {
                        be->reset();
                        first = true;
                        reply = "{\"ok\":true}";
                    } else {
                        const json::Value* sys = req.find("system");
                        const json::Value* th = req.find("think");
                        o.system = sys ? sys->as_str() : base.system;
                        o.think = th ? th->b : base.think;
                        o.max_new = (int)req.get_int("max_tokens", base.max_new);
                        const json::Value* stm = req.find("stream");
                        const json::Value* intr = req.find("interrupt_file");
                        o.stream = stm && stm->b;
                        o.interrupt_file = intr ? intr->as_str() : "";
                        const Turn t = run_turn(req["user"].as_str(), first, true);
                        first = false;
                        char num[256];
                        std::snprintf(num, sizeof num, ",\"finished\":%s,\"interrupted\":%s,\"gen_tokens\":%d,\"prefill_tokens\":%zu,"
                                      "\"gen_tps\":%.2f,\"prefill_tps\":%.1f,\"ctx_used\":%d,\"ctx\":%d}",
                                      t.ended ? "true" : "false", t.interrupted ? "true" : "false", t.n_gen, t.n_prefill, t.n_gen / std::max(t.gen_s, 1e-9),
                                      t.n_prefill / std::max(t.prefill_s, 1e-9), be->pos(), be->max_ctx());
                        reply = "{\"text\":" + json_quote(t.text) + num;
                    }
                } catch (const std::exception& e) {
                    reply = "{\"error\":" + json_quote(e.what()) + "}";
                }
                std::fwrite(reply.data(), 1, reply.size(), stdout);
                std::fputc('\n', stdout);
                std::fflush(stdout);
            }
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
