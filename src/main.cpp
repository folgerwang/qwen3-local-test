// qwen: local inference for Qwen3.5/3.6 MoE (FP8) with CPU and Vulkan backends.
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iterator>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "backend.h"
#include "json.h"
#include "model.h"
#include "safetensors.h"
#include "sysmem.h"
#include "term.h"
#include "tokenizer.h"
#include "tools.h"
#include "vision.h"

namespace fs = std::filesystem;

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
    std::string resume;  // session to load at startup
    std::vector<std::string> images;  // attached to the -p prompt (or --eval, at the image token)
    int image_tokens = 1024;          // per-image token budget (32x32 pixels per token)
    bool no_vision = false;
    bool no_tools = false;
    std::string permission = "ask";  // ask | edits | all | read-only
    double vram_reserve_gb = 5.0;    // VRAM left free for other programs
    double ram_reserve_gb = 16.0;    // RAM left free (experts are not pinned if that would cut into it)
    std::string first_msg;           // positional prompt: sent as the first message of the interactive chat
    bool print = false;              // -p: answer `prompt` and exit
    std::string theme;               // --theme (otherwise the saved one)
    bool resume_pick = false;        // --resume without a name: choose from a menu
};

constexpr const char* kVersion = "0.3.0";

void usage() {
    std::printf(
        "Usage: qwen [options] [prompt]\n"
        "\n"
        "  qwen                      start an interactive chat\n"
        "  qwen \"question\"           start a chat with a first message\n"
        "  qwen -p \"question\"        answer and exit (also reads the prompt from a pipe: type file | qwen -p)\n"
        "  qwen -c                   continue the last conversation\n"
        "\n"
        "Session\n"
        "  -p, --print [TEXT]        print the answer to TEXT (or to piped input) and exit\n"
        "  -f, --prompt-file FILE    like -p, with the prompt read from a file\n"
        "  -c, --continue            resume the last conversation with this model\n"
        "  -r, --resume [NAME]       resume a saved conversation; without a name, choose from a menu\n"
        "  --system-prompt TEXT      system prompt (also --system, --append-system-prompt)\n"
        "  --image FILE              attach an image to the -p prompt (repeatable; vulkan backend)\n"
        "\n"
        "Model\n"
        "  --model NAME|DIR          model folder, or part of a downloaded model's name (35b, 122b, flash)\n"
        "  --backend cpu|vulkan      inference backend (default cpu)\n"
        "  --ctx N                   context length (default 32768)\n"
        "  --effort LEVEL            reasoning effort for Qwen3.8-Flash-Next: xhigh (default), medium, low\n"
        "  --no-think                turn thinking off\n"
        "  --hide-think              show a spinner instead of the reasoning (/think toggles)\n"
        "  --keep-think, --drop-think  keep / drop earlier turns' reasoning in the history (default: the model's template)\n"
        "\n"
        "Sampling\n"
        "  -n, --max-tokens N        max new tokens per reply (default 4096)\n"
        "  --temp T  --top-k K  --top-p P  --seed S  --greedy\n"
        "\n"
        "Tools and permissions\n"
        "  --permission-mode MODE    ask (default: changes and commands ask first), edits (file changes allowed,\n"
        "                            commands ask), all (nothing asks), read-only (changes and commands refused).\n"
        "                            Also accepted: default, acceptEdits, bypassPermissions, plan; and --permission\n"
        "  --dangerously-skip-permissions   same as --permission-mode all\n"
        "  --no-tools                no tools (files in the working folder, PowerShell commands, web search)\n"
        "\n"
        "Appearance\n"
        "  --theme NAME              dark, light, dark-colorblind, light-colorblind, ansi, mono (/theme in the chat)\n"
        "\n"
        "Memory\n"
        "  --vram-reserve GB         VRAM left free for other programs (default 5)\n"
        "  --ram-reserve GB          RAM left free; if pinning the experts would cut into it, the CPU runs uncached\n"
        "                            experts instead (slower) (default 16)\n"
        "  --expert-cache MB         fixed GPU expert cache size (default: all VRAM but the reserve)\n"
        "  --image-tokens N          largest image size in tokens, 32x32 pixels each (default 1024)\n"
        "  --no-vision               don't load the vision tower (frees ~1 GB of VRAM for the expert cache)\n"
        "  --threads N               CPU threads (default: all)\n"
        "\n"
        "Other\n"
        "  --tokenize TEXT           print token ids and exit\n"
        "  --eval IDS                comma-separated ids: run and print top-10 logits of the last position\n"
        "                            (with --image, the image token id marks where the image goes)\n"
        "  --tool NAME key=value...  run one tool directly\n"
        "  -v, --version             print the version\n"
        "  -h, --help                this help\n"
        "\n"
        "In the chat, /help lists the commands, / opens the command menu and /config the settings.\n");
}

// User settings that outlive a session (~/.qwen/settings.json): currently the theme.
fs::path settings_path() {
    const char* home = std::getenv("USERPROFILE");
    if (!home) home = std::getenv("HOME");
    return home ? fs::path(home) / ".qwen" / "settings.json" : fs::path();
}
std::string load_saved_theme() {
    try {
        std::ifstream f(settings_path(), std::ios::binary);
        if (!f) return "";
        const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const json::Value v = json::parse(text);
        if (const json::Value* t = v.find("theme")) return t->as_str();
    } catch (...) {
    }
    return "";
}
void save_theme(const std::string& name) {
    const fs::path p = settings_path();
    if (p.empty()) return;
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    std::ofstream(p, std::ios::binary) << "{\"theme\":\"" << name << "\"}\n";
}

// A few lines in the current theme, shown under the theme menu.
std::string theme_sample() {
    const term::Theme& t = term::theme();
    const std::string off = term::sgr("0");
    return "\n   " + term::sgr(t.accent) + "\xE2\x97\x8F" + off + " An answer with " + term::sgr("1") + "bold" + off + " text and " +
           term::sgr(t.code) + "inline code" + off + "\n     " + term::sgr(t.heading) + "A heading" + off + "   " + term::sgr(t.dim) +
           "faint text, like the reasoning" + off + "\n     " + term::sgr(t.fence) + "int main() { return 0; }" + off + "\n     " +
           term::sgr(t.ok) + "+ added line" + off + "   " + term::sgr(t.err) + "- removed line" + off + "   " + term::sgr(t.warn) +
           "needs approval" + off + "\n\n";
}

// Theme menu with a live preview; returns false when cancelled (the previous theme is then restored).
bool pick_theme(const std::string& title) {
    const std::string before = term::theme().name;
    std::vector<term::Choice> items;
    int cur = 0;
    for (size_t i = 0; i < term::themes().size(); i++) {
        items.push_back({term::themes()[i].name, term::themes()[i].desc});
        if (before == term::themes()[i].name) cur = (int)i;
    }
    const int idx = term::select(title, items, cur, [&](int i) { term::set_theme(items[i].label); }, [](int) { return theme_sample(); });
    term::set_theme(idx < 0 ? before : items[idx].label);
    if (idx >= 0) save_theme(items[idx].label);
    return idx >= 0;
}

// --model: a folder with a config.json, or part of the name of a model folder under models/ (next to the
// program's project, or in the working folder).
std::string resolve_model(const std::string& arg, const char* argv0) {
    std::error_code ec;
    if (fs::exists(fs::path(arg) / "config.json", ec)) return arg;
    std::string q = fs::path(arg).filename().string();
    for (char& c : q) c = (char)std::tolower((unsigned char)c);
    const fs::path exe = fs::absolute(fs::path(argv0), ec).parent_path();
    for (const fs::path& root : {fs::path("models"), exe / "models", exe / ".." / ".." / "models"}) {
        std::vector<std::string> hits;
        for (auto& e : fs::directory_iterator(root, ec)) {
            if (!fs::exists(e.path() / "config.json", ec)) continue;
            std::string n = e.path().filename().string();
            for (char& c : n) c = (char)std::tolower((unsigned char)c);
            if (n == q) return e.path().string();
            if (n.find(q) != std::string::npos) hits.push_back(e.path().string());
        }
        if (hits.size() > 1) {  // several quantizations of one model: the FP8 one
            std::vector<std::string> fp8;
            for (auto& h : hits)
                if (h.size() > 4 && (h.compare(h.size() - 4, 4, "-FP8") == 0 || h.compare(h.size() - 4, 4, "-fp8") == 0)) fp8.push_back(h);
            if (fp8.size() == 1) hits = fp8;
        }
        if (hits.size() == 1) return hits[0];
        if (hits.size() > 1) {
            std::string l;
            for (auto& h : hits) l += (l.empty() ? "" : ", ") + fs::path(h).filename().string();
            throw std::runtime_error("--model " + arg + " matches several models: " + l);
        }
    }
    return arg;  // let the loader report it
}

// A conversation: every token fed to the backend (re-fed to restore it) plus the visible turns.
// Image tokens appear in `tokens` as image-pad ids; `images` says which file produced them.
struct Session {
    struct Turn { std::string user, assistant; };
    struct Image { std::string path; size_t at = 0; int tokens = 0; int64_t max_pixels = 0; };
    std::vector<int> tokens;
    std::vector<Image> images;
    std::vector<Turn> turns;
    std::string system;
};

std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"') o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char)c;
    }
    return o + "\"";
}

void save_session(const fs::path& path, const std::string& model, const Session& s) {
    fs::create_directories(path.parent_path());
    std::string j = "{\"model\":" + json_str(model) + ",\"system\":" + json_str(s.system) + ",\"turns\":[";
    for (size_t i = 0; i < s.turns.size(); i++)
        j += (i ? "," : "") + std::string("{\"user\":") + json_str(s.turns[i].user) + ",\"assistant\":" + json_str(s.turns[i].assistant) + "}";
    j += "],\"images\":[";
    for (size_t i = 0; i < s.images.size(); i++)
        j += (i ? "," : "") + std::string("{\"path\":") + json_str(s.images[i].path) + ",\"at\":" + std::to_string(s.images[i].at) +
             ",\"tokens\":" + std::to_string(s.images[i].tokens) + ",\"max_pixels\":" + std::to_string(s.images[i].max_pixels) + "}";
    j += "],\"tokens\":[";
    for (size_t i = 0; i < s.tokens.size(); i++) j += (i ? "," : "") + std::to_string(s.tokens[i]);
    j += "]}\n";
    const fs::path tmp = path.string() + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary);
        if (!f.write(j.data(), (std::streamsize)j.size())) throw std::runtime_error("cannot write " + tmp.string());
    }
    fs::rename(tmp, path);
}

Session load_session(const fs::path& path, const std::string& model) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("no saved conversation " + path.stem().string());
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const json::Value v = json::parse(text);
    if (v["model"].as_str() != model)
        throw std::runtime_error("conversation belongs to " + v["model"].as_str() + ", not " + model);
    Session s;
    s.system = v["system"].as_str();
    for (auto& t : v["turns"].arr) s.turns.push_back({t["user"].as_str(), t["assistant"].as_str()});
    if (const json::Value* im = v.find("images"))
        for (auto& i : im->arr)
            s.images.push_back({i["path"].as_str(), (size_t)i["at"].as_int(), (int)i["tokens"].as_int(), i["max_pixels"].as_int()});
    for (auto& t : v["tokens"].arr) s.tokens.push_back((int)t.as_int());
    return s;
}

// Session names become file names.
std::string clean_name(std::string s) {
    for (char& c : s)
        if (!std::isalnum((unsigned char)c) && c != '-' && c != '_' && c != '.') c = '_';
    return s;
}

// Model folder name without its weight-format suffix.
std::string short_name_of(std::string n) {
    for (const char* suf : {"-FP8", "-fp8", "-BF16"})
        if (n.size() > std::strlen(suf) && n.compare(n.size() - std::strlen(suf), std::string::npos, suf) == 0)
            n.resize(n.size() - std::strlen(suf));
    return n;
}

std::string thousands(long long v) {
    std::string s = std::to_string(v);
    for (int i = (int)s.size() - 3; i > 0; i -= 3) s.insert((size_t)i, ",");
    return s;
}

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

// UTF-8 <-> filesystem paths (std::string paths would go through the ANSI code page on Windows).
fs::path u8path(const std::string& s) { return fs::path(std::u8string(s.begin(), s.end())); }
std::string u8str(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

// Removes the paths of existing image files (plain or "quoted", as drag and drop pastes them) from a
// message and returns them.
std::vector<std::string> take_image_paths(std::string& msg) {
    std::vector<std::string> found;
    std::string rest;
    size_t i = 0;
    while (i < msg.size()) {
        if (std::isspace((unsigned char)msg[i])) { rest += msg[i++]; continue; }
        size_t j;
        std::string word;
        if (msg[i] == '"' || msg[i] == '\'') {
            const size_t q = msg.find(msg[i], i + 1);
            j = q == std::string::npos ? msg.size() : q + 1;
            word = msg.substr(i + 1, (q == std::string::npos ? msg.size() : q) - i - 1);
        } else {
            j = msg.find_first_of(" \t\r\n", i);
            if (j == std::string::npos) j = msg.size();
            word = msg.substr(i, j - i);
        }
        std::error_code ec;
        if (is_image_path(word) && fs::is_regular_file(u8path(word), ec)) found.push_back(u8str(fs::absolute(u8path(word), ec)));
        else rest += msg.substr(i, j - i);
        i = j;
    }
    const size_t b = rest.find_first_not_of(" \t\r\n"), e = rest.find_last_not_of(" \t\r\n");
    msg = b == std::string::npos ? "" : rest.substr(b, e - b + 1);
    return found;
}

// At most n bytes of s (whole UTF-8 characters), with "..." when cut.
std::string utf8_cut(const std::string& s, size_t n) {
    if (s.size() <= n) return s;
    while (n > 0 && ((uint8_t)s[n] & 0xC0) == 0x80) n--;
    return s.substr(0, n) + "...";
}

// Project instructions: AGENTS.md (else CLAUDE.md) in the working folder and each parent up to the project
// root (the folder holding .git), outermost first.
std::vector<std::pair<fs::path, std::string>> project_instructions(const fs::path& start) {
    std::vector<std::pair<fs::path, std::string>> found;
    std::error_code ec;
    fs::path d = fs::absolute(start, ec);
    for (int depth = 0; depth < 32 && !d.empty(); depth++) {
        for (const char* n : {"AGENTS.md", "CLAUDE.md"}) {
            const fs::path f = d / n;
            if (!fs::is_regular_file(f, ec)) continue;
            std::ifstream in(f, std::ios::binary);
            std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (s.size() > 20000) s = s.substr(0, 20000) + "\n[truncated]";
            found.push_back({f, s});
            break;
        }
        if (fs::exists(d / ".git", ec)) break;
        const fs::path parent = d.parent_path();
        if (parent == d) break;
        d = parent;
    }
    std::reverse(found.begin(), found.end());
    return found;
}

// Length of the longest suffix of `s` that is a proper prefix of `tag` (bytes that may start the tag).
size_t partial_tag(const std::string& s, const std::string& tag) {
    for (size_t k = std::min(s.size(), tag.size() - 1); k > 0; k--)
        if (s.compare(s.size() - k, k, tag, 0, k) == 0) return k;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    // Once the Windows imaging DLLs are loaded the CRT's flush at exit can be skipped: flush on every return.
    struct FlushAtExit {
        ~FlushAtExit() {
            std::fflush(stdout);
            std::fflush(stderr);
        }
    } flush_at_exit;
    term::init();
    Options o;
    if (const std::string saved = load_saved_theme(); !saved.empty()) term::set_theme(saved);
    const bool first_run = load_saved_theme().empty();
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(1); }
            return argv[++i];
        };
        // a value that may be left out: the next argument unless it is another option
        auto has_value = [&] { return i + 1 < argc && argv[i + 1][0] != '-'; };
        if (a == "--model") {
            try {
                o.model_dir = resolve_model(next(), argv[0]);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "%s\n", e.what());
                return 1;
            }
        }
        else if (a == "--backend") o.backend = next();
        else if (a == "-p" || a == "--print" || a == "--prompt") {
            o.print = true;
            if (has_value()) o.prompt = next();
        }
        else if (a == "-f" || a == "--prompt-file") {
            std::ifstream f(next(), std::ios::binary);
            if (!f) { std::fprintf(stderr, "cannot read prompt file\n"); return 1; }
            o.prompt.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
            o.print = true;
        }
        else if (a == "--system" || a == "--system-prompt" || a == "--append-system-prompt") o.system = next();
        else if (a == "--max-tokens") o.max_new = std::stoi(next());
        else if (a == "--theme") {
            o.theme = next();
            if (!term::set_theme(o.theme)) {
                std::fprintf(stderr, "--theme must be one of:");
                for (auto& t : term::themes()) std::fprintf(stderr, " %s", t.name);
                std::fprintf(stderr, "\n");
                return 1;
            }
        }
        else if (a == "-v" || a == "--version") { std::printf("qwen %s\n", kVersion); return 0; }
        else if (a == "--dangerously-skip-permissions") o.permission = "all";
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
        else if (a == "-c" || a == "--continue") o.resume = "last";
        else if (a == "-r" || a == "--resume") {
            if (has_value()) o.resume = next();
            else o.resume_pick = true;
        }
        else if (a == "--image") o.images.push_back(next());
        else if (a == "--image-tokens") o.image_tokens = std::max(64, std::stoi(next()));
        else if (a == "--no-vision") o.no_vision = true;
        else if (a == "--no-tools") o.no_tools = true;
        else if (a == "--vram-reserve") o.vram_reserve_gb = std::max(0.0, std::stod(next()));
        else if (a == "--ram-reserve") o.ram_reserve_gb = std::max(0.0, std::stod(next()));
        else if (a == "--permission" || a == "--permission-mode") {
            o.permission = next();
            if (o.permission == "default") o.permission = "ask";
            else if (o.permission == "acceptEdits") o.permission = "edits";
            else if (o.permission == "bypassPermissions") o.permission = "all";
            else if (o.permission == "plan") o.permission = "read-only";
            if (o.permission != "ask" && o.permission != "edits" && o.permission != "all" && o.permission != "read-only") {
                std::fprintf(stderr, "--permission must be ask, edits, all or read-only\n");
                return 1;
            }
        }
        else if (a == "--tool") {  // run one tool directly: --tool NAME key=value ...
            ToolCall call;
            call.name = next();
            while (i + 1 < argc) {
                const std::string kv = argv[++i];
                const size_t eq = kv.find('=');
                call.args.push_back({kv.substr(0, eq), eq == std::string::npos ? "" : kv.substr(eq + 1)});
            }
            Tools t(fs::current_path(), false);
            const ToolResult r = t.run(call, [](const std::string&, const std::string& title, const std::string& preview, std::string&) {
                std::fprintf(stderr, "[%s]\n%s\nallow? [y/n] ", title.c_str(), preview.c_str());
                std::string ans;
                std::getline(std::cin, ans);
                return ans == "y";
            });
            std::printf("%s\n", r.text.c_str());
            std::fprintf(stderr, "[%s]\n", r.summary.c_str());
            return r.ok ? 0 : 1;
        }
        else if (a == "--tokenize") o.tokenize = next();
        else if (a == "--tokenize-file") o.tokenize_file = next();
        else if (a == "--eval") o.eval_ids = next();
        else if (a == "--eval-split") o.eval_split = std::stoi(next());
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a[0] != '-' && o.first_msg.empty()) o.first_msg = a;
        else { std::fprintf(stderr, "unknown option %s (qwen --help lists them)\n", a.c_str()); return 1; }
    }
    if (o.print && o.prompt.empty()) {  // qwen -p "text" also parses as a positional prompt; else read a pipe
        o.prompt = o.first_msg;
        if (o.prompt.empty() && !term::interactive())
            o.prompt.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
        if (o.prompt.find_first_not_of(" \t\r\n") == std::string::npos) {
            std::fprintf(stderr, "-p needs a prompt: qwen -p \"question\", or pipe one in\n");
            return 1;
        }
    }
    // first interactive start: choose the theme (saved in ~/.qwen/settings.json; /theme changes it)
    const bool special = !o.tokenize.empty() || !o.tokenize_file.empty() || !o.eval_ids.empty();
    if (first_run && o.theme.empty() && !o.print && !special && term::interactive()) {
        std::printf("\n");
        pick_theme("Choose the text style that looks best in your terminal (change it later with /theme)");
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

        // The model: config, mapped weights, backend. load_model() can run again to switch models (/model) or the
        // context size (/ctx); the backend goes first since it points into the weights, which point into `st`.
        Config cfg;
        SafeTensors st;
        ModelWeights w;
        std::unique_ptr<Backend> be;
        auto load_model = [&]() {
            const auto t0 = std::chrono::steady_clock::now();
            be.reset();
            w = ModelWeights{};
            st = SafeTensors();
            cfg = Config::load(o.model_dir);
            tok = Tokenizer();
            tok.load(o.model_dir);
            if (cfg.expert_q4 && o.backend == "cpu") throw std::runtime_error("4-bit (GPTQ) models need --backend vulkan");
            st = SafeTensors(o.model_dir);
            w.load(st, cfg);
            if (o.backend == "cpu") be = make_cpu_backend(cfg, w, o.max_ctx, o.threads);
#ifdef QW_VULKAN
            else if (o.backend == "vulkan" || o.backend == "vk") {
                VulkanOptions vo;
                vo.expert_cache_mb = o.expert_cache_mb;
                vo.stats_path = o.no_stats ? "" : o.model_dir + "/expert_usage.bin";
                vo.pin_experts = !o.no_pin;
                vo.hybrid = o.hybrid;
                vo.vision = !o.no_vision;
                vo.image_max_pixels = (int64_t)o.image_tokens * 1024;
                vo.vram_reserve_mb = (size_t)(o.vram_reserve_gb * 1024);
                vo.ram_reserve_mb = (size_t)(o.ram_reserve_gb * 1024);
                be = make_vulkan_backend(cfg, w, o.max_ctx, vo);
                // the GPU now has its own copies of the weights: the model files' pages (read once to make them)
                // no longer need to count against this process
                st.release_pages();
            }
#endif
            else throw std::runtime_error("unknown/unavailable backend: " + o.backend);
            const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "[%s backend, %d layers, %zu tensors, ready in %.1fs]\n", be->name(), cfg.n_layers,
                         st.count(), load_s);
        };
        load_model();

        std::vector<float> logits(cfg.vocab);
        const int64_t image_max_pixels = (int64_t)o.image_tokens * 1024;
        auto no_vision_reason = [&]() -> std::string {
            if (be->has_vision()) return "";
            if (!cfg.vis.depth || w.vis.blocks.empty()) return "this model has no supported vision tower";
            if (o.no_vision) return "the vision tower is off (--no-vision)";
            return "images need the vulkan backend";
        };
        auto load_input_image = [&](const std::string& path) {
            if (!be->has_vision()) throw std::runtime_error(no_vision_reason());
            ImageInput in = prepare_image(load_image(path), cfg.vis, w.vis, image_max_pixels);
            in.path = path;
            return in;
        };

        if (!o.eval_ids.empty()) {
            std::vector<int> ids;
            for (size_t p = 0; p < o.eval_ids.size();) {
                size_t q = o.eval_ids.find(',', p);
                if (q == std::string::npos) q = o.eval_ids.size();
                ids.push_back(std::stoi(o.eval_ids.substr(p, q - p)));
                p = q + 1;
            }
            if (!o.images.empty()) {
                // the image token marks the image; QW_VISION_DUMP=dir writes the patches and embeddings for tests
                const auto it = std::find(ids.begin(), ids.end(), cfg.vis.image_token);
                if (it == ids.end()) throw std::runtime_error("--eval with --image needs the image token id in the list");
                const ImageInput img = load_input_image(o.images[0]);
                const auto te = std::chrono::steady_clock::now();
                const std::vector<float> emb = be->encode_image(img);
                std::fprintf(stderr, "[image %dx%d -> %dx%d, %d patches, %d tokens, encoded in %.0f ms]\n", img.width, img.height,
                             img.rw, img.rh, img.gh * img.gw, img.tokens(),
                             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - te).count());
                if (const char* dump = std::getenv("QW_VISION_DUMP")) {
                    const fs::path d(dump);
                    fs::create_directories(d);
                    std::ofstream(d / "patches.f32", std::ios::binary).write((const char*)img.patches.data(), img.patches.size() * 4);
                    std::ofstream(d / "embeds.f32", std::ios::binary).write((const char*)emb.data(), emb.size() * 4);
                    std::ofstream(d / "grid.txt") << img.gh << " " << img.gw << " " << img.rh << " " << img.rw << "\n";
                }
                const int at = (int)(it - ids.begin());
                if (at) be->forward(ids.data(), at, logits.data());
                be->forward_image(emb.data(), img.lgh(), img.lgw(), logits.data());
                ids.erase(ids.begin(), it + 1);
            }
            // prefill all but the last `eval_split` tokens, then feed those one at a time (decode path)
            const int split = std::min<int>(o.eval_split, (int)ids.size() - 1);
            if ((int)ids.size() - split > 0) be->forward(ids.data(), (int)ids.size() - split, logits.data());
            for (int i = (int)ids.size() - split; i < (int)ids.size(); i++) be->forward(&ids[i], 1, logits.data());
            std::vector<int> idx(cfg.vocab);
            for (int i = 0; i < cfg.vocab; i++) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
            for (int i = 0; i < 10; i++) std::printf("%d %.4f\n", idx[i], logits[idx[i]]);
            if (const char* dump = std::getenv("QW_VISION_DUMP"))
                std::ofstream(fs::path(dump) / "logits.f32", std::ios::binary).write((const char*)logits.data(), logits.size() * 4);
            return 0;
        }

        Sampler sampler(o);
        auto is_eos = [&](int id) { return std::find(cfg.eos_ids.begin(), cfg.eos_ids.end(), id) != cfg.eos_ids.end(); };
        auto secs_since = [](std::chrono::steady_clock::time_point t) {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
        };
        // per-model values, refreshed after load_model()
        int im_end = 0;
        fs::path model_path, session_dir;
        std::string model_name;
        bool keep_think = false;
        auto refresh_model = [&] {
            im_end = tok.token_id("<|im_end|>");
            model_path = fs::absolute(u8path(o.model_dir)).lexically_normal();
            if (!model_path.has_filename()) model_path = model_path.parent_path();
            model_name = u8str(model_path.filename());
            session_dir = model_path / "sessions";
            keep_think = o.keep_think < 0 ? cfg.qwen4 : o.keep_think == 1;
            logits.assign(cfg.vocab, 0.0f);
        };
        refresh_model();
        const bool ui = term::color();
        // role colors of the current theme (apply_theme() refreshes them after /theme)
        const std::string off = term::sgr("0"), bold = term::sgr("1");
        std::string dim, red, cyan, green, yellow, accent_b;
        auto apply_theme = [&] {
            const term::Theme& t = term::theme();
            dim = term::sgr(t.dim);
            red = term::sgr(t.err);
            cyan = term::sgr(t.accent);
            green = term::sgr(t.ok);
            yellow = term::sgr(t.warn);
            accent_b = bold + cyan;
        };
        apply_theme();

        // The conversation is mirrored token by token as it is fed, so it can be saved and re-fed later.
        Session sess;
        sess.system = o.system;
        size_t saved_tokens = 0;
        auto feed = [&](const std::vector<int>& ids) {
            if (ids.empty()) return;
            be->forward(ids.data(), (int)ids.size(), logits.data());
            sess.tokens.insert(sess.tokens.end(), ids.begin(), ids.end());
        };
        // vision tower + image tokens, between <|vision_start|> and <|vision_end|>
        auto feed_image = [&](const ImageInput& img) {
            feed({cfg.vis.vision_start});
            const std::vector<float> emb = be->encode_image(img);
            be->forward_image(emb.data(), img.lgh(), img.lgw(), logits.data());
            sess.images.push_back({img.path, sess.tokens.size(), img.tokens(), image_max_pixels});
            sess.tokens.insert(sess.tokens.end(), (size_t)img.tokens(), cfg.vis.image_token);
            feed({cfg.vis.vision_end});
        };
        auto truncate_session = [&](size_t n) {
            sess.tokens.resize(n);
            while (!sess.images.empty() && sess.images.back().at >= n) sess.images.pop_back();
        };
        auto image_label = [&](const ImageInput& img) {
            return u8str(u8path(img.path).filename()) + " \xC2\xB7 " + std::to_string(img.width) + "\xC3\x97" +
                   std::to_string(img.height) + " \xE2\x86\x92 " + thousands(img.tokens()) + " tokens";
        };

        if (o.effort != "xhigh" && o.effort != "medium" && o.effort != "low")
            throw std::runtime_error("--effort must be xhigh, medium or low");
        // qwen4 template: with thinking on, a reasoning-effort instruction leads the system message
        auto effort_instr = [&]() -> std::string {
            if (!cfg.qwen4 || !o.think) return "";
            if (o.effort == "xhigh")
                return "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
                       "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
            if (o.effort == "low")
                return "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
                       "conclusion without unnecessary elaboration.";
            return "";
        };
        // Tools: files in the working folder, PowerShell commands, the web. Changes need the user's approval.
        Tools tools(fs::current_path(), be->has_vision());
        tools.set_cancel([] { return term::interrupted(); });
        bool tools_on = !o.no_tools;
        std::set<std::string> always_allow;  // tools the user allowed for the rest of the session
        bool abort_turn = false;              // Ctrl+C at an approval prompt
        const Tools::Approver approver = [&](const std::string& tool, const std::string& title, const std::string& preview,
                                             std::string& note) {
            if (abort_turn) return false;
            if (o.permission == "read-only") {
                note = "read-only mode is on: in this session you may not change files or run commands. Work with what you can "
                       "read, and tell the user what you would change.";
                std::printf("    %s%s: refused (read-only mode)%s\n", dim.c_str(), title.c_str(), off.c_str());
                return false;
            }
            const bool is_edit = tool == "write_file" || tool == "edit_file";
            const char* auto_why = o.permission == "all" ? "permission: all"
                                   : o.permission == "edits" && is_edit ? "permission: edits"
                                   : always_allow.count(tool) ? "allowed for this session"
                                                              : nullptr;
            std::printf("  %s\xE2\x8E\xBF%s  %s%s%s\n", dim.c_str(), off.c_str(), yellow.c_str(), title.c_str(), off.c_str());
            size_t a = 0;
            int shown_lines = 0;
            while (a <= preview.size() && shown_lines < 40) {
                const size_t e = preview.find('\n', a);
                const std::string l = preview.substr(a, e == std::string::npos ? std::string::npos : e - a);
                const std::string& col = l.rfind("+ ", 0) == 0 ? green : l.rfind("- ", 0) == 0 ? red : off;
                std::printf("    %s\xE2\x94\x82%s %s%s%s\n", dim.c_str(), off.c_str(), col.c_str(), l.c_str(), off.c_str());
                shown_lines++;
                if (e == std::string::npos) break;
                a = e + 1;
            }
            if (auto_why) {
                std::printf("    %s(approved: %s)%s\n", dim.c_str(), auto_why, off.c_str());
                return true;
            }
            std::string ans;
            if (term::interactive()) {  // a menu, as long as there is a terminal to show it on
                const int pick = term::select("Do you want to allow this?",
                                              {{"Yes", ""},
                                               {"Yes, and don't ask again for " + tool + " this session", ""},
                                               {"No", "the model is told you declined"},
                                               {"No, and tell the model what to do instead", ""}},
                                              -1);
                if (pick == 0) return true;
                if (pick == 1) {
                    always_allow.insert(tool);
                    return true;
                }
                if (pick == 2) return false;
                if (pick < 0) {  // Esc / Ctrl+C: stop the turn
                    abort_turn = true;
                    return false;
                }
                const term::Input r = term::read_message("    " + dim + "what should it do instead?" + off + " " + accent_b + "\xE2\x9D\xAF " + off, ans);
                if (r == term::Input::Cancel) abort_turn = true;
                else note = ans;
                return false;
            }
            const term::Input r = term::read_message("    " + bold + "allow?" + off + " " + dim + "[y]es / [n]o / [a]lways for " +
                                                         tool + " / or type what to do instead" + off + " " + bold + "\xE2\x80\xBA " + off,
                                                     ans);
            if (r == term::Input::Cancel) {
                abort_turn = true;
                return false;
            }
            const std::string t = [&] { std::string s = ans; for (char& ch : s) ch = (char)std::tolower((unsigned char)ch); return s; }();
            if (t == "y" || t == "yes") return true;
            if (t == "a" || t == "always") {
                always_allow.insert(tool);
                return true;
            }
            if (t != "n" && t != "no") note = ans;
            return false;
        };

        constexpr int kSummaryReserve = 2048;  // context kept free for writing a summary when compacting

        // One assistant message, streamed: reasoning faint or behind a spinner, the answer as markdown, tool-call
        // markup hidden. Plain (non-terminal) output prints the reasoning raw, tags included.
        struct Gen {
            std::string text;
            size_t answer_at = std::string::npos;  // where the answer starts in `text`
            int n = 0;
            bool stopped = false, full = false, shown_answer = false;
        };
        auto generate = [&]() -> Gen {
            Gen g;
            const std::string end_tag = "</think>", call_tag = "<tool_call>";
            const bool hide = o.hide_think;
            bool thinking = o.think;
            term::Markdown thought(ui, true), answer(ui);
            answer.prefix(cyan + "\xE2\x97\x8F" + off + " ", "  ");    // ● answer, later lines indented under it
            thought.prefix(dim + "\xE2\x9C\xBB" + off + " ", "  ");    // ✻ reasoning
            term::Spinner spin(ui && hide, "Thinking");
            if (thinking && !ui && !hide) std::printf("<think>\n");
            std::string& text = g.text;
            size_t shown = 0;     // bytes of `text` displayed or skipped so far
            bool lead_nl = true;  // skipping the newlines that open the answer
            int n_think = 0;
            auto show_answer = [&](bool final) {
                size_t end = text.size();
                if (tools_on) {
                    const size_t tc = text.find(call_tag, g.answer_at == std::string::npos ? 0 : g.answer_at);
                    end = tc != std::string::npos ? tc : end - partial_tag(text, call_tag);
                }
                if (!final || tools_on)  // trailing newlines wait for more text
                    while (end > shown && (text[end - 1] == '\n' || text[end - 1] == ' ')) end--;
                if (end <= shown) return;
                std::string s = text.substr(shown, end - shown);
                shown = end;
                if (lead_nl) {
                    const size_t b = s.find_first_not_of('\n');
                    if (b == std::string::npos) return;
                    s.erase(0, b);
                    lead_nl = false;
                }
                g.shown_answer = true;
                answer.put(s);
            };
            auto show_thought = [&](size_t end) {
                if (ui)
                    while (end > shown && text[end - 1] == '\n') end--;
                if (!hide) thought.put(text.substr(shown, end - shown));
                shown = end;
            };
            for (; g.n < o.max_new; g.n++) {
                if (term::interrupted()) { g.stopped = true; break; }
                const int next = sampler.sample(logits.data(), cfg.vocab);
                if (is_eos(next)) break;
                text += tok.decode(next);
                if (thinking) {
                    n_think++;
                    const size_t p = text.find(end_tag, shown);
                    if (p == std::string::npos) {
                        show_thought(text.size() - partial_tag(text, end_tag));
                        spin.update(n_think);
                    } else {
                        show_thought(p);
                        thinking = false;
                        shown = g.answer_at = p + end_tag.size();
                        if (hide) {
                            spin.clear();
                            if (ui) std::printf("%s\xE2\x9C\xBB Thought for %.0fs (%d tokens)%s\n\n", dim.c_str(), spin.seconds(), n_think, off.c_str());
                        } else {
                            thought.finish();
                            std::fputs(ui ? "\n\n" : end_tag.c_str(), stdout);
                        }
                    }
                } else if (g.answer_at == std::string::npos) {
                    g.answer_at = 0;
                }
                if (!thinking) show_answer(false);
                // stop short of the end so a summary still fits (the turn then compacts and continues)
                if (be->pos() + 2 >= be->max_ctx() - kSummaryReserve) { g.full = true; break; }
                feed({next});
            }
            if (thinking) {  // the reasoning did not finish
                spin.clear();
                show_thought(text.size());
                thought.finish();
                if (!g.stopped && !g.full)
                    std::printf("\n%s[reasoning did not finish within %d tokens; raise -n]%s", dim.c_str(), o.max_new, off.c_str());
            } else {
                show_answer(true);
            }
            answer.finish();
            if (g.stopped) std::printf("\n%s[stopped]%s", dim.c_str(), off.c_str());
            if (g.shown_answer || g.stopped || g.full || thinking) std::printf("\n");
            std::fflush(stdout);
            return g;
        };

        // A tool response that fits in the context. One too big for the space left, but not for a fresh context, is
        // replaced by a note and `defer` is set: the conversation is then summarized and the model repeats the call.
        // Only a response too big for any context is cut.
        auto fit_response = [&](std::string text, bool& defer) {
            const int room = be->max_ctx() - be->pos() - kSummaryReserve - 512;  // keeps room for a summary if the context must be compacted
            const std::vector<int> ids = tok.encode(text);
            if ((int)ids.size() <= room) return text;
            if ((int)ids.size() <= be->max_ctx() / 3) {
                defer = true;
                return "[Result not shown: it does not fit in the context that is left (" + thousands((long long)ids.size()) +
                       " tokens). The conversation is about to be summarized; afterwards, call this tool again with the same "
                       "arguments to see the result.]";
            }
            if (room < 64) return std::string("Error: the context is full; finish without more tool calls.");
            size_t keep = text.size() * (size_t)room / ids.size() * 9 / 10;
            while (keep > 0 && ((uint8_t)text[keep] & 0xC0) == 0x80) keep--;
            return text.substr(0, keep) + "\n[output truncated: the context is nearly full]";
        };

        std::string carry_summary;  // summary of a compacted conversation, for the next system message
        std::string undo_note;      // tells the model on the next message that the user undid its file changes

        // The system message: tool instructions, the reasoning-effort line (qwen4), environment, project
        // instructions (AGENTS.md), the user's system prompt and, after a compaction, the summary.
        auto system_block = [&]() {
            std::string s;
            auto add = [&](const std::string& part) {
                if (!part.empty()) s += (s.empty() ? "" : "\n\n") + part;
            };
            add(effort_instr());
            if (tools_on) add(tools.context_note());
            const auto proj = project_instructions(fs::current_path());
            if (!proj.empty()) {
                std::string p = "# Project instructions\nThe user's project provides these instructions (AGENTS.md); follow them.";
                for (auto& [path, text] : proj) p += "\n\n## " + u8str(path) + "\n" + text;
                add(p);
            }
            add(sess.system);
            if (!carry_summary.empty())
                add("# Earlier in this conversation\nThe conversation so far was condensed to free up context. Your summary of it:\n\n" +
                    carry_summary);
            return tools_on ? tools.system_prompt() + (s.empty() ? "" : "\n\n" + s) : s;
        };

        // One chat turn: the user message, then assistant messages and tool rounds until an answer without tool
        // calls. A plain answer is rolled back to the snapshot after "<|im_start|>assistant\n" and re-fed without
        // its reasoning (as the chat template renders history); turns that used tools stay as generated, which
        // the template's preserve_thinking form allows, so long tool outputs are not processed twice.
        // Returns true when the turn stopped mid-task because the context is nearly full (compact, then continue).
        auto run_turn = [&](const std::string& msg, const std::vector<ImageInput>& images) -> bool {
            const bool first = sess.tokens.empty();
            const std::string sys_block = first ? system_block() : "";
            std::string head;
            if (first && !sys_block.empty()) head += "<|im_start|>system\n" + sys_block + "<|im_end|>\n";
            if (!first) head += "\n";  // newline after the previous <|im_end|>
            head += "<|im_start|>user\n";
            const std::string umsg = undo_note.empty() ? msg : undo_note + "\n\n" + msg;
            // images lead the user message, as the chat template renders [image..., text]
            const std::vector<int> pre = tok.encode(head), post = tok.encode(umsg + "<|im_end|>\n<|im_start|>assistant\n");
            const std::vector<int> gen_prompt = tok.encode(o.think ? "<think>\n" : "<think>\n\n</think>\n\n");
            size_t n_prefill = pre.size() + post.size() + gen_prompt.size();
            for (const ImageInput& im : images) n_prefill += (size_t)im.tokens() + 2;
            if (be->pos() + (int)n_prefill >= be->max_ctx())
                throw std::runtime_error("context full; /new starts a new conversation");

            term::clear_interrupt();
            abort_turn = false;
            undo_note.clear();
            tools.begin_turn();
            struct EndTurn {  // file changes of this turn become one /undo step, however the turn ends
                Tools& t;
                ~EndTurn() { t.end_turn(); }
            } end_turn{tools};
            const auto ts = std::chrono::steady_clock::now();
            feed(pre);
            for (const ImageInput& im : images) feed_image(im);
            feed(post);
            if (!keep_think) {
                be->save_state();
                saved_tokens = sess.tokens.size();
            }
            feed(gen_prompt);
            const double prefill_s = secs_since(ts);

            const auto tg = std::chrono::steady_clock::now();
            int n_gen = 0, n_calls = 0;
            bool used_tools = false, open = true;  // open: the last assistant message still needs its <|im_end|>
            bool needs_compaction = false;
            std::string content;
            constexpr int kMaxRounds = 20;          // tool rounds per turn; the last one asks for the answer
            std::map<std::string, std::string> seen;  // read-only calls of this turn -> result text
            for (int step = 0;; step++) {
                const Gen g = generate();
                n_gen += g.n;
                const std::string ans = g.answer_at != std::string::npos ? g.text.substr(g.answer_at) : "";
                content = ans.substr(0, tools_on ? ans.find("<tool_call>") : std::string::npos);
                if (g.full) {  // out of context mid-message: close it, then summarize and continue
                    feed({im_end});
                    open = false;
                    needs_compaction = true;
                    break;
                }
                const std::vector<ToolCall> calls = tools_on && !g.stopped ? parse_tool_calls(ans) : std::vector<ToolCall>{};
                if (calls.empty()) break;
                if (step >= kMaxRounds) {  // it was told to answer and still called tools
                    std::printf("%s[stopping: %d tool rounds in one turn]%s\n", dim.c_str(), kMaxRounds, off.c_str());
                    break;
                }
                const bool last_round = step == kMaxRounds - 1;
                bool defer = false;
                used_tools = true;
                feed({im_end});
                feed(tok.encode("\n<|im_start|>user"));
                if (g.shown_answer) std::printf("\n");
                for (const ToolCall& call : calls) {
                    n_calls++;
                    std::printf("%s\xE2\x97\x8F%s %s%s%s\n", green.c_str(), off.c_str(), bold.c_str(), tools.describe(call).c_str(), off.c_str());
                    ToolResult r;
                    std::string key = call.name;
                    for (auto& [k, v] : call.args) key += "\x1f" + k + "=" + v;
                    const bool read_only = call.name == "list_dir" || call.name == "read_file" || call.name == "search_text" ||
                                           call.name == "web_search" || call.name == "fetch_url";
                    if (abort_turn || term::interrupted()) {
                        r = {"Not run: the user stopped the turn.", "", "not run", "", false};
                    } else if (auto it = seen.find(key); read_only && it != seen.end()) {
                        r = {"(Same call as earlier in this turn; the result has not changed. Use a different approach.)\n" + it->second, "",
                             "repeated call, earlier result reused", "", true};
                    } else {
                        r = tools.run(call, approver);
                        if (read_only && r.image.empty()) seen[key] = r.text;
                        if (!read_only) seen.clear();  // files or the world may have changed
                    }
                    if (last_round)
                        r.text += "\n\n[This was the last tool round of this turn: do not call any more tools. Answer the user now "
                                  "with what you have, and say what is still missing.]";
                    if (!r.detail.empty()) {  // command output: its last lines
                        std::vector<std::string> lines;
                        size_t a = 0;
                        while (a < r.detail.size()) {
                            const size_t e = r.detail.find('\n', a);
                            std::string l = r.detail.substr(a, e == std::string::npos ? std::string::npos : e - a);
                            if (!l.empty() && l.back() == '\r') l.pop_back();
                            lines.push_back(l);
                            if (e == std::string::npos) break;
                            a = e + 1;
                        }
                        const size_t from = lines.size() > 12 ? lines.size() - 12 : 0;
                        if (from) std::printf("    %s\xE2\x94\x82 ... %zu earlier lines%s\n", dim.c_str(), from, off.c_str());
                        for (size_t i = from; i < lines.size(); i++)
                            std::printf("    %s\xE2\x94\x82 %s%s\n", dim.c_str(), lines[i].substr(0, 200).c_str(), off.c_str());
                    }
                    std::printf("  %s\xE2\x8E\xBF  %s%s%s\n", dim.c_str(), r.ok ? "" : red.c_str(), r.summary.c_str(), off.c_str());
                    std::printf("\n");
                    std::fflush(stdout);
                    feed(tok.encode("\n<tool_response>\n"));
                    if (!r.image.empty()) {
                        try {
                            feed_image(load_input_image(r.image));
                        } catch (const std::exception& e) {
                            r.text += std::string(" (the image could not be loaded: ") + e.what() + ")";
                        }
                    }
                    feed(tok.encode(fit_response(r.text, defer) + "\n</tool_response>"));
                }
                feed(tok.encode("<|im_end|>"));
                open = false;
                if (defer) {  // a result waits for a fresh context
                    needs_compaction = true;
                    break;
                }
                if (abort_turn || term::interrupted()) {
                    std::printf("%s[stopped]%s\n", dim.c_str(), off.c_str());
                    break;
                }
                if (be->pos() >= be->max_ctx() * 4 / 5) {  // mid-task: summarize and carry on in a fresh context
                    needs_compaction = true;
                    break;
                }
                feed(tok.encode("\n<|im_start|>assistant\n"));
                feed(gen_prompt);
                open = true;
            }
            const double gen_s = secs_since(tg);

            {
                const size_t b = content.find_first_not_of('\n');
                content = b == std::string::npos ? "" : content.substr(b);
                while (!content.empty() && std::isspace((unsigned char)content.back())) content.pop_back();
            }
            if (open) {
                if (keep_think || used_tools) {
                    feed({im_end});  // close the message: a sampled <|im_end|> ends generation without being fed
                } else {
                    be->restore_state();
                    truncate_session(saved_tokens);
                    feed(tok.encode(content + "<|im_end|>"));
                }
            }
            std::string shown_msg;
            for (const ImageInput& im : images) shown_msg += "[image: " + u8str(u8path(im.path).filename()) + "] ";
            sess.turns.push_back({shown_msg + msg, content});
            const double tps = n_gen / std::max(gen_s, 1e-9);
            const std::string calls_s = n_calls ? " \xC2\xB7 " + std::to_string(n_calls) + (n_calls == 1 ? " tool call" : " tool calls") : "";
            if (ui)
                std::fprintf(stderr, "\n%s  %d tokens \xC2\xB7 %.1f tok/s%s \xC2\xB7 prompt %zu tokens at %.0f tok/s \xC2\xB7 context %s / %s%s\n",
                             dim.c_str(), n_gen, tps, calls_s.c_str(), n_prefill, n_prefill / prefill_s, thousands(be->pos()).c_str(),
                             thousands(be->max_ctx()).c_str(), off.c_str());
            else
                std::fprintf(stderr, "[prefill %zu tok %.1f tok/s | gen %d tok %.2f tok/s | ctx %d]\n", n_prefill,
                             n_prefill / prefill_s, n_gen, tps, be->pos());
            return needs_compaction;
        };

        // Compaction: in the current context the model writes a summary of the conversation; the context is then
        // cleared and the summary goes into the next system message. False when there was nothing to compact.
        auto compact = [&](bool mid_task) -> bool {
            if (sess.tokens.empty()) return false;
            const int before = be->pos();
            const std::string req =
                std::string("Pause here: this conversation's context is about to be cleared, and a summary you write now is all that "
                            "will remain of it (together with the system prompt). Write that summary for yourself. Include: the user's "
                            "goals and requests, with their exact requirements; important facts, findings and decisions, with exact file "
                            "paths, commands, URLs, names and numbers; the files you created or changed, and how; ") +
                (mid_task ? "the task in progress: what is done, what remains, and the exact next step (if the last tool result was "
                            "not shown because the context was full, the next step is to repeat that exact call). "
                          : "the current state of the work and any open questions. ") +
                "Use the user's language. Be complete but concise (under 700 words). Do not call tools.";
            const std::vector<int> ids = tok.encode("\n<|im_start|>user\n" + req + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
            const int room = be->max_ctx() - be->pos() - (int)ids.size();
            std::string summary;
            if (room > 300) {
                feed(ids);
                term::Spinner spin(ui, "summarizing the conversation");
                for (int n = 0; n < std::min(1500, room - 8); n++) {
                    const int next = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
                    if (is_eos(next)) break;
                    summary += tok.decode(next);
                    spin.update(n + 1);
                    feed({next});
                }
                spin.clear();
                summary = summary.substr(0, summary.find("<tool_call>"));
            }
            const size_t b = summary.find_first_not_of(" \n"), e = summary.find_last_not_of(" \n");
            summary = b == std::string::npos ? "" : summary.substr(b, e - b + 1);
            if (summary.empty()) {  // no room left to write one: keep the visible turns instead
                for (const auto& t : sess.turns) summary += "User: " + t.user + "\nAssistant: " + t.assistant + "\n\n";
                if (summary.size() > 8000) summary = "..." + summary.substr(summary.size() - 8000);
            }
            be->reset();
            sess.tokens.clear();
            sess.images.clear();
            carry_summary = summary;
            const std::vector<int> st = tok.encode(summary);
            std::printf("%s\xE2\x86\xBB context summarized: %s tokens \xE2\x86\x92 a %s-token summary; continuing in a fresh context%s\n",
                        dim.c_str(), thousands(before).c_str(), thousands((long long)st.size()).c_str(), off.c_str());
            return true;
        };

        // A user turn, with automatic compaction: before it when the context is mostly used, and in the middle of
        // a long tool task (then the task continues in the fresh context).
        auto do_turn = [&](const std::string& msg, const std::vector<ImageInput>& images) {
            if (be->pos() >= be->max_ctx() * 7 / 10) compact(false);
            bool more = run_turn(msg, images);
            for (int k = 0; more && k < 5; k++) {
                compact(true);
                std::printf("%s\xE2\x86\xBB continuing the task%s\n", dim.c_str(), off.c_str());
                more = run_turn("Continue from where you stopped (the task, or your answer); your summary in the system prompt says "
                                "what is done and what remains.", {});
            }
        };

        if (!o.prompt.empty()) {
            std::vector<ImageInput> imgs;
            for (const std::string& p : o.images) imgs.push_back(load_input_image(p));
            term::catch_ctrl_c(true);
            do_turn(o.prompt, imgs);
            return 0;
        }

        // ---- interactive chat ----------------------------------------------------------------------
        auto new_conversation = [&] {
            be->reset();
            sess = Session{};
            sess.system = o.system;
            carry_summary.clear();
            undo_note.clear();
        };
        auto autosave = [&] {
            try {
                save_session(session_dir / "last.json", model_name, sess);
            } catch (const std::exception& e) {
                std::printf("%scould not save the conversation: %s%s\n", red.c_str(), e.what(), off.c_str());
            }
        };
        auto session_names = [&] {  // saved conversations, newest first
            std::vector<std::pair<fs::file_time_type, std::string>> v;
            std::error_code ec;
            for (auto& e : fs::directory_iterator(session_dir, ec))
                if (e.path().extension() == ".json") v.push_back({e.last_write_time(), e.path().stem().string()});
            std::sort(v.rbegin(), v.rend());
            std::vector<std::string> names;
            for (auto& [t, n] : v) names.push_back(n);
            return names;
        };
        auto list_sessions = [&] {
            std::vector<std::pair<fs::file_time_type, std::string>> v;
            std::error_code ec;
            for (auto& e : fs::directory_iterator(session_dir, ec))
                if (e.path().extension() == ".json") v.push_back({e.last_write_time(), e.path().stem().string()});
            if (v.empty()) {
                std::printf("%sno saved conversations yet (/save NAME)%s\n", dim.c_str(), off.c_str());
                return;
            }
            std::sort(v.rbegin(), v.rend());
            for (auto& [t, name] : v) {
                const auto sys = std::chrono::system_clock::now() +
                                 std::chrono::duration_cast<std::chrono::system_clock::duration>(t - fs::file_time_type::clock::now());
                const std::time_t tt = std::chrono::system_clock::to_time_t(sys);
                char when[32];
                std::strftime(when, sizeof when, "%Y-%m-%d %H:%M", std::localtime(&tt));
                std::printf("  %-28s %s%s%s\n", name.c_str(), dim.c_str(), when, off.c_str());
            }
        };
        auto resume = [&](const std::string& name) {
            Session s = load_session(session_dir / (clean_name(name) + ".json"), model_name);
            if ((int)s.tokens.size() + 2 >= be->max_ctx()) throw std::runtime_error("conversation is longer than --ctx");
            new_conversation();
            sess.system = s.system;
            const auto t = std::chrono::steady_clock::now();
            if (!s.tokens.empty()) {
                std::printf("%sloading %s tokens...%s", dim.c_str(), thousands((long long)s.tokens.size()).c_str(), off.c_str());
                std::fflush(stdout);
                // text spans are re-fed as tokens, images re-encoded from their files
                size_t cur = 0;
                for (const Session::Image& im : s.images) {
                    if (im.at < cur || im.at + im.tokens > s.tokens.size()) throw std::runtime_error("corrupt conversation file");
                    feed(std::vector<int>(s.tokens.begin() + cur, s.tokens.begin() + im.at - 1));  // up to <|vision_start|>
                    if (!be->has_vision()) throw std::runtime_error("the conversation has images: " + no_vision_reason());
                    ImageInput in = prepare_image(load_image(im.path), cfg.vis, w.vis, im.max_pixels);
                    in.path = im.path;
                    if (in.tokens() != im.tokens) throw std::runtime_error("image changed since it was attached: " + im.path);
                    feed_image(in);
                    cur = im.at + im.tokens + 1;  // past <|vision_end|>
                }
                feed(std::vector<int>(s.tokens.begin() + cur, s.tokens.end()));
                std::printf(ui ? "\r\x1b[K" : "\n");
            }
            sess.turns = s.turns;
            std::printf("%sresumed '%s': %zu turns, %s tokens (%.1fs)%s\n", dim.c_str(), name.c_str(), sess.turns.size(),
                        thousands((long long)sess.tokens.size()).c_str(), secs_since(t), off.c_str());
            if (!sess.turns.empty()) {  // remind of where the conversation left off
                const auto& last = sess.turns.back();
                std::string u = last.user.size() > 300 ? last.user.substr(0, 300) + "..." : last.user;
                std::printf("\n%s\xE2\x9D\xAF %s%s\n\n", accent_b.c_str(), off.c_str(), u.c_str());
                term::Markdown md(ui);
                md.put(last.assistant.size() > 1500 ? last.assistant.substr(0, 1500) + "\n..." : last.assistant);
                md.finish();
                std::printf("\n");
            }
        };

        auto info = [&] {
            std::error_code ec;
            const fs::path cwd = fs::current_path(ec);
            std::printf("  %smodel%s    %s%s%s %s(%s \xC2\xB7 context %s \xC2\xB7 reasoning %s)%s\n", dim.c_str(), off.c_str(),
                        bold.c_str(), model_name.c_str(), off.c_str(), dim.c_str(), be->name(),
                        thousands(be->max_ctx()).c_str(), !o.think ? "off" : o.hide_think ? "on, hidden" : "on", off.c_str());
            std::printf("  %sweights%s  %s%s%s\n", dim.c_str(), off.c_str(), cyan.c_str(), u8str(model_path).c_str(), off.c_str());
            std::printf("  %sfolder%s   %s%s%s\n", dim.c_str(), off.c_str(), cyan.c_str(), u8str(cwd).c_str(), off.c_str());
            std::printf("  %ssessions%s %s%s%s\n", dim.c_str(), off.c_str(), cyan.c_str(), u8str(session_dir).c_str(), off.c_str());
            if (be->has_vision())
                std::printf("  %svision%s   on %s(up to %s tokens per image; /image FILE or drop a file into the window)%s\n",
                            dim.c_str(), off.c_str(), dim.c_str(), thousands(o.image_tokens).c_str(), off.c_str());
            else
                std::printf("  %svision%s   off %s(%s)%s\n", dim.c_str(), off.c_str(), dim.c_str(), no_vision_reason().c_str(), off.c_str());
            std::printf("  %stools%s    %s %s(%s)%s\n", dim.c_str(), off.c_str(), tools_on ? "on" : "off", dim.c_str(),
                        tools_on ? ("files in the folder above, commands, web; permission: " + o.permission + " (/permission)").c_str()
                                 : "/tools on enables them",
                        off.c_str());
            const auto proj = project_instructions(cwd);
            std::string ps;
            for (auto& [p, t] : proj) ps += (ps.empty() ? "" : ", ") + u8str(p);
            std::printf("  %sproject%s  %s\n", dim.c_str(), off.c_str(),
                        ps.empty() ? (dim + "no AGENTS.md (/init writes one)" + off).c_str() : ps.c_str());
            std::printf("  %scontext%s  %s / %s tokens %s(summarized automatically when it fills up; /compact)%s\n", dim.c_str(), off.c_str(),
                        thousands(be->pos()).c_str(), thousands(be->max_ctx()).c_str(), dim.c_str(), off.c_str());
            const MemoryStats ms = memory_stats();
            char mb[200];
            std::snprintf(mb, sizeof mb, "RAM: qwen uses %.1f GB \xC2\xB7 %.1f of %.0f GB free for other programs (reserve %.0f GB)",
                          ms.working_set, ms.available, ms.total, o.ram_reserve_gb);
            std::printf("  %smemory%s   %s\n", dim.c_str(), off.c_str(), mb);
            if (!be->memory_report().empty())
                std::printf("           %s %s(reserve %.1f GB)%s\n", be->memory_report().c_str(), dim.c_str(), o.vram_reserve_gb, off.c_str());
        };

        // Welcome box: what is running and where (/info has the details).
        auto banner = [&] {
            if (!ui) {
                info();
                return;
            }
            std::error_code ec;
            const int inner = std::max(30, std::min(term::columns() - 6, 76));  // columns between the borders
            std::vector<std::pair<std::string, std::string>> rows = {  // (plain text, styled text)
                {"\xE2\x9C\xBB Welcome to qwen", cyan + "\xE2\x9C\xBB" + off + " Welcome to " + bold + "qwen" + off},
                {"", ""},
            };
            auto row = [&](const std::string& key, const std::string& value, const std::string& extra) {
                const std::string v = term::fit(value, inner - 12 - term::width(extra));
                rows.push_back({"  " + key + v + extra, "  " + dim + key + off + v + dim + extra + off});
            };
            row("model   ", short_name_of(model_name), std::string(" \xC2\xB7 ") + be->name() + " \xC2\xB7 context " + thousands(be->max_ctx()));
            row("folder  ", u8str(fs::current_path(ec)), "");
            row("tools   ", tools_on ? "on \xC2\xB7 permission " + o.permission : "off", "");
            std::string bar;
            for (int i = 0; i < inner + 2; i++) bar += "\xE2\x94\x80";
            std::printf("%s\xE2\x95\xAD%s\xE2\x95\xAE%s\n", cyan.c_str(), bar.c_str(), off.c_str());
            for (auto& [plain, styled] : rows)
                std::printf("%s\xE2\x94\x82%s %s%*s %s\xE2\x94\x82%s\n", cyan.c_str(), off.c_str(), styled.c_str(),
                            std::max(0, inner - term::width(plain)), "", cyan.c_str(), off.c_str());
            std::printf("%s\xE2\x95\xB0%s\xE2\x95\xAF%s\n", cyan.c_str(), bar.c_str(), off.c_str());
            std::printf("%s  /help for commands \xC2\xB7 / for the command menu \xC2\xB7 /config for settings \xC2\xB7 Ctrl+C stops a reply%s\n",
                        dim.c_str(), off.c_str());
        };

        term::catch_ctrl_c(true);
        std::printf("\n");
        banner();
        if (o.resume_pick) {  // --resume without a name
            const std::vector<std::string> names = session_names();
            std::vector<term::Choice> items;
            for (auto& n : names) items.push_back({n, ""});
            const int idx = term::select("Resume which conversation?", items, -1);
            if (idx >= 0) o.resume = names[idx];
            else if (names.empty()) std::printf("%sno saved conversations yet%s\n", dim.c_str(), off.c_str());
        }
        if (!o.resume.empty()) {
            try {
                resume(o.resume);
            } catch (const std::exception& e) {
                std::printf("%s%s%s\n", red.c_str(), e.what(), off.c_str());
            }
        }
        std::vector<ImageInput> pending;  // images attached to the next message
        auto attach = [&](const std::string& path) {
            pending.push_back(load_input_image(path));
            std::printf("%sattached %s%s\n", dim.c_str(), image_label(pending.back()).c_str(), off.c_str());
        };

        // ---- models and settings ------------------------------------------------------------------------
        auto short_name = [](const std::string& n) { return short_name_of(n); };
        // Model folders next to the current one: (name, "" if usable, else why not)
        auto available_models = [&]() {
            std::vector<std::pair<std::string, std::string>> v;
            std::error_code ec;
            for (auto& e : fs::directory_iterator(model_path.parent_path(), ec)) {
                if (!e.is_directory(ec) || !fs::exists(e.path() / "config.json", ec)) continue;
                std::string why;
                try {
                    if (!Config::load(u8str(e.path())).n_experts) why = "not supported by this engine";
                } catch (...) {
                    why = "not supported by this engine";
                }
                if (why.empty()) {  // shards "...-00001-of-00039.safetensors": all present?
                    int have = 0, want = 0;
                    for (auto& f : fs::directory_iterator(e.path(), ec)) {
                        const std::string fn = u8str(f.path().filename());
                        const size_t of = fn.find("-of-");
                        if (f.path().extension() != ".safetensors") continue;
                        have++;
                        if (of != std::string::npos) want = std::max(want, std::atoi(fn.c_str() + of + 4));
                    }
                    if (!have || have < want) why = "not fully downloaded (" + std::to_string(have) + "/" + std::to_string(want) + " files)";
                }
                v.push_back({u8str(e.path().filename()), why});
            }
            std::sort(v.begin(), v.end());
            return v;
        };
        auto default_backend = [](const std::string& name) {
            std::string l = name;
            for (char& ch : l) ch = (char)std::tolower((unsigned char)ch);
            return l.find("flash") != std::string::npos ? std::string("cpu") : std::string("vulkan");
        };
        // Loads another model or context size. The conversation is carried over as a summary written by the current
        // model; if the new model fails to load, the previous one is loaded again.
        auto switch_model = [&](const std::string& dir, const std::string& backend, int ctx) {
            const std::string old_dir = o.model_dir, old_backend = o.backend;
            const int old_ctx = o.max_ctx;
            if (!sess.tokens.empty()) {
                autosave();  // the full conversation stays resumable with the old model (/model back, then /load last)
                compact(false);
            }
            pending.clear();
            o.model_dir = dir;
            o.backend = backend;
            o.max_ctx = ctx;
            std::printf("%sloading %s (%s, context %s)...%s\n", dim.c_str(), u8str(u8path(dir).filename()).c_str(), backend.c_str(),
                        thousands(ctx).c_str(), off.c_str());
            std::fflush(stdout);
            try {
                load_model();
            } catch (const std::exception& e) {
                std::printf("%scould not load it: %s; loading %s again%s\n", red.c_str(), e.what(),
                            u8str(u8path(old_dir).filename()).c_str(), off.c_str());
                o.model_dir = old_dir;
                o.backend = old_backend;
                o.max_ctx = old_ctx;
                load_model();
            }
            refresh_model();
            tools.set_images(be->has_vision());
            sess.tokens.clear();
            sess.images.clear();
            std::printf("%snow using %s%s%s\n", dim.c_str(), bold.c_str(), short_name(model_name).c_str(), off.c_str());
        };
        auto set_title = [&] {
            if (!ui) return;
            std::error_code ec;
            std::printf("\x1b]0;qwen \xC2\xB7 %s \xC2\xB7 %s\x07", short_name(model_name).c_str(), u8str(fs::current_path(ec).filename()).c_str());
        };
        // Status line shown above each prompt: model, backend, context, permission, reasoning.
        auto status_line = [&]() {
            const std::string sep = dim + " \xC2\xB7 " + off;
            std::string s = dim + "\xE2\x94\x80 " + off + accent_b + short_name(model_name) + off + sep + dim + be->name() + off + sep +
                            dim + "context " + thousands(be->pos()) + " / " + thousands(be->max_ctx()) + off;
            if (tools_on) s += sep + dim + "permission " + off + (o.permission == "ask" ? dim : yellow) + o.permission + off;
            else s += sep + dim + "tools off" + off;
            s += sep + dim + "reasoning " + (!o.think ? "off" : o.hide_think ? "hidden" : "shown") + off;
            return s;
        };
        auto print_settings = [&] {
            auto row = [&](const char* k, const std::string& v, const char* how) {
                std::printf("  %-14s %-28s %s%s%s\n", k, v.c_str(), dim.c_str(), how, off.c_str());
            };
            row("model", short_name(model_name), "/model NAME");
            row("backend", be->name(), "/model NAME cpu|vulkan");
            row("context", thousands(be->max_ctx()) + " tokens", "/ctx N");
            row("permission", tools_on ? o.permission : "(tools off)", "/permission ask|edits|all|read-only");
            row("tools", tools_on ? "on" : "off", "/tools on|off");
            row("reasoning", !o.think ? "off" : o.hide_think ? "on, hidden" : "on, shown", "/reasoning on|off, /think");
            row("temperature", o.greedy ? "greedy" : [&] { char b[16]; std::snprintf(b, sizeof b, "%.2f", o.temperature); return std::string(b); }(),
                "/set temp T, /set greedy on|off");
            row("top-p / top-k", [&] { char b[32]; std::snprintf(b, sizeof b, "%.2f / %d", o.top_p, o.top_k); return std::string(b); }(),
                "/set top-p P, /set top-k K");
            row("max reply", thousands(o.max_new) + " tokens", "/set max-tokens N");
            row("VRAM reserve", [&] { char b[16]; std::snprintf(b, sizeof b, "%.1f GB", o.vram_reserve_gb); return std::string(b); }(),
                "/set vram-reserve GB (reloads the model)");
            row("RAM reserve", [&] { char b[16]; std::snprintf(b, sizeof b, "%.0f GB", o.ram_reserve_gb); return std::string(b); }(),
                "/set ram-reserve GB (reloads the model)");
            row("system prompt", sess.system.empty() ? "(none)" : utf8_cut(sess.system, 28), "/system TEXT");
        };
        set_title();
        // Runs one slash command; false = quit. (/config and the command menu call it too.)
        bool in_config = false;  // inside the /config menu: commands skip their closing printouts
        std::function<bool(std::string)> command = [&](std::string line) -> bool {
            const size_t sp = line.find_first_of(" \t");
            const std::string cmd = line.substr(0, sp);
            const size_t a0 = sp == std::string::npos ? std::string::npos : line.find_first_not_of(" \t", sp);
            std::string arg = a0 == std::string::npos ? "" : line.substr(a0);
            while (!arg.empty() && std::isspace((unsigned char)arg.back())) arg.pop_back();
            const bool menus = term::interactive();
            try {
                if (cmd == "/exit" || cmd == "/quit") return false;
                else if (cmd == "/") {  // the command menu
                    static const std::vector<term::Choice> cmds = {
                        {"/config", "all settings, in a menu"},
                        {"/model", "switch to another model"},
                        {"/theme", "colors of the chat"},
                        {"/permission", "how tool actions are approved"},
                        {"/new", "start a new conversation"},
                        {"/load", "resume a saved conversation"},
                        {"/save", "save this conversation"},
                        {"/compact", "summarize the conversation into a fresh context"},
                        {"/think", "show / hide the reasoning"},
                        {"/undo", "revert the model's last file changes"},
                        {"/tools", "list the tools"},
                        {"/init", "have the model write AGENTS.md for this project"},
                        {"/info", "model, folders and memory use"},
                        {"/help", "every command"},
                        {"/exit", "quit"},
                    };
                    if (!menus) return command("/help");
                    const int idx = term::select("Commands", cmds, -1);
                    return idx < 0 ? true : command(cmds[idx].label);
                } else if (cmd == "/theme") {
                    if (arg.empty() && menus) {
                        if (pick_theme("Theme")) std::printf("%stheme: %s%s\n", term::sgr(term::theme().dim).c_str(), term::theme().name, off.c_str());
                    } else if (arg.empty()) {
                        for (auto& t : term::themes()) std::printf("  %-18s %s\n", t.name, t.desc);
                    } else {
                        if (!term::set_theme(arg)) throw std::runtime_error("no theme '" + arg + "' (/theme lists them)");
                        save_theme(arg);
                    }
                    apply_theme();
                } else if (cmd == "/settings" || cmd == "/config") {
                    if (!menus) {
                        print_settings();
                        return true;
                    }
                    // Settings menu: Enter changes the highlighted one, Esc leaves.
                    in_config = true;
                    struct Leave {
                        bool& f;
                        ~Leave() { f = false; }
                    } leave{in_config};
                    auto num = [](double v, const char* fmt) {
                        char b[32];
                        std::snprintf(b, sizeof b, fmt, v);
                        return std::string(b);
                    };
                    auto ask = [&](const std::string& what, const std::string& set_cmd) {  // a typed value -> a command
                        std::string v;
                        if (term::read_message("  " + what + " " + accent_b + "\xE2\x9D\xAF " + off, v) == term::Input::Line &&
                            v.find_first_not_of(" \t") != std::string::npos)
                            command(set_cmd + " " + v);
                    };
                    for (int at = 0;;) {
                        const std::string reasoning = !o.think ? "off" : o.hide_think ? "on, hidden" : "on, shown";
                        const std::vector<term::Choice> rows = {
                            {"Theme", term::theme().name},
                            {"Model", short_name(model_name) + " (" + be->name() + ")"},
                            {"Permission mode", tools_on ? o.permission : "(tools off)"},
                            {"Reasoning", reasoning},
                            {"Tools", tools_on ? "on" : "off"},
                            {"Temperature", o.greedy ? "greedy" : num(o.temperature, "%.2f")},
                            {"Top-p", num(o.top_p, "%.2f")},
                            {"Top-k", std::to_string(o.top_k)},
                            {"Max reply", thousands(o.max_new) + " tokens"},
                            {"Context size", thousands(be->max_ctx()) + " tokens (changing it reloads the model)"},
                            {"VRAM reserve", num(o.vram_reserve_gb, "%.1f GB") + " kept free (changing it reloads the model)"},
                            {"RAM reserve", num(o.ram_reserve_gb, "%.0f GB") + " kept free (changing it reloads the model)"},
                        };
                        at = term::select("Settings", rows, at);
                        if (at < 0) break;
                        switch (at) {
                            case 0: command("/theme"); break;
                            case 1: command("/model"); break;
                            case 2: command("/permission"); break;
                            case 3: {
                                const int r = term::select("Reasoning", {{"on, shown", "the reasoning is printed faintly before the answer"},
                                                                         {"on, hidden", "a spinner while the model thinks"},
                                                                         {"off", "no thinking: faster, weaker on hard questions"}},
                                                           !o.think ? 2 : o.hide_think ? 1 : 0);
                                if (r >= 0) {
                                    o.think = r != 2;
                                    if (r != 2) o.hide_think = r == 1;
                                }
                                break;
                            }
                            case 4: {
                                const int r = term::select("Tools (changing this starts a new conversation)",
                                                           {{"on", "files in the working folder, commands, web"}, {"off", ""}}, tools_on ? 0 : 1);
                                if (r >= 0 && (r == 0) != tools_on) command(r == 0 ? "/tools on" : "/tools off");
                                break;
                            }
                            case 5: ask("Temperature, 0 to 2 (0 = greedy)", "/set temp"); break;
                            case 6: ask("Top-p, 0.01 to 1", "/set top-p"); break;
                            case 7: ask("Top-k, 0 to 1000 (0 = off)", "/set top-k"); break;
                            case 8: ask("Max tokens per reply", "/set max-tokens"); break;
                            case 9: ask("Context size in tokens (e.g. 64k)", "/ctx"); break;
                            case 10: ask("VRAM to keep free, in GB", "/set vram-reserve"); break;
                            case 11: ask("RAM to keep free, in GB", "/set ram-reserve"); break;
                        }
                    }
                }
                else if (cmd == "/help" || cmd == "/?") {
                    std::printf(
                        "  /                  the command menu\n"
                        "  /config            all settings in a menu (also /settings)\n"
                        "  /theme [NAME]      colors: dark, light, dark-colorblind, light-colorblind, ansi, mono\n"
                        "  /model [NAME]      choose a model, or switch by name (the conversation carries over as a summary)\n"
                        "  /permission [MODE] how tool actions are approved: ask, edits, all, read-only\n"
                        "  /ctx [N]           show the context use, or reload with room for N tokens (e.g. /ctx 64k)\n"
                        "  /settings          all settings;  /set temp|top-p|top-k|greedy|max-tokens VALUE changes one\n"
                        "  /new               start a new conversation (also /clear, /reset)\n"
                        "  /think             show / hide the reasoning\n"
                        "  /reasoning on|off  turn thinking on or off\n"
                        "  /system [TEXT]     show the system prompt, or set it and start a new conversation\n"
                        "  /save [NAME]       save this conversation (default name: date and time)\n"
                        "  /load NAME         resume a saved conversation (/load last = the previous session)\n"
                        "  /sessions          list saved conversations\n"
                        "  /info              show the model, its weights folder and the working folder\n"
                        "  /image FILE        attach an image to the next message (/image lists, /image clear drops)\n"
                        "  /tools [on|off]    list the tools, or turn them on / off (starts a new conversation)\n"
                        "  /undo              revert the files changed by the model in its last turn that changed files\n"
                        "  /compact           summarize the conversation into a fresh context (also automatic when full)\n"
                        "  /init              have the model study the project and write AGENTS.md (instructions it reads)\n"
                        "  /exit              quit (also Ctrl+C twice)\n"
                        "%s  Input: Enter sends; pasted multi-line text is sent as one message; end a line with \\ to\n"
                        "  continue it, or put \"\"\" on its own line before and after a block. Ctrl+C stops a reply.\n"
                        "  Image files dropped into the window (or their paths typed in a message) are attached.%s\n",
                        dim.c_str(), off.c_str());
                } else if (cmd == "/model" || cmd == "/models") {
                    const auto models = available_models();
                    if (arg.empty() && menus) {  // choose from a menu
                        std::vector<term::Choice> items;
                        int cur = -1;
                        for (size_t i = 0; i < models.size(); i++) {
                            if (models[i].first == model_name) cur = (int)i;
                            items.push_back({models[i].first, !models[i].second.empty() ? models[i].second
                                                              : default_backend(models[i].first) == "cpu" ? "cpu backend by default" : "vulkan"});
                        }
                        const int idx = term::select("Model (the conversation carries over as a summary)", items, cur);
                        if (idx < 0 || idx == cur) return true;
                        arg = models[idx].first;
                    }
                    if (arg.empty()) {
                        std::printf("  %smodels in %s%s\n", dim.c_str(), u8str(model_path.parent_path()).c_str(), off.c_str());
                        for (auto& [name, why] : models) {
                            const bool cur = name == model_name;
                            std::printf("  %s%s%s %-28s %s%s%s\n", cur ? accent_b.c_str() : "", cur ? "\xE2\x97\x8F" : " ",
                                        off.c_str(), name.c_str(), dim.c_str(),
                                        cur ? ("current, " + std::string(be->name())).c_str()
                                            : !why.empty() ? why.c_str()
                                            : default_backend(name) == "cpu" ? "cpu backend by default" : "",
                                        off.c_str());
                        }
                        std::printf("  %s/model NAME [cpu|vulkan] switches (any unique part of the name works, e.g. 122b); the "
                                    "conversation is carried over as a summary%s\n", dim.c_str(), off.c_str());
                    } else {
                        std::string q = arg, backend;
                        if (const size_t sp2 = q.find_last_of(" \t"); sp2 != std::string::npos) {
                            const std::string last = q.substr(sp2 + 1);
                            if (last == "cpu" || last == "vulkan") {
                                backend = last;
                                q = q.substr(0, q.find_last_not_of(" \t", sp2) + 1);
                            }
                        }
                        std::string lq = q;
                        for (char& ch : lq) ch = (char)std::tolower((unsigned char)ch);
                        std::vector<std::string> hits;
                        for (auto& [name, why] : models) {
                            std::string ln = name;
                            for (char& ch : ln) ch = (char)std::tolower((unsigned char)ch);
                            if (ln == lq) { hits = {name}; break; }
                            if (ln.find(lq) != std::string::npos) hits.push_back(name);
                        }
                        if (hits.empty()) throw std::runtime_error("no model matches '" + q + "' (/model lists them)");
                        if (hits.size() > 1) {
                            std::string l;
                            for (auto& h : hits) l += (l.empty() ? "" : ", ") + h;
                            throw std::runtime_error("'" + q + "' matches several models: " + l);
                        }
                        const auto it = std::find_if(models.begin(), models.end(), [&](auto& m) { return m.first == hits[0]; });
                        if (!it->second.empty()) throw std::runtime_error(hits[0] + " is " + it->second);
                        if (backend.empty()) backend = default_backend(hits[0]);
                        if (hits[0] == model_name && backend == be->name()) {
                            std::printf("%salready using %s%s\n", dim.c_str(), hits[0].c_str(), off.c_str());
                        } else {
                            switch_model(u8str(model_path.parent_path() / u8path(hits[0])), backend, o.max_ctx);
                            set_title();
                        }
                    }
                } else if (cmd == "/ctx" || cmd == "/context") {
                    if (arg.empty()) {
                        std::printf("%scontext: %s / %s tokens used; /ctx N reloads the model with room for N tokens%s\n", dim.c_str(),
                                    thousands(be->pos()).c_str(), thousands(be->max_ctx()).c_str(), off.c_str());
                    } else {
                        int n = 0;
                        try { n = std::stoi(arg); } catch (...) {}
                        if (arg.back() == 'k' || arg.back() == 'K') n *= 1024;
                        if (n < 4096 || n > 262144) throw std::runtime_error("/ctx takes 4096 to 262144 tokens (e.g. /ctx 65536 or /ctx 64k)");
                        if (n != be->max_ctx()) switch_model(o.model_dir, o.backend, n);
                    }
                } else if (cmd == "/permission" || cmd == "/permissions") {
                    static const std::pair<const char*, const char*> modes[] = {
                        {"ask", "file changes and commands ask first (default)"},
                        {"edits", "file writes and edits are allowed; deleting and commands still ask"},
                        {"all", "nothing asks: the model changes files and runs commands on its own"},
                        {"read-only", "file changes and commands are refused without asking"}};
                    if (arg.empty() && menus) {
                        std::vector<term::Choice> items;
                        int cur = 0;
                        for (size_t i = 0; i < std::size(modes); i++) {
                            items.push_back({modes[i].first, modes[i].second});
                            if (o.permission == modes[i].first) cur = (int)i;
                        }
                        const int idx = term::select("Permission mode", items, cur);
                        if (idx < 0) return true;
                        arg = modes[idx].first;
                    }
                    if (arg.empty()) {
                        for (auto& [m, d] : modes)
                            std::printf("  %s%s%s %-10s %s%s%s\n", m == o.permission ? accent_b.c_str() : "",
                                        m == o.permission ? "\xE2\x97\x8F" : " ", off.c_str(), m, dim.c_str(), d, off.c_str());
                        if (!always_allow.empty()) {
                            std::string s;
                            for (auto& t : always_allow) s += (s.empty() ? "" : ", ") + t;
                            std::printf("  %salso allowed for this session: %s (/permission reset clears)%s\n", dim.c_str(), s.c_str(), off.c_str());
                        }
                    } else if (arg == "reset") {
                        always_allow.clear();
                        std::printf("%s'always' approvals cleared%s\n", dim.c_str(), off.c_str());
                    } else {
                        if (std::none_of(std::begin(modes), std::end(modes), [&](auto& m) { return arg == m.first; }))
                            throw std::runtime_error("usage: /permission ask|edits|all|read-only|reset");
                        o.permission = arg;
                        std::printf("%spermission: %s%s\n", dim.c_str(), arg.c_str(), off.c_str());
                    }
                } else if (cmd == "/set") {
                    const size_t sp2 = arg.find_first_of(" \t");
                    const std::string key = arg.substr(0, sp2), val = sp2 == std::string::npos ? "" : arg.substr(arg.find_first_not_of(" \t", sp2));
                    auto num = [&](double lo, double hi) {
                        double v;
                        try { v = std::stod(val); } catch (...) { throw std::runtime_error("/set " + key + " needs a number"); }
                        if (v < lo || v > hi) throw std::runtime_error("/set " + key + ": out of range");
                        return v;
                    };
                    if (key == "temp" || key == "temperature") { o.temperature = (float)num(0.0, 2.0); o.greedy = false; }
                    else if (key == "top-p") o.top_p = (float)num(0.01, 1.0);
                    else if (key == "top-k") o.top_k = (int)num(0, 1000);
                    else if (key == "greedy") o.greedy = val == "on" || val == "true" || val == "1";
                    else if (key == "max-tokens") o.max_new = (int)num(16, 1 << 20);
                    else if (key == "vram-reserve" || key == "ram-reserve") {
                        (key == "vram-reserve" ? o.vram_reserve_gb : o.ram_reserve_gb) = num(0, key == "vram-reserve" ? 32 : 1024);
                        switch_model(o.model_dir, o.backend, o.max_ctx);  // the cache and pinning are set up at load
                    }
                    else throw std::runtime_error("usage: /set temp|top-p|top-k|greedy|max-tokens|vram-reserve|ram-reserve VALUE");
                    if (!in_config) print_settings();
                } else if (cmd == "/undo") {
                    const std::vector<std::string> done = tools.undo();
                    if (done.empty()) {
                        std::printf("%snothing to undo (only file changes made by the model can be undone, not commands)%s\n",
                                    dim.c_str(), off.c_str());
                    } else {
                        std::string list;
                        for (const std::string& d : done) {
                            std::printf("  %s\xE2\x86\xB6 %s%s\n", dim.c_str(), d.c_str(), off.c_str());
                            list += (list.empty() ? "" : "; ") + d;
                        }
                        undo_note = "[Note: I reverted the file changes you made in an earlier turn (" + list +
                                    "). Those files are back to how they were before; I did this on purpose, so do not "
                                    "redo those changes unless I ask for them again.]";
                        std::printf("%sthe model will be told with your next message%s%s\n", dim.c_str(),
                                    tools.undo_depth() ? (" \xC2\xB7 " + std::to_string(tools.undo_depth()) + " more /undo steps").c_str() : "",
                                    off.c_str());
                    }
                } else if (cmd == "/compact") {
                    if (!compact(false)) std::printf("%snothing to summarize yet%s\n", dim.c_str(), off.c_str());
                    else autosave();
                } else if (cmd == "/init") {
                    if (!tools_on) throw std::runtime_error("/init needs the tools (/tools on)");
                    do_turn("Study this project so that a coding agent new to it can work on it well: look at its files, layout, "
                            "build and test setup, and conventions (use list_dir, read_file and search_text; read the README and "
                            "build files). Then create AGENTS.md in the working folder, or update it if it exists, with concise "
                            "instructions for AI coding agents: what the project is, how to build, run and test it (exact "
                            "commands), the code layout, code style and conventions, and important gotchas. Keep it under 80 "
                            "lines and only include what you verified.",
                            {});
                    autosave();
                } else if (cmd == "/tools") {
                    if (arg == "on" || arg == "off") {
                        tools_on = arg == "on";
                        always_allow.clear();
                        new_conversation();
                        std::printf("%stools %s; new conversation%s\n", dim.c_str(), arg.c_str(), off.c_str());
                    } else {
                        std::printf("  %stools are %s%s\n", dim.c_str(), tools_on ? "on" : "off (/tools on)", off.c_str());
                        std::printf("  free:         list_dir, read_file, search_text, web_search, fetch_url\n"
                                    "  asks first:   write_file, edit_file, delete_path, run_command (PowerShell)\n"
                                    "  %sfiles: only inside %s; commands start there. Answer 'a' at a prompt to allow\n"
                                    "  that tool for the rest of the session.%s\n",
                                    dim.c_str(), u8str(tools.root()).c_str(), off.c_str());
                        if (!always_allow.empty()) {
                            std::string s;
                            for (auto& t : always_allow) s += (s.empty() ? "" : ", ") + t;
                            std::printf("  %sallowed for this session: %s%s\n", dim.c_str(), s.c_str(), off.c_str());
                        }
                    }
                } else if (cmd == "/image" || cmd == "/img") {
                    if (arg == "clear") {
                        pending.clear();
                        std::printf("%sattachments dropped%s\n", dim.c_str(), off.c_str());
                    } else if (arg.empty()) {
                        if (pending.empty()) std::printf("%sno images attached (/image FILE)%s\n", dim.c_str(), off.c_str());
                        for (const ImageInput& im : pending) std::printf("  %s\n", image_label(im).c_str());
                    } else {
                        std::string msg = arg;
                        const std::vector<std::string> paths = take_image_paths(msg);
                        if (paths.empty()) throw std::runtime_error("not an image file: " + arg);
                        for (const std::string& p : paths) attach(p);
                    }
                } else if (cmd == "/new" || cmd == "/clear" || cmd == "/reset") {
                    new_conversation();
                    std::printf("%snew conversation%s\n", dim.c_str(), off.c_str());
                } else if (cmd == "/think") {
                    o.hide_think = !o.hide_think;
                    std::printf("%sreasoning %s%s\n", dim.c_str(), o.hide_think ? "hidden" : "shown", off.c_str());
                } else if (cmd == "/reasoning") {
                    if (arg == "on") o.think = true;
                    else if (arg == "off") o.think = false;
                    else throw std::runtime_error("usage: /reasoning on|off");
                    std::printf("%sthinking %s%s\n", dim.c_str(), o.think ? "on" : "off", off.c_str());
                } else if (cmd == "/system") {
                    if (arg.empty()) {
                        std::printf("%s%s%s\n", dim.c_str(), sess.system.empty() ? "(no system prompt)" : sess.system.c_str(), off.c_str());
                    } else {
                        o.system = arg;
                        new_conversation();
                        std::printf("%ssystem prompt set; new conversation%s\n", dim.c_str(), off.c_str());
                    }
                } else if (cmd == "/save") {
                    std::string name = arg;
                    if (name.empty()) {
                        const std::time_t now = std::time(nullptr);
                        char b[32];
                        std::strftime(b, sizeof b, "%Y-%m-%d_%H-%M-%S", std::localtime(&now));
                        name = b;
                    }
                    name = clean_name(name);
                    const fs::path p = session_dir / (name + ".json");
                    save_session(p, model_name, sess);
                    std::printf("%ssaved '%s' (%s)%s\n", dim.c_str(), name.c_str(), p.string().c_str(), off.c_str());
                } else if (cmd == "/load") {
                    if (arg.empty() && menus) {
                        const std::vector<std::string> names = session_names();
                        std::vector<term::Choice> items;
                        for (auto& n : names) items.push_back({n, ""});
                        if (names.empty()) list_sessions();
                        const int idx = term::select("Resume which conversation?", items, -1);
                        if (idx >= 0) resume(names[idx]);
                    } else if (arg.empty()) list_sessions();
                    else resume(arg);
                } else if (cmd == "/info") {
                    info();
                } else if (cmd == "/sessions" || cmd == "/list") {
                    list_sessions();
                } else {
                    std::printf("%sunknown command %s (/help lists them; start with a space to send text beginning with /)%s\n",
                                red.c_str(), cmd.c_str(), off.c_str());
                }
            } catch (const std::exception& e) {
                std::printf("%s%s%s\n", red.c_str(), e.what(), off.c_str());
            }
            return true;
        };
        bool exit_armed = false;  // a second Ctrl+C at the prompt quits
        if (!o.first_msg.empty()) {  // qwen "question": the first message
            std::printf("\n%s\xE2\x9D\xAF %s%s\n\n", accent_b.c_str(), off.c_str(), o.first_msg.c_str());
            try {
                do_turn(o.first_msg, {});
                autosave();
            } catch (const std::exception& e) {
                std::printf("%serror: %s%s\n", red.c_str(), e.what(), off.c_str());
            }
        }
        for (;;) {
            std::string line;
            const std::string prompt = "\n" + status_line() + "\n" +
                                       (pending.empty() ? "" : cyan + "[" + std::to_string(pending.size()) +
                                                                   (pending.size() == 1 ? " image] " : " images] ")) +
                                       accent_b + "\xE2\x9D\xAF " + off;
            const term::Input r = term::read_message(prompt, line);
            if (r == term::Input::Eof) break;
            if (r == term::Input::Cancel) {
                if (exit_armed) break;
                exit_armed = true;
                std::printf("%s(press Ctrl+C again or type /exit to quit)%s\n", dim.c_str(), off.c_str());
                continue;
            }
            exit_armed = false;
            if (line.find_first_not_of(" \t\n") == std::string::npos) continue;
            if (line[0] == '/') {
                if (!command(line)) break;
                continue;
            }
            try {
                // image files dropped into the window arrive as (quoted) paths: attach them
                std::string msg = line;
                for (const std::string& p : take_image_paths(msg)) attach(p);
                if (msg.empty()) {
                    std::printf("%snow type your message about %s%s\n", dim.c_str(), pending.size() == 1 ? "the image" : "the images", off.c_str());
                    continue;
                }
                do_turn(msg, pending);
                pending.clear();
                autosave();
            } catch (const std::exception& e) {
                std::printf("%serror: %s%s\n", red.c_str(), e.what(), off.c_str());
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
