#include "tools.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "http.h"
#include "json.h"
#include "vision.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

fs::path u8path(const std::string& s) { return fs::path(std::u8string(s.begin(), s.end())); }
std::string u8str(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}
std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}
std::string trim(const std::string& s) {
    const size_t b = s.find_first_not_of(" \t\r\n"), e = s.find_last_not_of(" \t\r\n");
    return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}
bool truthy(const std::string* v) {
    if (!v) return false;
    const std::string t = lower(trim(*v));
    return t == "true" || t == "1" || t == "yes";
}
int int_arg(const ToolCall& c, const char* k, int def, int lo, int hi) {
    const std::string* v = c.arg(k);
    if (!v || trim(*v).empty()) return def;
    try {
        return std::clamp(std::stoi(trim(*v)), lo, hi);
    } catch (...) {
        return def;
    }
}
const std::string& need(const ToolCall& c, const char* k) {
    const std::string* v = c.arg(k);
    if (!v) throw std::runtime_error(std::string("missing required parameter '") + k + "'");
    return *v;
}
std::string human_size(uintmax_t n) {
    char b[32];
    if (n < 1024) std::snprintf(b, sizeof b, "%llu B", (unsigned long long)n);
    else if (n < (1u << 20)) std::snprintf(b, sizeof b, "%.1f KB", n / 1024.0);
    else if (n < (1u << 30)) std::snprintf(b, sizeof b, "%.1f MB", n / 1048576.0);
    else std::snprintf(b, sizeof b, "%.1f GB", n / 1073741824.0);
    return b;
}
// cut to at most n bytes without splitting a UTF-8 sequence
std::string utf8_prefix(const std::string& s, size_t n) {
    if (s.size() <= n) return s;
    while (n > 0 && ((uint8_t)s[n] & 0xC0) == 0x80) n--;
    return s.substr(0, n);
}
size_t utf8_floor(const std::string& s, size_t n) {
    if (n >= s.size()) return s.size();
    while (n > 0 && ((uint8_t)s[n] & 0xC0) == 0x80) n--;
    return n;
}
// long outputs keep their head and tail
std::string clip(const std::string& s, size_t head, size_t tail) {
    if (s.size() <= head + tail) return s;
    const size_t h = utf8_floor(s, head), t = utf8_floor(s, s.size() - tail);
    return s.substr(0, h) + "\n... [" + std::to_string(t - h) + " bytes omitted] ...\n" + s.substr(t);
}
std::string read_bytes(const fs::path& p, size_t max = SIZE_MAX) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + u8str(p));
    std::string s;
    char buf[65536];
    while (s.size() < max) {
        f.read(buf, sizeof buf);
        const std::streamsize n = f.gcount();
        if (n <= 0) break;
        s.append(buf, (size_t)n);
    }
    if (s.size() > max) s.resize(max);
    return s;
}
bool looks_binary(const std::string& s) { return s.find('\0', 0) != std::string::npos && s.find('\0') < 8192; }

// File text as UTF-8 (UTF-8 BOM dropped, UTF-16 converted); sets `utf16` when the file was UTF-16.
std::string decode_text(const std::string& raw, bool* bom = nullptr, bool* utf16 = nullptr) {
    if (bom) *bom = false;
    if (utf16) *utf16 = false;
    if (raw.size() >= 3 && (uint8_t)raw[0] == 0xEF && (uint8_t)raw[1] == 0xBB && (uint8_t)raw[2] == 0xBF) {
        if (bom) *bom = true;
        return raw.substr(3);
    }
#ifdef _WIN32
    if (raw.size() >= 2 && (uint8_t)raw[0] == 0xFF && (uint8_t)raw[1] == 0xFE) {
        if (utf16) *utf16 = true;
        const std::wstring w((const wchar_t*)(raw.data() + 2), (raw.size() - 2) / 2);
        const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
        std::string s(n, '\0');
        WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
        return s;
    }
#endif
    return raw;
}

void write_bytes(const fs::path& p, const std::string& s) {
    if (p.has_parent_path()) fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f.write(s.data(), (std::streamsize)s.size())) throw std::runtime_error("cannot write " + u8str(p));
}

std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> v;
    size_t a = 0;
    while (a <= s.size()) {
        size_t e = s.find('\n', a);
        if (e == std::string::npos) {
            if (a < s.size()) v.push_back(s.substr(a));
            break;
        }
        std::string l = s.substr(a, e - a);
        if (!l.empty() && l.back() == '\r') l.pop_back();
        v.push_back(l);
        a = e + 1;
    }
    return v;
}

bool glob_match(const std::string& pat, const std::string& name) {  // * and ?, case-insensitive
    size_t p = 0, n = 0, star = std::string::npos, mark = 0;
    while (n < name.size()) {
        if (p < pat.size() && (pat[p] == '?' || std::tolower((unsigned char)pat[p]) == std::tolower((unsigned char)name[n]))) { p++; n++; }
        else if (p < pat.size() && pat[p] == '*') { star = p++; mark = n; }
        else if (star != std::string::npos) { p = star + 1; n = ++mark; }
        else return false;
    }
    while (p < pat.size() && pat[p] == '*') p++;
    return p == pat.size();
}

bool skip_dir(const fs::path& p) {
    const std::string n = lower(u8str(p.filename()));
    return n == ".git" || n == "node_modules" || n == "__pycache__" || n == ".venv" || n == ".hg" || n == ".svn";
}

std::string base64(const uint8_t* d, size_t n) {
    static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string o;
    for (size_t i = 0; i < n; i += 3) {
        const uint32_t v = (uint32_t)d[i] << 16 | (i + 1 < n ? (uint32_t)d[i + 1] << 8 : 0) | (i + 2 < n ? d[i + 2] : 0);
        o += t[v >> 18];
        o += t[(v >> 12) & 63];
        o += i + 1 < n ? t[(v >> 6) & 63] : '=';
        o += i + 2 < n ? t[v & 63] : '=';
    }
    return o;
}
std::string base64url_decode(const std::string& s) {
    std::string o;
    uint32_t v = 0;
    int bits = 0;
    for (char c : s) {
        int x;
        if (c >= 'A' && c <= 'Z') x = c - 'A';
        else if (c >= 'a' && c <= 'z') x = c - 'a' + 26;
        else if (c >= '0' && c <= '9') x = c - '0' + 52;
        else if (c == '-' || c == '+') x = 62;
        else if (c == '_' || c == '/') x = 63;
        else continue;
        v = v << 6 | (uint32_t)x;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            o += (char)((v >> bits) & 0xFF);
        }
    }
    return o;
}

// ---- processes ----------------------------------------------------------------------------------

struct ProcOut {
    int code = -1;
    std::string out;
    bool timed_out = false, cancelled = false;
};

#ifdef _WIN32
std::wstring widen(const std::string& s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
    return w;
}

std::string find_shell() {
    wchar_t buf[MAX_PATH];
    if (SearchPathW(nullptr, L"pwsh.exe", nullptr, MAX_PATH, buf, nullptr)) return u8str(fs::path(buf));
    if (SearchPathW(nullptr, L"powershell.exe", nullptr, MAX_PATH, buf, nullptr)) return u8str(fs::path(buf));
    return "powershell.exe";
}

// Runs a PowerShell script in `cwd` with stdin from NUL and stdout+stderr captured. The process tree lives
// in a job object so a timeout or cancel can end all of it; programs it starts detached keep running.
ProcOut run_powershell(const std::string& shell, const std::string& script, const fs::path& cwd, int timeout_s,
                       const std::function<bool()>& cancelled) {
    const std::string full = "$ProgressPreference='SilentlyContinue'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; "
                             "$OutputEncoding=[System.Text.Encoding]::UTF8\n" + script;
    const std::wstring wscript = widen(full);
    std::wstring cmd = L"\"" + widen(shell) + L"\" -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand " +
                       widen(base64((const uint8_t*)wscript.data(), wscript.size() * 2));
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) throw std::runtime_error("cannot create a pipe");
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
    STARTUPINFOW si{sizeof(si)};
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = nul;
    si.hStdOutput = wr;
    si.hStdError = wr;
    PROCESS_INFORMATION pi{};
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    const std::wstring wcwd = cwd.wstring();
    const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT,
                                   nullptr, wcwd.c_str(), &si, &pi);
    CloseHandle(wr);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!ok) {
        CloseHandle(rd);
        CloseHandle(job);
        throw std::runtime_error("cannot start " + shell);
    }
    AssignProcessToJobObject(job, pi.hProcess);
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);

    ProcOut r;
    std::mutex mu;
    std::thread reader([&] {
        char b[8192];
        DWORD n = 0;
        while (ReadFile(rd, b, sizeof b, &n, nullptr) && n) {
            std::lock_guard<std::mutex> l(mu);
            if (r.out.size() < (4u << 20)) r.out.append(b, n);
        }
    });
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        if (WaitForSingleObject(pi.hProcess, 100) == WAIT_OBJECT_0) break;
        if (cancelled && cancelled()) r.cancelled = true;
        else if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(timeout_s)) r.timed_out = true;
        if (r.cancelled || r.timed_out) {
            TerminateJobObject(job, 1);
            WaitForSingleObject(pi.hProcess, 5000);
            break;
        }
    }
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    r.code = (int)code;
    // a background child that inherited the pipe would keep the reader waiting: give it a moment, then cut it
    for (int i = 0; i < 30; i++) {
        DWORD avail = 0;
        if (!PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr)) break;  // writer side gone
        if (!avail) Sleep(100);
    }
    CancelSynchronousIo(reader.native_handle());
    reader.join();
    CloseHandle(rd);
    CloseHandle(pi.hProcess);
    CloseHandle(job);
    return r;
}
#else
std::string find_shell() { return "/bin/sh"; }
ProcOut run_powershell(const std::string&, const std::string& script, const fs::path& cwd, int, const std::function<bool()>&) {
    ProcOut r;
    const std::string cmd = "cd '" + cwd.string() + "' && (" + script + ") 2>&1";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) throw std::runtime_error("cannot start the shell");
    char b[4096];
    size_t n;
    while ((n = fread(b, 1, sizeof b, p)) > 0) r.out.append(b, n);
    r.code = pclose(p);
    return r;
}
#endif

// ---- web search --------------------------------------------------------------------------------

struct Hit { std::string title, url, snippet; };

std::string attr(const std::string& tag, const char* name) {
    const std::string key = std::string(name) + "=";
    size_t p = tag.find(key);
    if (p == std::string::npos) return "";
    p += key.size();
    const char q = tag[p];
    if (q != '"' && q != '\'') return "";
    const size_t e = tag.find(q, p + 1);
    return e == std::string::npos ? "" : html_decode_entities(tag.substr(p + 1, e - p - 1));
}

// <a ...class="cls"...>inner</a> occurrences: (opening tag, inner html)
std::vector<std::pair<std::string, std::string>> anchors(const std::string& html, const std::string& cls_marker) {
    std::vector<std::pair<std::string, std::string>> v;
    size_t p = 0;
    while ((p = html.find(cls_marker, p)) != std::string::npos) {
        const size_t a = html.rfind('<', p), gt = html.find('>', p);
        if (a == std::string::npos || gt == std::string::npos) break;
        const std::string tag = html.substr(a, gt - a + 1);
        const size_t end = html.find("</", gt);
        v.push_back({tag, html.substr(gt + 1, end == std::string::npos ? 0 : end - gt - 1)});
        p = gt;
    }
    return v;
}

std::string ddg_url(std::string href) {
    if (size_t u = href.find("uddg="); u != std::string::npos) {
        const size_t e = href.find('&', u);
        href = url_decode(href.substr(u + 5, e == std::string::npos ? std::string::npos : e - u - 5));
    }
    if (href.rfind("//", 0) == 0) href = "https:" + href;
    return href;
}

bool challenged(const HttpResponse& r) {
    return r.status != 200 || r.body.find("anomaly-modal") != std::string::npos || r.body.find("challenge-form") != std::string::npos;
}

std::vector<Hit> search_ddg_html(const std::string& q) {
    const HttpResponse r = http_request("https://html.duckduckgo.com/html/", "q=" + url_encode(q) + "&kl=wt-wt");
    if (challenged(r)) throw std::runtime_error("challenged");
    std::vector<Hit> hits;
    const auto titles = anchors(r.body, "class=\"result__a\"");
    const auto snippets = anchors(r.body, "class=\"result__snippet\"");
    for (size_t i = 0; i < titles.size(); i++) {
        const std::string href = attr(titles[i].first, "href");
        if (href.find("duckduckgo.com/y.js") != std::string::npos) continue;  // ad
        hits.push_back({strip_tags(titles[i].second), ddg_url(href), i < snippets.size() ? strip_tags(snippets[i].second) : ""});
    }
    return hits;
}

std::vector<Hit> search_ddg_lite(const std::string& q) {
    const HttpResponse r = http_request("https://lite.duckduckgo.com/lite/", "q=" + url_encode(q) + "&kl=wt-wt");
    if (challenged(r)) throw std::runtime_error("challenged");
    std::vector<Hit> hits;
    const auto links = anchors(r.body, "class='result-link'");
    std::vector<std::string> snips;
    size_t p = 0;
    while ((p = r.body.find("class='result-snippet'", p)) != std::string::npos) {
        const size_t gt = r.body.find('>', p), e = r.body.find("</td>", gt);
        if (gt == std::string::npos || e == std::string::npos) break;
        snips.push_back(strip_tags(r.body.substr(gt + 1, e - gt - 1)));
        p = e;
    }
    for (size_t i = 0; i < links.size(); i++) {
        const std::string href = attr(links[i].first, "href");
        if (href.find("duckduckgo.com/y.js") != std::string::npos) continue;
        hits.push_back({strip_tags(links[i].second), ddg_url(href), i < snips.size() ? snips[i] : ""});
    }
    return hits;
}

std::vector<Hit> search_bing(const std::string& q) {
    const HttpResponse r = http_request("https://www.bing.com/search?q=" + url_encode(q) + "&setlang=en&cc=US");
    if (r.status != 200) throw std::runtime_error("status " + std::to_string(r.status));
    std::vector<Hit> hits;
    size_t p = 0;
    while ((p = r.body.find("class=\"b_algo\"", p)) != std::string::npos) {
        const size_t next = r.body.find("class=\"b_algo\"", p + 10);
        const std::string blk = r.body.substr(p, (next == std::string::npos ? r.body.size() : next) - p);
        p = p + 10;
        const size_t h2 = blk.find("<h2");
        if (h2 == std::string::npos) continue;
        const size_t a = blk.find("<a ", h2), gt = blk.find('>', a), ae = blk.find("</a>", gt);
        if (a == std::string::npos || gt == std::string::npos || ae == std::string::npos) continue;
        std::string href = attr(blk.substr(a, gt - a + 1), "href");
        if (size_t u = href.find("&u=a1"); href.find("bing.com/ck/a") != std::string::npos && u != std::string::npos) {
            const size_t e = href.find('&', u + 5);
            href = base64url_decode(href.substr(u + 5, e == std::string::npos ? std::string::npos : e - u - 5));
        }
        std::string snippet;
        if (size_t ps = blk.find("<p", ae); ps != std::string::npos) {
            const size_t pg = blk.find('>', ps), pe = blk.find("</p>", pg);
            if (pg != std::string::npos && pe != std::string::npos) snippet = strip_tags(blk.substr(pg + 1, pe - pg - 1));
        }
        hits.push_back({strip_tags(blk.substr(gt + 1, ae - gt - 1)), href, snippet});
    }
    return hits;
}

// Brave Search API (optional: BRAVE_API_KEY), the reliable choice when the keyless pages are rate limited.
std::string brave_key() {
    for (const char* v : {"BRAVE_API_KEY", "QWEN_BRAVE_KEY"})
        if (const char* k = std::getenv(v); k && *k) return k;
    return "";
}

std::vector<Hit> search_brave(const std::string& q) {
    const HttpResponse r = http_request("https://api.search.brave.com/res/v1/web/search?count=10&q=" + url_encode(q), "", 20,
                                        8u << 20, "Accept: application/json\r\nX-Subscription-Token: " + brave_key() + "\r\n");
    if (r.status != 200) throw std::runtime_error("status " + std::to_string(r.status));
    const json::Value v = json::parse(r.body);
    std::vector<Hit> hits;
    if (const json::Value* web = v.find("web"))
        if (const json::Value* res = web->find("results"))
            for (const json::Value& h : res->arr) {
                const json::Value* d = h.find("description");
                hits.push_back({strip_tags(h["title"].as_str()), h["url"].as_str(), d ? strip_tags(d->as_str()) : ""});
            }
    return hits;
}

// ---- tool schemas ------------------------------------------------------------------------------

struct Param { const char* name; const char* type; const char* desc; bool required; };
struct Spec { const char* name; const char* desc; std::vector<Param> params; };

const std::vector<Spec>& specs() {
    static const std::vector<Spec> s = {
        {"list_dir", "List the files and folders in a folder of the working folder (sizes included).",
         {{"path", "string", "Folder path relative to the working folder (default \".\").", false},
          {"recursive", "boolean", "List subfolders too (default false).", false}}},
        {"read_file", "Read a text file from the working folder, with line numbers. For an image file the image itself is shown to you.",
         {{"path", "string", "File path relative to the working folder.", true},
          {"offset", "integer", "First line to read, 1-based (default 1).", false},
          {"limit", "integer", "Maximum number of lines (default 400).", false}}},
        {"search_text", "Search the text files under a folder for a regular expression; returns matching lines as path:line: text.",
         {{"pattern", "string", "Regular expression (ECMAScript syntax).", true},
          {"path", "string", "Folder or file to search (default \".\").", false},
          {"file_glob", "string", "Only files whose name matches this glob, e.g. \"*.py\".", false},
          {"case_sensitive", "boolean", "Default false.", false}}},
        {"write_file", "Create or overwrite a file in the working folder with the given content. The user must approve.",
         {{"path", "string", "File path relative to the working folder.", true},
          {"content", "string", "The complete new file content.", true}}},
        {"edit_file", "Replace exact text in a file of the working folder. old_text must match the file exactly (including indentation) and be unique unless replace_all is true. The user must approve.",
         {{"path", "string", "File path relative to the working folder.", true},
          {"old_text", "string", "The exact text to replace.", true},
          {"new_text", "string", "The replacement text.", true},
          {"replace_all", "boolean", "Replace every occurrence (default false).", false}}},
        {"delete_path", "Delete a file or a folder (with its contents) in the working folder. The user must approve.",
         {{"path", "string", "Path relative to the working folder.", true}}},
        {"run_command", "Run a PowerShell command in the working folder and return its output and exit code (stdin is empty; no interactive programs). Use it to run programs and scripts, build, test, or use git. The user must approve.",
         {{"command", "string", "The PowerShell command or script.", true},
          {"timeout", "integer", "Seconds before the command is stopped (default 120, max 1800).", false}}},
        {"web_search", "Search the web. Returns titles, URLs and snippets; use fetch_url to read a result.",
         {{"query", "string", "The search query.", true},
          {"max_results", "integer", "Default 8.", false}}},
        {"fetch_url", "Download a web page and return its readable text (or show an image).",
         {{"url", "string", "http or https URL.", true},
          {"start", "integer", "Character offset to continue a long page from (default 0).", false},
          {"max_chars", "integer", "Maximum characters returned (default 15000).", false}}},
    };
    return s;
}

std::string json_esc(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"') o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char)c;
    }
    return o + "\"";
}

}  // namespace

// ---------------------------------------------------------------------------------------------

const std::string* ToolCall::arg(const std::string& key) const {
    for (auto& [k, v] : args)
        if (k == key) return &v;
    return nullptr;
}

std::vector<ToolCall> parse_tool_calls(const std::string& text) {
    std::vector<ToolCall> calls;
    size_t p = 0;
    while ((p = text.find("<tool_call>", p)) != std::string::npos) {
        const size_t end = text.find("</tool_call>", p);
        const std::string blk = text.substr(p, end == std::string::npos ? std::string::npos : end - p);
        p = end == std::string::npos ? text.size() : end + 12;
        ToolCall c;
        const size_t f = blk.find("<function=");
        const size_t fe = f == std::string::npos ? f : blk.find('>', f);
        if (fe == std::string::npos || blk.find("</function>") == std::string::npos) {
            c.args.push_back({"error", "malformed tool call: expected <function=NAME> ... </function> inside <tool_call>"});
            calls.push_back(c);
            continue;
        }
        c.name = trim(blk.substr(f + 10, fe - f - 10));
        size_t q = fe;
        while ((q = blk.find("<parameter=", q)) != std::string::npos) {
            const size_t ke = blk.find('>', q), ve = blk.find("</parameter>", ke);
            if (ke == std::string::npos || ve == std::string::npos) break;
            std::string v = blk.substr(ke + 1, ve - ke - 1);
            if (!v.empty() && v.front() == '\n') v.erase(0, 1);
            if (!v.empty() && v.back() == '\n') v.pop_back();
            c.args.push_back({trim(blk.substr(q + 11, ke - q - 11)), v});
            q = ve + 12;
        }
        calls.push_back(c);
    }
    return calls;
}

Tools::Tools(fs::path root, bool images) : images_(images), shell_(find_shell()) {
    std::error_code ec;
    root_ = fs::weakly_canonical(fs::absolute(root), ec);
    if (ec) root_ = fs::absolute(root);
}

std::string Tools::system_prompt() const {
    std::string s = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const Spec& t : specs()) {
        std::string props, req;
        for (const Param& p : t.params) {
            props += (props.empty() ? "" : ", ") + json_esc(p.name) + ": {\"type\": " + json_esc(p.type) + ", \"description\": " + json_esc(p.desc) + "}";
            if (p.required) req += (req.empty() ? "" : ", ") + json_esc(p.name);
        }
        s += "\n{\"type\": \"function\", \"function\": {\"name\": " + json_esc(t.name) + ", \"description\": " + json_esc(t.desc) +
             ", \"parameters\": {\"type\": \"object\", \"properties\": {" + props + "}, \"required\": [" + req + "]}}}";
    }
    s += "\n</tools>";
    s += "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
         "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
         "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the "
         "specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required "
         "parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function "
         "call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do "
         "not tell the user about function calls\n</IMPORTANT>";
    return s;
}

std::string Tools::context_note() const {
    const std::time_t now = std::time(nullptr);
    char date[64];
    std::strftime(date, sizeof date, "%A, %Y-%m-%d", std::localtime(&now));
    std::string shell = lower(u8str(u8path(shell_).filename()));
    return "You are running locally on the user's computer (Windows) and can use the tools above. Working folder: " + u8str(root_) +
           " (file paths are relative to it; the file tools cannot leave it). Commands run in " +
           (shell == "pwsh.exe" ? std::string("PowerShell 7") : std::string("Windows PowerShell")) + ". Today is " + date +
           ". Use tools when they help; for changes, prefer edit_file for small edits and explain what you did.";
}

fs::path Tools::resolve(const std::string& p) const {
    std::string s = trim(p);
    if (s.size() >= 2 && (s.front() == '"' || s.front() == '\'') && s.back() == s.front()) s = s.substr(1, s.size() - 2);
    if (s.empty()) s = ".";
    fs::path q = u8path(s);
    if (q.is_relative()) q = root_ / q;
    std::error_code ec;
    fs::path c = fs::weakly_canonical(q, ec);
    if (ec) c = q.lexically_normal();
    const fs::path r = u8path(lower(u8str(c))).lexically_relative(u8path(lower(u8str(root_))));
    if (r.empty() || *r.begin() == "..") throw std::runtime_error("path is outside the working folder: " + s);
    return c;
}

std::string Tools::rel(const fs::path& p) const {
    const fs::path r = p.lexically_relative(root_);
    return r.empty() || u8str(r) == "." ? "." : u8str(r);
}

std::string Tools::describe(const ToolCall& c) const {
    auto a = [&](const char* k) { const std::string* v = c.arg(k); return v ? trim(*v) : std::string(); };
    if (c.name == "list_dir") return "list " + (a("path").empty() ? "." : a("path")) + (truthy(c.arg("recursive")) ? " (recursive)" : "");
    if (c.name == "read_file") return "read " + a("path") + (a("offset").empty() ? "" : " from line " + a("offset"));
    if (c.name == "search_text") return "search \"" + a("pattern") + "\" in " + (a("path").empty() ? "." : a("path")) + (a("file_glob").empty() ? "" : " (" + a("file_glob") + ")");
    if (c.name == "write_file") return "write " + a("path");
    if (c.name == "edit_file") return "edit " + a("path");
    if (c.name == "delete_path") return "delete " + a("path");
    if (c.name == "run_command") { std::string cmd = a("command"); return "run " + utf8_prefix(cmd.substr(0, cmd.find('\n')), 100); }
    if (c.name == "web_search") return "search the web: " + a("query");
    if (c.name == "fetch_url") return "fetch " + a("url");
    return c.name.empty() ? "malformed tool call" : c.name;
}

ToolResult Tools::run(const ToolCall& c, const Approver& approve) {
    try {
        if (c.name.empty()) throw std::runtime_error(c.args.empty() ? "malformed tool call" : c.args[0].second);
        if (c.name == "list_dir") return list_dir(c);
        if (c.name == "read_file") return read_file(c);
        if (c.name == "search_text") return search_text(c);
        if (c.name == "write_file") return write_file(c, approve);
        if (c.name == "edit_file") return edit_file(c, approve);
        if (c.name == "delete_path") return delete_path(c, approve);
        if (c.name == "run_command") return run_command(c, approve);
        if (c.name == "web_search") return web_search(c);
        if (c.name == "fetch_url") return fetch_url(c);
        throw std::runtime_error("unknown function '" + c.name + "'");
    } catch (const std::exception& e) {
        ToolResult r;
        r.ok = false;
        r.text = std::string("Error: ") + e.what();
        r.summary = e.what();
        return r;
    }
}

namespace {
ToolResult denied(const std::string& what, const std::string& note) {
    ToolResult r;
    r.ok = false;
    r.summary = "denied";
    r.text = "The user denied " + what + "." +
             (note.empty() ? " Do not try the same action again in another form: continue without it, or ask the user how to proceed."
                           : " The user said: " + note);
    return r;
}
}  // namespace

// ---- undo ---------------------------------------------------------------------------------------

constexpr uintmax_t kBackupMax = 64u << 20;  // a deleted folder larger than this is not kept

void Tools::backup(const fs::path& p) {
    Backup b;
    b.path = p;
    std::error_code ec;
    b.existed = fs::exists(p, ec);
    if (b.existed && fs::is_directory(p, ec)) {
        b.dir = true;
        uintmax_t total = 0;
        for (auto it = fs::recursive_directory_iterator(p, ec); it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) break;
            const std::string r = u8str(it->path().lexically_relative(p));
            if (it->is_directory(ec)) {
                b.tree.push_back({r + "/", ""});
            } else if (it->is_regular_file(ec)) {
                total += it->file_size(ec);
                if (total > kBackupMax) { b.kept = false; b.tree.clear(); break; }
                b.tree.push_back({r, read_bytes(it->path())});
            }
        }
    } else if (b.existed) {
        if (fs::file_size(p, ec) > kBackupMax) b.kept = false;
        else b.bytes = read_bytes(p);
    }
    current_.push_back(std::move(b));
}

void Tools::end_turn() {
    if (!current_.empty()) undo_.push_back(std::move(current_));
    current_.clear();
}

std::vector<std::string> Tools::undo() {
    std::vector<std::string> done;
    if (undo_.empty()) return done;
    std::vector<Backup> g = std::move(undo_.back());
    undo_.pop_back();
    std::error_code ec;
    for (auto it = g.rbegin(); it != g.rend(); ++it) {  // newest change first, so the oldest state wins
        const Backup& b = *it;
        try {
            if (!b.kept) {
                done.push_back("could not restore " + rel(b.path) + " (it was too large to back up)");
            } else if (!b.existed) {
                fs::remove_all(b.path, ec);
                done.push_back("removed " + rel(b.path) + " (it did not exist before)");
            } else if (b.dir) {
                fs::create_directories(b.path, ec);
                for (auto& [r, bytes] : b.tree) {
                    if (!r.empty() && r.back() == '/') fs::create_directories(b.path / u8path(r.substr(0, r.size() - 1)), ec);
                    else write_bytes(b.path / u8path(r), bytes);
                }
                done.push_back("restored folder " + rel(b.path));
            } else {
                write_bytes(b.path, b.bytes);
                done.push_back("restored " + rel(b.path));
            }
        } catch (const std::exception& e) {
            done.push_back("could not restore " + rel(b.path) + ": " + e.what());
        }
    }
    std::vector<std::string> unique;  // a file changed twice in the turn is reported once
    for (auto& d : done)
        if (std::find(unique.begin(), unique.end(), d) == unique.end()) unique.push_back(d);
    return unique;
}

ToolResult Tools::list_dir(const ToolCall& c) {
    const std::string* pa = c.arg("path");
    const fs::path dir = resolve(pa ? *pa : ".");
    if (!fs::is_directory(dir)) throw std::runtime_error("not a folder: " + rel(dir));
    const bool rec = truthy(c.arg("recursive"));
    std::vector<std::string> lines;
    size_t files = 0, dirs = 0;
    const size_t cap = 400;
    std::error_code ec;
    auto add = [&](const fs::directory_entry& e) {
        const bool d = e.is_directory(ec);
        std::string name = u8str(e.path().lexically_relative(dir));
        if (d) { dirs++; name += "/"; }
        else {
            files++;
            name += "  (" + human_size(e.file_size(ec)) + ")";
        }
        if (lines.size() < cap) lines.push_back(name);
    };
    if (rec) {
        for (auto it = fs::recursive_directory_iterator(dir, fs::directory_options::skip_permission_denied, ec);
             it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (ec) break;
            if (it->is_directory(ec) && skip_dir(it->path())) { add(*it); it.disable_recursion_pending(); continue; }
            add(*it);
        }
    } else {
        std::vector<fs::directory_entry> v;
        for (auto& e : fs::directory_iterator(dir, fs::directory_options::skip_permission_denied, ec)) v.push_back(e);
        std::sort(v.begin(), v.end(), [&](const fs::directory_entry& a, const fs::directory_entry& b) {
            const bool da = a.is_directory(ec), db = b.is_directory(ec);
            return da != db ? da : lower(u8str(a.path().filename())) < lower(u8str(b.path().filename()));
        });
        for (auto& e : v) add(e);
    }
    ToolResult r;
    r.text = "Folder " + rel(dir) + ": " + std::to_string(dirs) + " folders, " + std::to_string(files) + " files" +
             (lines.size() < files + dirs ? " (first " + std::to_string(cap) + " shown)" : "") + "\n";
    for (auto& l : lines) r.text += l + "\n";
    r.summary = std::to_string(dirs) + " folders, " + std::to_string(files) + " files";
    return r;
}

ToolResult Tools::read_file(const ToolCall& c) {
    const fs::path p = resolve(need(c, "path"));
    if (!fs::is_regular_file(p)) throw std::runtime_error(fs::is_directory(p) ? "is a folder (use list_dir): " + rel(p) : "no such file: " + rel(p));
    ToolResult r;
    if (is_image_path(u8str(p))) {
        if (!images_) throw std::runtime_error("this is an image, and image input is not available");
        r.image = u8str(p);
        r.text = "Image file " + rel(p) + " (" + human_size(fs::file_size(p)) + "):";
        r.summary = "image shown to the model";
        return r;
    }
    const std::string raw = read_bytes(p, 64u << 20);
    if (looks_binary(raw) && !(raw.size() >= 2 && (uint8_t)raw[0] == 0xFF && (uint8_t)raw[1] == 0xFE))
        throw std::runtime_error("binary file (" + human_size(fs::file_size(p)) + "), not shown: " + rel(p));
    const std::vector<std::string> lines = split_lines(decode_text(raw));
    const int offset = int_arg(c, "offset", 1, 1, std::max(1, (int)lines.size()));
    const int limit = int_arg(c, "limit", 400, 1, 5000);
    std::string body;
    int last = offset - 1;
    for (int i = offset - 1; i < (int)lines.size() && i < offset - 1 + limit; i++) {
        std::string l = lines[i].size() > 2000 ? utf8_prefix(lines[i], 2000) + " [line truncated]" : lines[i];
        if (body.size() + l.size() > 60000) break;
        body += std::to_string(i + 1) + ": " + l + "\n";
        last = i + 1;
    }
    r.text = rel(p) + " (lines " + std::to_string(offset) + "-" + std::to_string(last) + " of " + std::to_string(lines.size()) + ")" +
             (last < (int)lines.size() ? ", continue with offset=" + std::to_string(last + 1) : "") + "\n" + body;
    r.summary = "lines " + std::to_string(offset) + "-" + std::to_string(last) + " of " + std::to_string(lines.size());
    return r;
}

ToolResult Tools::search_text(const ToolCall& c) {
    const std::string& pat = need(c, "pattern");
    const std::string* pa = c.arg("path");
    const fs::path base = resolve(pa ? *pa : ".");
    const std::string* g = c.arg("file_glob");
    const std::string glob = g ? trim(*g) : "";
    std::regex re;
    try {
        re = std::regex(pat, truthy(c.arg("case_sensitive")) ? std::regex::ECMAScript : std::regex::ECMAScript | std::regex::icase);
    } catch (const std::regex_error& e) {
        throw std::runtime_error(std::string("invalid regular expression: ") + e.what());
    }
    std::string out;
    int matches = 0, files_hit = 0, scanned = 0;
    const int cap = 200;
    auto scan = [&](const fs::path& f) {
        std::error_code ec;
        if (!glob.empty() && !glob_match(glob, u8str(f.filename()))) return;
        if (fs::file_size(f, ec) > (4u << 20) || is_image_path(u8str(f))) return;
        std::string raw;
        try { raw = read_bytes(f); } catch (...) { return; }
        if (looks_binary(raw)) return;
        scanned++;
        const std::vector<std::string> lines = split_lines(decode_text(raw));
        bool hit = false;
        for (size_t i = 0; i < lines.size() && matches < cap; i++) {
            if (!std::regex_search(lines[i], re)) continue;
            hit = true;
            matches++;
            std::string l = trim(lines[i]);
            if (l.size() > 300) l = utf8_prefix(l, 300) + "...";
            out += rel(f) + ":" + std::to_string(i + 1) + ": " + l + "\n";
        }
        files_hit += hit;
    };
    std::error_code ec;
    if (fs::is_regular_file(base)) scan(base);
    else
        for (auto it = fs::recursive_directory_iterator(base, fs::directory_options::skip_permission_denied, ec);
             it != fs::recursive_directory_iterator() && matches < cap; it.increment(ec)) {
            if (ec) break;
            if (it->is_directory(ec)) {
                if (skip_dir(it->path())) it.disable_recursion_pending();
                continue;
            }
            if (it->is_regular_file(ec)) scan(it->path());
        }
    ToolResult r;
    r.text = std::to_string(matches) + " matching lines in " + std::to_string(files_hit) + " files (" + std::to_string(scanned) +
             " files searched" + (matches >= cap ? "; stopped at " + std::to_string(cap) + " matches" : "") + ")\n" + out;
    r.summary = std::to_string(matches) + " matches in " + std::to_string(files_hit) + " files";
    return r;
}

ToolResult Tools::write_file(const ToolCall& c, const Approver& approve) {
    const fs::path p = resolve(need(c, "path"));
    const std::string& content = need(c, "content");
    if (fs::is_directory(p)) throw std::runtime_error("is a folder: " + rel(p));
    const bool exists = fs::exists(p);
    const auto nl = split_lines(content);
    std::string preview = exists ? "overwrites " + rel(p) + " (" + std::to_string(split_lines(decode_text(read_bytes(p))).size()) + " lines) with " +
                                       std::to_string(nl.size()) + " lines"
                                 : "new file " + rel(p) + ", " + std::to_string(nl.size()) + " lines";
    for (size_t i = 0; i < nl.size() && i < 12; i++) preview += "\n+ " + utf8_prefix(nl[i], 200);
    if (nl.size() > 12) preview += "\n  ... " + std::to_string(nl.size() - 12) + " more lines";
    std::string note;
    if (!approve("write_file", "write " + rel(p), preview, note))
        return denied("writing " + rel(p), note);
    backup(p);
    write_bytes(p, content);
    ToolResult r;
    r.text = (exists ? "Overwrote " : "Created ") + rel(p) + " (" + std::to_string(nl.size()) + " lines, " + human_size(content.size()) + ").";
    r.summary = (exists ? "overwritten, " : "created, ") + std::to_string(nl.size()) + " lines";
    return r;
}

ToolResult Tools::edit_file(const ToolCall& c, const Approver& approve) {
    const fs::path p = resolve(need(c, "path"));
    if (!fs::is_regular_file(p)) throw std::runtime_error("no such file: " + rel(p));
    std::string old_text = need(c, "old_text"), new_text = need(c, "new_text");
    if (old_text.empty()) throw std::runtime_error("old_text is empty");
    bool bom = false, utf16 = false;
    std::string text = decode_text(read_bytes(p), &bom, &utf16);
    if (utf16) throw std::runtime_error("the file is UTF-16; rewrite it with write_file instead");
    const bool all = truthy(c.arg("replace_all"));
    auto count = [&](const std::string& needle) {
        size_t n = 0, q = 0;
        while ((q = text.find(needle, q)) != std::string::npos) { n++; q += needle.size(); }
        return n;
    };
    size_t n = count(old_text);
    if (n == 0 && text.find("\r\n") != std::string::npos) {  // the file uses CRLF line ends
        auto crlf = [](const std::string& s) {
            std::string o;
            for (size_t i = 0; i < s.size(); i++) {
                if (s[i] == '\n' && (i == 0 || s[i - 1] != '\r')) o += '\r';
                o += s[i];
            }
            return o;
        };
        old_text = crlf(old_text);
        new_text = crlf(new_text);
        n = count(old_text);
    }
    if (n == 0) throw std::runtime_error("old_text was not found in " + rel(p) + " (it must match exactly; read the file first)");
    if (n > 1 && !all)
        throw std::runtime_error("old_text occurs " + std::to_string(n) + " times in " + rel(p) + "; add surrounding lines to make it unique, or set replace_all");
    std::string preview;
    const auto ol = split_lines(old_text), nl = split_lines(new_text);
    auto add = [&](const std::string& l) { preview += (preview.empty() ? "" : "\n") + l; };
    for (size_t i = 0; i < ol.size() && i < 15; i++) add("- " + utf8_prefix(ol[i], 200));
    if (ol.size() > 15) add("  ... " + std::to_string(ol.size() - 15) + " more");
    for (size_t i = 0; i < nl.size() && i < 15; i++) add("+ " + utf8_prefix(nl[i], 200));
    if (nl.size() > 15) add("  ... " + std::to_string(nl.size() - 15) + " more");
    std::string note;
    if (!approve("edit_file", "edit " + rel(p) + (n > 1 && all ? " (" + std::to_string(n) + " places)" : ""), preview, note))
        return denied("the edit of " + rel(p), note);
    std::string out;
    size_t q = 0, prev = 0;
    while ((q = text.find(old_text, prev)) != std::string::npos) {
        out += text.substr(prev, q - prev) + new_text;
        prev = q + old_text.size();
        if (!all) break;
    }
    out += text.substr(prev);
    backup(p);
    write_bytes(p, (bom ? "\xEF\xBB\xBF" : "") + out);
    ToolResult r;
    r.text = "Edited " + rel(p) + ": replaced " + std::to_string(all ? n : 1) + " occurrence(s).";
    r.summary = "-" + std::to_string(ol.size()) + " +" + std::to_string(nl.size()) + " lines";
    return r;
}

ToolResult Tools::delete_path(const ToolCall& c, const Approver& approve) {
    const fs::path p = resolve(need(c, "path"));
    if (p == root_) throw std::runtime_error("refusing to delete the working folder itself");
    if (!fs::exists(p)) throw std::runtime_error("no such file or folder: " + rel(p));
    std::string what;
    uintmax_t total = 0;
    std::error_code ec;
    if (fs::is_directory(p)) {
        size_t n = 0;
        for (auto it = fs::recursive_directory_iterator(p, ec); it != fs::recursive_directory_iterator(); it.increment(ec))
            if (it->is_regular_file(ec)) { n++; total += it->file_size(ec); }
        what = "folder " + rel(p) + " and its " + std::to_string(n) + " files (" + human_size(total) + ")";
    } else {
        total = fs::file_size(p, ec);
        what = "file " + rel(p) + " (" + human_size(total) + ")";
    }
    std::string note;
    const std::string preview = "delete " + what + (total > kBackupMax ? "\ntoo large to keep a copy: /undo cannot bring it back" : "");
    if (!approve("delete_path", "delete " + rel(p), preview, note))
        return denied("deleting " + rel(p), note);
    backup(p);
    fs::remove_all(p);
    return {"Deleted " + what + ".", "", "deleted", "", true};
}

ToolResult Tools::run_command(const ToolCall& c, const Approver& approve) {
    const std::string cmd = need(c, "command");
    const int timeout = int_arg(c, "timeout", 120, 1, 1800);
    std::string note;
    if (!approve("run_command", "run a command", cmd, note))
        return denied("running the command", note);
    const auto t0 = std::chrono::steady_clock::now();
    const ProcOut o = run_powershell(shell_, cmd, root_, timeout, cancelled_);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::string out = o.out;
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ')) out.pop_back();
    std::string status = o.cancelled ? "stopped by the user" : o.timed_out ? "stopped after the " + std::to_string(timeout) + " s timeout"
                                                                            : "exit code " + std::to_string(o.code);
    ToolResult r;
    r.ok = !o.cancelled && !o.timed_out && o.code == 0;
    r.text = "[" + status + "]\n" + (out.empty() ? "(no output)" : clip(out, 4000, 12000));
    char b[32];
    std::snprintf(b, sizeof b, "%.1fs", secs);
    r.summary = status + ", " + b;
    r.detail = out;
    return r;
}

ToolResult Tools::web_search(const ToolCall& c) {
    const std::string q = trim(need(c, "query"));
    const int max = int_arg(c, "max_results", 8, 1, 20);
    std::vector<Hit> hits;
    std::string engine, errors;
    std::vector<std::pair<const char*, std::vector<Hit> (*)(const std::string&)>> engines = {
        {"DuckDuckGo", search_ddg_html}, {"DuckDuckGo Lite", search_ddg_lite}, {"Bing", search_bing}};
    if (!brave_key().empty()) engines.insert(engines.begin(), {"Brave", search_brave});
    for (auto& [name, fn] : engines) {
        try {
            hits = fn(q);
            if (!hits.empty()) { engine = name; break; }
            errors += std::string(name) + ": no results; ";
        } catch (const std::exception& e) {
            errors += std::string(name) + ": " + e.what() + "; ";
        }
    }
    if (hits.empty()) throw std::runtime_error("web search failed (" + errors + ")");
    ToolResult r;
    r.text = "Results for \"" + q + "\" (" + engine + "):\n";
    for (int i = 0; i < (int)hits.size() && i < max; i++)
        r.text += "\n" + std::to_string(i + 1) + ". " + hits[i].title + "\n   " + hits[i].url + (hits[i].snippet.empty() ? "" : "\n   " + hits[i].snippet) + "\n";
    r.summary = std::to_string(std::min<int>(max, (int)hits.size())) + " results (" + engine + ")";
    return r;
}

ToolResult Tools::fetch_url(const ToolCall& c) {
    const std::string url = trim(need(c, "url"));
    if (url.rfind("http://", 0) != 0 && url.rfind("https://", 0) != 0) throw std::runtime_error("only http and https URLs");
    const int start = int_arg(c, "start", 0, 0, 1 << 30), max_chars = int_arg(c, "max_chars", 15000, 500, 60000);
    const HttpResponse h = http_request(url);
    if (h.status >= 400) throw std::runtime_error("HTTP status " + std::to_string(h.status) + " for " + url);
    ToolResult r;
    const std::string& ct = h.content_type;
    if (ct.rfind("image/", 0) == 0) {
        if (!images_) throw std::runtime_error("the URL is an image (" + ct + "), and image input is not available");
        std::string ext = ct.substr(6);
        if (ext == "jpeg") ext = "jpg";
        if (!is_image_path("x." + ext)) throw std::runtime_error("unsupported image type " + ct);
        const fs::path tmp = fs::temp_directory_path() / ("qwen-web-" + std::to_string(std::hash<std::string>{}(url)) + "." + ext);
        write_bytes(tmp, h.body);
        r.image = u8str(tmp);
        r.text = "Image from " + h.final_url + ":";
        r.summary = "image shown to the model";
        return r;
    }
    std::string text, title;
    if (ct.find("html") != std::string::npos || ct.empty()) text = html_to_text(h.body, &title);
    else if (ct.rfind("text/", 0) == 0 || ct.find("json") != std::string::npos || ct.find("xml") != std::string::npos ||
             ct.find("javascript") != std::string::npos)
        text = h.body;
    else
        throw std::runtime_error("unsupported content type " + ct + " (" + human_size(h.body.size()) + ")");
    const size_t a = utf8_floor(text, (size_t)start), b = utf8_floor(text, std::min(text.size(), a + (size_t)max_chars));
    const bool thin = ct.find("html") != std::string::npos && text.size() < 1500;
    r.text = (title.empty() ? "" : "Title: " + title + "\n") + "URL: " + h.final_url + "\n(characters " + std::to_string(a) + "-" +
             std::to_string(b) + " of " + std::to_string(text.size()) + (b < text.size() ? "; continue with start=" + std::to_string(b) : "") +
             ")\n" + (thin ? "Note: this page has very little readable text. If you expected more, it is probably built by "
                             "JavaScript (which is not run), so fetching it again will not help: use another source.\n" : "") +
             "\n" + text.substr(a, b - a);
    r.summary = (title.empty() ? h.final_url : title) + " (" + human_size(text.size()) + " of text)";
    return r;
}
