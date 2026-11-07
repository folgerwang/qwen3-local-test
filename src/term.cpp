#include "term.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <poll.h>
#include <signal.h>
#include <unistd.h>
#endif

namespace term {

namespace {

std::atomic<bool> g_interrupt{false}, g_catch{false};
bool g_color = false;

#ifdef _WIN32
BOOL WINAPI on_ctrl(DWORD type) {
    if (type != CTRL_C_EVENT || !g_catch) return FALSE;  // default handling: terminate
    g_interrupt = true;
    return TRUE;
}
#else
void on_sigint(int) {
    if (!g_catch) _exit(130);
    g_interrupt = true;
}
#endif

std::string trim(const std::string& s) {
    const size_t b = s.find_first_not_of(" \t"), e = s.find_last_not_of(" \t");
    return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}

// One line without its terminator.
Input read_line(std::string& out) {
    out.clear();
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    if (GetConsoleMode(h, &mode)) {
        std::wstring w;
        wchar_t buf[1024];
        for (;;) {
            DWORD n = 0;
            if (!ReadConsoleW(h, buf, 1024, &n, nullptr) || n == 0) {
                // Ctrl+C ends the read; the handler runs on its own thread, so give it a moment to set the flag
                for (int i = 0; i < 20 && !g_interrupt; i++) Sleep(5);
                if (g_interrupt.exchange(false)) return Input::Cancel;
                return Input::Eof;
            }
            w.append(buf, n);
            if (w.back() == L'\n') break;
        }
        while (!w.empty() && (w.back() == L'\n' || w.back() == L'\r')) w.pop_back();
        if (w == L"\x1a") return Input::Eof;  // Ctrl+Z, Enter
        const int len = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
        out.assign(len, '\0');
        WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), out.data(), len, nullptr, nullptr);
        return Input::Line;
    }
#endif
    if (!std::getline(std::cin, out)) {
        if (g_interrupt.exchange(false)) {
            std::cin.clear();
            return Input::Cancel;
        }
        return Input::Eof;
    }
    if (!out.empty() && out.back() == '\r') out.pop_back();
    return Input::Line;
}

// True if more typed characters are already waiting, i.e. the line just read came from a multi-line paste.
bool input_pending() {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    if (!GetConsoleMode(h, &mode)) return false;
    Sleep(15);  // a paste arrives as a burst of key events
    DWORD n = 0;
    if (!GetNumberOfConsoleInputEvents(h, &n) || n == 0) return false;
    std::vector<INPUT_RECORD> rec(n);
    DWORD got = 0;
    if (!PeekConsoleInputW(h, rec.data(), n, &got)) return false;
    for (DWORD i = 0; i < got; i++) {
        const KEY_EVENT_RECORD& k = rec[i].Event.KeyEvent;
        if (rec[i].EventType == KEY_EVENT && k.bKeyDown && k.uChar.UnicodeChar) return true;
    }
    return false;
#else
    if (!isatty(0)) return false;
    if (std::cin.rdbuf()->in_avail() > 0) return true;
    usleep(15000);
    pollfd p{0, POLLIN, 0};
    return poll(&p, 1, 0) > 0;
#endif
}

}  // namespace

void init() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode;
    if (GetConsoleMode(out, &mode) && SetConsoleMode(out, mode | ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING))
        g_color = true;
    SetConsoleCtrlHandler(on_ctrl, TRUE);
#else
    g_color = isatty(1);
    struct sigaction sa {};
    sa.sa_handler = on_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;  // no SA_RESTART: a blocked read returns so the chat can cancel the line
    sigaction(SIGINT, &sa, nullptr);
#endif
    if (std::getenv("NO_COLOR")) g_color = false;
}

bool color() { return g_color; }
void catch_ctrl_c(bool on) { g_catch = on; }
bool interrupted() { return g_interrupt; }
void clear_interrupt() { g_interrupt = false; }

std::string sgr(const char* code) { return g_color ? std::string("\x1b[") + code + "m" : std::string(); }

Input read_message(const std::string& prompt, std::string& out) {
    out.clear();
    const std::string cont = sgr("2") + "... " + sgr("0");
    std::fputs(prompt.c_str(), stdout);
    std::fflush(stdout);
    bool block = false, any = false;
    std::string line;
    for (;;) {
        const Input r = read_line(line);
        if (r != Input::Line) {
            std::fputs("\n", stdout);
            return r;
        }
        const std::string t = trim(line);
        if (t == "\"\"\"") {
            if (block) return Input::Line;
            if (!any) {
                block = true;
                std::fputs(cont.c_str(), stdout);
                std::fflush(stdout);
                continue;
            }
        }
        bool more = block;
        if (!block && !line.empty() && line.back() == '\\') {
            line.pop_back();
            more = true;
        }
        if (any) out += '\n';
        out += line;
        any = true;
        if (!more && !input_pending()) return Input::Line;
        std::fputs(cont.c_str(), stdout);
        std::fflush(stdout);
    }
}

// ---------------------------------------------------------------------------------------------

void Utf8Out::put(const std::string& bytes) {
    pending_ += bytes;
    // walk back over continuation bytes to the last lead byte and check whether its sequence is complete
    const size_t n = pending_.size();
    size_t i = n, back = 0;
    while (i > 0 && back < 4 && ((uint8_t)pending_[i - 1] & 0xC0) == 0x80) { i--; back++; }
    size_t ok = n;
    if (i == 0) ok = back >= 4 ? n : 0;
    else {
        const uint8_t lead = (uint8_t)pending_[i - 1];
        const size_t need = lead < 0x80 ? 1 : (lead >> 5) == 6 ? 2 : (lead >> 4) == 14 ? 3 : (lead >> 3) == 30 ? 4 : 1;
        if (n - (i - 1) < need) ok = i - 1;
    }
    if (!ok) return;
    std::fwrite(pending_.data(), 1, ok, stdout);
    std::fflush(stdout);
    pending_.erase(0, ok);
}

void Utf8Out::flush() {
    std::fwrite(pending_.data(), 1, pending_.size(), stdout);
    std::fflush(stdout);
    pending_.clear();
}

// ---------------------------------------------------------------------------------------------

std::string Markdown::style() const {
    std::string s = "\x1b[0";
    if (dim_) s += ";2";
    if (fence_) s += ";36";
    if (heading_) s += ";1;35";
    if (bold_) s += ";1";
    if (code_) s += ";33";
    return s + "m";
}

void Markdown::put(const std::string& text) {
    if (!on_) {
        out_.put(text);
        return;
    }
    std::string o;
    if (!started_) {
        started_ = true;
        o += style();
    }
    for (char c : text) ch(c, o);
    out_.put(o);
}

void Markdown::finish() {
    if (!on_) {
        out_.flush();
        return;
    }
    std::string o;
    if (fence_line_) o += hold_;
    else {
        std::string h;
        h.swap(hold_);
        bol_ = false;
        for (char c : h) inline_ch(c, o);
    }
    if (star_) o += '*';
    hold_.clear();
    bol_ = true;
    fence_ = fence_line_ = code_ = bold_ = star_ = heading_ = false;
    started_ = false;
    if (on_) o += "\x1b[0m";
    out_.put(o);
    out_.flush();
}

void Markdown::ch(char c, std::string& o) {
    if (fence_line_) {  // a ``` line: printed whole once complete
        if (c != '\n') {
            hold_ += c;
            return;
        }
        fence_ = !fence_;
        fence_line_ = false;
        o += "\x1b[0;2m" + hold_ + "\n" + style();
        hold_.clear();
        bol_ = true;
        return;
    }
    if (bol_) {
        hold_ += c;
        line_start(o);
        return;
    }
    if (fence_) {
        o += c;
        if (c == '\n') bol_ = true;
        return;
    }
    inline_ch(c, o);
}

// Decides what the held start of a line is once enough of it has arrived.
void Markdown::line_start(std::string& o) {
    const size_t i = hold_.find_first_not_of(' ');
    if (i == std::string::npos) return;  // only indentation so far
    const std::string_view r(hold_.data() + i, hold_.size() - i);
    const std::string indent = hold_.substr(0, i);
    auto text = [&] {  // ordinary line: replay the held characters through the inline styling
        std::string h;
        h.swap(hold_);
        bol_ = false;
        for (char x : h) inline_ch(x, o);
    };
    if (r.find_first_not_of('`') == std::string_view::npos) {
        if (r.size() == 3) fence_line_ = true;
        return;
    }
    if (fence_) {
        o += hold_;
        bol_ = r.back() == '\n';
        hold_.clear();
        return;
    }
    if (r[0] == '#') {
        const size_t k = r.find_first_not_of('#');
        if (k == std::string_view::npos && r.size() <= 6) return;
        if (k != std::string_view::npos && k <= 6 && r[k] == ' ') {
            heading_ = true;
            o += indent + style();
            hold_.clear();
            bol_ = false;
            return;
        }
        return text();
    }
    if (r[0] == '-' || r[0] == '*' || r[0] == '+') {
        if (r.size() == 1) return;
        if (r[1] == ' ') {
            o += indent + "\x1b[0;36m" + "\xE2\x80\xA2" + style() + " ";  // bullet
            hold_.clear();
            bol_ = false;
            return;
        }
    }
    text();
}

void Markdown::inline_ch(char c, std::string& o) {
    if (star_) {
        star_ = false;
        if (c == '*') {
            bold_ = !bold_;
            o += style();
            return;
        }
        o += '*';
    }
    if (c == '*' && !code_) {
        star_ = true;
        return;
    }
    if (c == '`') {
        code_ = !code_;
        o += style();
        return;
    }
    if (c == '\n') {  // inline styles never run past the end of a line
        const bool reset = bold_ || code_ || heading_;
        bold_ = code_ = heading_ = false;
        if (reset) o += style();
        o += '\n';
        bol_ = true;
        return;
    }
    o += c;
}

// ---------------------------------------------------------------------------------------------

double Spinner::seconds() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
}

void Spinner::update(int tokens) {
    if (!on_) return;
    const auto now = std::chrono::steady_clock::now();
    if (drawn_ && now - last_ < std::chrono::milliseconds(100)) return;
    last_ = now;
    static const char* frames[] = {"\xE2\xA0\x8B", "\xE2\xA0\x99", "\xE2\xA0\xB9", "\xE2\xA0\xB8", "\xE2\xA0\xBC",
                                   "\xE2\xA0\xB4", "\xE2\xA0\xA6", "\xE2\xA0\xA7", "\xE2\xA0\x87", "\xE2\xA0\x8F"};
    frame_ = (frame_ + 1) % 10;
    std::printf("\r\x1b[0;36m%s\x1b[0;2m %s %.0fs \xC2\xB7 %d tokens \xC2\xB7 Ctrl+C to stop\x1b[0m\x1b[K",
                frames[frame_], label_.c_str(), seconds(), tokens);
    std::fflush(stdout);
    drawn_ = true;
}

void Spinner::clear() {
    if (!drawn_) return;
    std::printf("\r\x1b[K");
    std::fflush(stdout);
    drawn_ = false;
}

}  // namespace term
