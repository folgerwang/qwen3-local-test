#include "term.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <string_view>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <termios.h>
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
    if (std::getenv("CLICOLOR_FORCE") || std::getenv("FORCE_COLOR")) g_color = true;  // styled output into a pipe
    if (std::getenv("NO_COLOR")) g_color = false;
}

bool color() { return g_color; }
void catch_ctrl_c(bool on) { g_catch = on; }
bool interrupted() { return g_interrupt; }
void clear_interrupt() { g_interrupt = false; }

std::string sgr(const char* code) { return g_color && *code ? std::string("\x1b[") + code + "m" : std::string(); }

// ---------------------------------------------------------------------------------------------
// Themes

const std::vector<Theme>& themes() {
    // name, description, then SGR codes: accent, dim, ok, warn, err, heading, code, fence
    static const std::vector<Theme> v = {
        {"dark", "for dark terminals", "38;5;141", "38;5;245", "38;5;114", "38;5;179", "38;5;203", "1;38;5;141", "38;5;180", "38;5;110"},
        {"light", "for light terminals", "38;5;55", "38;5;243", "38;5;28", "38;5;130", "38;5;160", "1;38;5;55", "38;5;130", "38;5;24"},
        {"dark-colorblind", "dark, without red/green contrasts", "38;5;75", "38;5;245", "38;5;75", "38;5;220", "38;5;208",
         "1;38;5;75", "38;5;222", "38;5;153"},
        {"light-colorblind", "light, without red/green contrasts", "38;5;25", "38;5;243", "38;5;25", "38;5;94", "38;5;166",
         "1;38;5;25", "38;5;94", "38;5;24"},
        {"ansi", "the terminal's own 16 colors", "36", "2", "32", "33", "31", "1;35", "33", "36"},
        {"mono", "no colors, only bold and faint", "1", "2", "", "1", "1", "1", "4", ""},
    };
    return v;
}

namespace {
int g_theme = 0;
}

const Theme& theme() { return themes()[g_theme]; }

bool set_theme(const std::string& name) {
    const auto& v = themes();
    for (size_t i = 0; i < v.size(); i++)
        if (name == v[i].name) {
            g_theme = (int)i;
            return true;
        }
    return false;
}

// ---------------------------------------------------------------------------------------------
// Geometry, measuring, menus

int columns() {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO i;
    if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &i)) return std::max(20, i.srWindow.Right - i.srWindow.Left + 1);
#else
    winsize w{};
    if (ioctl(1, TIOCGWINSZ, &w) == 0 && w.ws_col) return std::max<int>(20, w.ws_col);
#endif
    return 80;
}

namespace {
// One UTF-8 character at s[i]: its code point and length in bytes.
uint32_t decode(const std::string& s, size_t i, int& len) {
    const uint8_t c = (uint8_t)s[i];
    len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (i + len > s.size()) len = 1;
    if (len == 1) return c;
    uint32_t cp = c & (0xFF >> (len + 1));
    for (int k = 1; k < len; k++) cp = (cp << 6) | ((uint8_t)s[i + k] & 0x3F);
    return cp;
}
int cp_width(uint32_t cp) {
    if (cp < 0x20 || (cp >= 0x300 && cp <= 0x36F) || cp == 0x200B || cp == 0xFE0F) return 0;
    const bool wide = (cp >= 0x1100 && cp <= 0x115F) || (cp >= 0x2E80 && cp <= 0xA4CF) || (cp >= 0xAC00 && cp <= 0xD7A3) ||
                      (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0xFE30 && cp <= 0xFE4F) || (cp >= 0xFF00 && cp <= 0xFF60) ||
                      (cp >= 0xFFE0 && cp <= 0xFFE6) || (cp >= 0x1F300 && cp <= 0x1FAFF) || (cp >= 0x20000 && cp <= 0x3FFFD);
    return wide ? 2 : 1;
}
}  // namespace

int width(const std::string& s) {
    int w = 0;
    for (size_t i = 0; i < s.size();) {
        if (s[i] == '\x1b') {  // escape sequence: ESC [ ... final byte
            size_t j = i + 1;
            if (j < s.size() && s[j] == '[') {
                j++;
                while (j < s.size() && !((uint8_t)s[j] >= 0x40 && (uint8_t)s[j] <= 0x7E)) j++;
            }
            i = std::min(j + 1, s.size());
            continue;
        }
        int len;
        w += cp_width(decode(s, i, len));
        i += len;
    }
    return w;
}

std::string fit(const std::string& s, int cols) {
    if (cols <= 0) return "";
    if (width(s) <= cols) return s;
    int w = 0;
    size_t i = 0;
    while (i < s.size()) {
        int len;
        const int cw = cp_width(decode(s, i, len));
        if (w + cw > cols - 1) break;
        w += cw;
        i += len;
    }
    return s.substr(0, i) + "\xE2\x80\xA6";
}

bool interactive() {
    if (!g_color) return false;
#ifdef _WIN32
    DWORD mode;
    return GetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), &mode) != 0;
#else
    return isatty(0);
#endif
}

namespace {
enum Key { K_UP = -1, K_DOWN = -2, K_ENTER = -3, K_ESC = -4, K_NONE = 0 };  // digits come back as 1..9

int read_key() {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_INPUT_HANDLE);
    for (;;) {
        if (g_interrupt.exchange(false)) return K_ESC;
        if (WaitForSingleObject(h, 50) != WAIT_OBJECT_0) continue;
        INPUT_RECORD r;
        DWORD n = 0;
        if (!ReadConsoleInputW(h, &r, 1, &n) || !n) return K_ESC;
        if (r.EventType != KEY_EVENT || !r.Event.KeyEvent.bKeyDown) continue;
        const WORD vk = r.Event.KeyEvent.wVirtualKeyCode;
        const wchar_t c = r.Event.KeyEvent.uChar.UnicodeChar;
        if (vk == VK_UP || c == L'k') return K_UP;
        if (vk == VK_DOWN || c == L'j' || vk == VK_TAB) return K_DOWN;
        if (vk == VK_RETURN) return K_ENTER;
        if (vk == VK_ESCAPE || c == L'q' || c == 3) return K_ESC;
        if (c >= L'1' && c <= L'9') return c - L'0';
    }
#else
    termios old{}, raw{};
    tcgetattr(0, &old);
    raw = old;
    raw.c_lflag &= ~(ICANON | ECHO);
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    tcsetattr(0, TCSANOW, &raw);
    int key = K_NONE;
    while (key == K_NONE) {
        unsigned char c;
        if (read(0, &c, 1) != 1 || g_interrupt.exchange(false)) { key = K_ESC; break; }
        if (c == 0x1b) {
            pollfd p{0, POLLIN, 0};
            unsigned char b[2];
            if (poll(&p, 1, 30) <= 0) { key = K_ESC; break; }
            if (read(0, b, 2) == 2 && b[0] == '[') key = b[1] == 'A' ? K_UP : b[1] == 'B' ? K_DOWN : K_NONE;
        } else if (c == 'k') key = K_UP;
        else if (c == 'j' || c == '\t') key = K_DOWN;
        else if (c == '\n' || c == '\r') key = K_ENTER;
        else if (c == 'q' || c == 3) key = K_ESC;
        else if (c >= '1' && c <= '9') key = c - '0';
    }
    tcsetattr(0, TCSANOW, &old);
    return key;
#endif
}
}  // namespace

int select(const std::string& title, const std::vector<Choice>& items, int current, const std::function<void(int)>& on_move,
           const std::function<std::string(int)>& footer) {
    const int n = (int)items.size();
    if (!interactive() || n == 0) return -1;
    int cur = std::clamp(current, 0, n - 1), top = 0, drawn = 0;
    const int vis = std::min(n, 12);
    int lw = 0;
    for (const Choice& c : items) lw = std::max(lw, width(c.label));
    auto draw = [&] {
        const Theme& t = theme();
        const int cols = columns();
        const std::string off = sgr("0"), dim = sgr(t.dim);
        if (cur < top) top = cur;
        if (cur >= top + vis) top = cur - vis + 1;
        std::string o;
        if (drawn) o += "\x1b[" + std::to_string(drawn) + "A";
        o += "\r\x1b[J";
        int lines = 0;
        auto line = [&](const std::string& s) {
            o += s + "\n";
            lines++;
        };
        line(" " + sgr("1") + fit(title, cols - 2) + off);
        for (int i = top; i < top + vis; i++) {
            std::string num = std::to_string(i + 1) + ". ";
            if (n > 9 && i < 9) num = " " + num;  // align one- and two-digit numbers
            const std::string label = fit(items[i].label, std::max(4, cols - 10));
            const int used = 3 + (int)num.size() + std::max(lw, width(label)) + 2 + 3;
            const std::string desc = cols - used > 8 ? fit(items[i].desc, cols - used) : "";
            const std::string pad((size_t)std::max(0, lw - width(label)) + 2, ' ');
            line((i == cur ? sgr(t.accent) + " \xE2\x9D\xAF " : "   ") + num + label + off + pad + dim + desc + off +
                 (i == current ? " " + sgr(t.ok) + "\xE2\x9C\x94" + off : ""));
        }
        if (n > vis) line(dim + "   " + std::to_string(cur + 1) + " of " + std::to_string(n) + off);
        if (footer) {
            const std::string f = footer(cur);
            o += f;
            lines += (int)std::count(f.begin(), f.end(), '\n');
        }
        line(dim + " \xE2\x86\x91/\xE2\x86\x93 move \xC2\xB7 Enter select \xC2\xB7 1-9 pick \xC2\xB7 Esc cancel" + off);
        drawn = lines;
        std::fputs(o.c_str(), stdout);
        std::fflush(stdout);
    };
    std::fputs("\x1b[?25l", stdout);
    int result = -1;
    for (;;) {
        draw();
        const int k = read_key();
        const int before = cur;
        if (k == K_UP) cur = (cur + n - 1) % n;
        else if (k == K_DOWN) cur = (cur + 1) % n;
        else if (k == K_ENTER) { result = cur; break; }
        else if (k == K_ESC) break;
        else if (k >= 1 && k <= 9 && k <= n) { result = k - 1; break; }
        if (cur != before && on_move) on_move(cur);
    }
    std::printf("\x1b[%dA\r\x1b[J\x1b[?25h", drawn);
    std::fflush(stdout);
    return result;
}

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
    const Theme& t = theme();
    std::string s = "\x1b[0";
    auto add = [&](const char* code) {
        if (*code) s += std::string(";") + code;
    };
    if (dim_) add(t.dim);  // the reasoning stays one faint color
    else {
        if (fence_) add(t.fence);
        if (heading_) add(t.heading);
        if (code_) add(t.code);
    }
    if (bold_) s += ";1";
    return s + "m";
}

// Puts the prefix in front of each line's first visible character (escape sequences pass through; the style
// in force, always a full "reset + attributes" sequence, is restored after the prefix).
std::string Markdown::indent(const std::string& o) {
    if (pfx_first_.empty() && pfx_rest_.empty()) return o;
    std::string r;
    for (char c : o) {
        if (in_esc_) {
            cur_esc_ += c;
            r += c;
            if (cur_esc_.size() > 2 && (uint8_t)c >= 0x40 && (uint8_t)c <= 0x7E) {
                in_esc_ = false;
                last_esc_ = cur_esc_;
            }
            continue;
        }
        if (c == '\x1b') {
            in_esc_ = true;
            cur_esc_ = c;
            r += c;
            continue;
        }
        if (c == '\n') {
            need_pfx_ = true;
            r += c;
            continue;
        }
        if (need_pfx_) {
            r += "\x1b[0m" + (first_line_ ? pfx_first_ : pfx_rest_) + last_esc_;
            need_pfx_ = first_line_ = false;
        }
        r += c;
    }
    return r;
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
    out_.put(indent(o));
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
    o += "\x1b[0m";
    out_.put(indent(o));
    out_.flush();
    need_pfx_ = first_line_ = true;
    in_esc_ = false;
    last_esc_.clear();
}

void Markdown::ch(char c, std::string& o) {
    if (fence_line_) {  // a ``` line: printed whole once complete
        if (c != '\n') {
            hold_ += c;
            return;
        }
        fence_ = !fence_;
        fence_line_ = false;
        o += "\x1b[0m" + sgr(theme().dim) + hold_ + "\n" + style();
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
            o += indent + "\x1b[0m" + sgr(theme().accent) + "\xE2\x80\xA2" + style() + " ";  // bullet
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
    std::printf("\r\x1b[0m%s%s\x1b[0m%s %s\xE2\x80\xA6 (%.0fs \xC2\xB7 %d tokens \xC2\xB7 Ctrl+C to stop)\x1b[0m\x1b[K",
                sgr(theme().accent).c_str(), frames[frame_], sgr(theme().dim).c_str(), label_.c_str(), seconds(), tokens);
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
