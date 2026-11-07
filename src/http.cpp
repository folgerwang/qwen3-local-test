#include "http.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")
#endif

namespace {

std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

#ifdef _WIN32
std::wstring widen(const std::string& s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring w(n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), n);
    return w;
}
std::string narrow(const std::wstring& w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0, nullptr, nullptr);
    std::string s(n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), s.data(), n, nullptr, nullptr);
    return s;
}

// Converts a body in a declared charset to UTF-8 (unknown charsets are left as they are).
std::string to_utf8(const std::string& body, std::string cs) {
    cs = lower(cs);
    UINT cp = 0;
    if (cs.empty() || cs == "utf-8" || cs == "utf8") return body;
    if (cs == "gbk" || cs == "gb2312" || cs == "x-gbk") cp = 936;
    else if (cs == "gb18030") cp = 54936;
    else if (cs == "big5") cp = 950;
    else if (cs == "shift_jis" || cs == "sjis" || cs == "x-sjis") cp = 932;
    else if (cs == "euc-jp") cp = 20932;
    else if (cs == "euc-kr" || cs == "ks_c_5601-1987") cp = 949;
    else if (cs == "iso-8859-1" || cs == "latin1" || cs == "windows-1252" || cs == "us-ascii") cp = 1252;
    else if (cs == "windows-1251") cp = 1251;
    else if (cs == "koi8-r") cp = 20866;
    else if (cs == "iso-8859-2") cp = 28592;
    if (!cp) return body;
    const int n = MultiByteToWideChar(cp, 0, body.data(), (int)body.size(), nullptr, 0);
    if (n <= 0) return body;
    std::wstring w(n, L'\0');
    MultiByteToWideChar(cp, 0, body.data(), (int)body.size(), w.data(), n);
    return narrow(w);
}

struct Handle {
    HINTERNET h = nullptr;
    ~Handle() { if (h) WinHttpCloseHandle(h); }
};
[[noreturn]] void fail(const char* what) {
    throw std::runtime_error(std::string("network error (") + what + ", code " + std::to_string(GetLastError()) + ")");
}
#endif

}  // namespace

HttpResponse http_request(const std::string& url, const std::string& form_body, int timeout_s, size_t max_bytes,
                          const std::string& extra_headers) {
#ifdef _WIN32
    const std::wstring wurl = widen(url);
    URL_COMPONENTS uc{sizeof(uc)};
    wchar_t host[256] = {}, path[4096] = {};
    uc.lpszHostName = host;
    uc.dwHostNameLength = 256;
    uc.lpszUrlPath = path;
    uc.dwUrlPathLength = 4096;
    wchar_t extra[4096] = {};
    uc.lpszExtraInfo = extra;
    uc.dwExtraInfoLength = 4096;
    if (!WinHttpCrackUrl(wurl.c_str(), 0, 0, &uc)) throw std::runtime_error("invalid URL: " + url);
    if (uc.nScheme != INTERNET_SCHEME_HTTP && uc.nScheme != INTERNET_SCHEME_HTTPS) throw std::runtime_error("only http and https URLs");
    Handle s, c, r;
    s.h = WinHttpOpen(L"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
                      WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!s.h) fail("open");
    const int ms = timeout_s * 1000;
    WinHttpSetTimeouts(s.h, ms, ms, ms, ms);
    DWORD dec = WINHTTP_DECOMPRESSION_FLAG_ALL;
    WinHttpSetOption(s.h, WINHTTP_OPTION_DECOMPRESSION, &dec, sizeof(dec));
    c.h = WinHttpConnect(s.h, host, uc.nPort, 0);
    if (!c.h) fail("connect");
    const std::wstring target = std::wstring(path) + extra;
    r.h = WinHttpOpenRequest(c.h, form_body.empty() ? L"GET" : L"POST", target.c_str(), nullptr, WINHTTP_NO_REFERER,
                             WINHTTP_DEFAULT_ACCEPT_TYPES, uc.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0);
    if (!r.h) fail("request");
    std::wstring headers = L"Accept: text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8\r\nAccept-Language: en-US,en;q=0.9\r\n";
    if (!form_body.empty()) headers += L"Content-Type: application/x-www-form-urlencoded\r\n";
    headers += widen(extra_headers);
    if (!WinHttpSendRequest(r.h, headers.c_str(), (DWORD)-1, (void*)form_body.data(), (DWORD)form_body.size(),
                            (DWORD)form_body.size(), 0))
        fail("send");
    if (!WinHttpReceiveResponse(r.h, nullptr)) fail("receive");

    HttpResponse resp;
    DWORD status = 0, sz = sizeof(status);
    WinHttpQueryHeaders(r.h, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status, &sz,
                        WINHTTP_NO_HEADER_INDEX);
    resp.status = (int)status;
    std::string ctype;
    {
        wchar_t buf[512];
        DWORD n = sizeof(buf);
        if (WinHttpQueryHeaders(r.h, WINHTTP_QUERY_CONTENT_TYPE, WINHTTP_HEADER_NAME_BY_INDEX, buf, &n, WINHTTP_NO_HEADER_INDEX))
            ctype = lower(narrow(std::wstring(buf, n / sizeof(wchar_t))));
    }
    {
        DWORD n = 0;
        WinHttpQueryOption(r.h, WINHTTP_OPTION_URL, nullptr, &n);
        std::wstring u(n / sizeof(wchar_t), L'\0');
        if (n && WinHttpQueryOption(r.h, WINHTTP_OPTION_URL, u.data(), &n)) {
            while (!u.empty() && u.back() == L'\0') u.pop_back();
            resp.final_url = narrow(u);
        } else {
            resp.final_url = url;
        }
    }
    for (;;) {
        DWORD avail = 0;
        if (!WinHttpQueryDataAvailable(r.h, &avail)) fail("read");
        if (!avail) break;
        const size_t old = resp.body.size();
        resp.body.resize(old + avail);
        DWORD got = 0;
        if (!WinHttpReadData(r.h, resp.body.data() + old, avail, &got)) fail("read");
        resp.body.resize(old + got);
        if (resp.body.size() > max_bytes) {
            resp.body.resize(max_bytes);
            break;
        }
    }
    // charset: header, else a <meta> near the top of the page
    std::string cs;
    if (size_t p = ctype.find("charset="); p != std::string::npos) cs = ctype.substr(p + 8);
    resp.content_type = ctype.substr(0, ctype.find(';'));
    if (cs.empty() && resp.content_type.find("html") != std::string::npos) {
        const std::string head = lower(resp.body.substr(0, 4096));
        if (size_t p = head.find("charset="); p != std::string::npos) {
            size_t b = p + 8;
            while (b < head.size() && (head[b] == '"' || head[b] == '\'')) b++;
            size_t e = b;
            while (e < head.size() && (std::isalnum((unsigned char)head[e]) || head[e] == '-' || head[e] == '_')) e++;
            cs = head.substr(b, e - b);
        }
    }
    while (!cs.empty() && (cs.back() == '"' || cs.back() == ' ')) cs.pop_back();
    if (!cs.empty() && cs.front() == '"') cs.erase(0, 1);
    resp.body = to_utf8(resp.body, cs);
    return resp;
#else
    (void)url; (void)form_body; (void)timeout_s; (void)max_bytes; (void)extra_headers;
    throw std::runtime_error("web access is only implemented on Windows");
#endif
}

std::string url_encode(const std::string& s) {
    std::string o;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') o += (char)c;
        else if (c == ' ') o += '+';
        else {
            char b[4];
            std::snprintf(b, sizeof b, "%%%02X", c);
            o += b;
        }
    }
    return o;
}

std::string url_decode(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '%' && i + 2 < s.size() && std::isxdigit((unsigned char)s[i + 1]) && std::isxdigit((unsigned char)s[i + 2])) {
            o += (char)std::stoi(s.substr(i + 1, 2), nullptr, 16);
            i += 2;
        } else if (s[i] == '+') {
            o += ' ';
        } else {
            o += s[i];
        }
    }
    return o;
}

namespace {
void put_utf8(std::string& o, uint32_t cp) {
    if (cp < 0x80) o += (char)cp;
    else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x110000) {
        o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 0x3F));
        o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F));
    }
}
}  // namespace

std::string html_decode_entities(const std::string& s) {
    static const std::pair<const char*, uint32_t> named[] = {
        {"amp", '&'}, {"lt", '<'}, {"gt", '>'}, {"quot", '"'}, {"apos", '\''}, {"nbsp", ' '}, {"mdash", 0x2014},
        {"ndash", 0x2013}, {"hellip", 0x2026}, {"rsquo", 0x2019}, {"lsquo", 0x2018}, {"ldquo", 0x201C}, {"rdquo", 0x201D},
        {"copy", 0xA9}, {"reg", 0xAE}, {"trade", 0x2122}, {"middot", 0xB7}, {"bull", 0x2022}, {"times", 0xD7},
        {"deg", 0xB0}, {"euro", 0x20AC}, {"pound", 0xA3}, {"laquo", 0xAB}, {"raquo", 0xBB}, {"rarr", 0x2192},
        {"larr", 0x2190}, {"shy", 0}, {"zwj", 0}, {"zwnj", 0}};
    std::string o;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] != '&') { o += s[i]; continue; }
        const size_t e = s.find(';', i);
        if (e == std::string::npos || e - i > 10) { o += '&'; continue; }
        const std::string name = s.substr(i + 1, e - i - 1);
        bool done = false;
        if (!name.empty() && name[0] == '#') {
            try {
                const uint32_t cp = (name.size() > 1 && (name[1] == 'x' || name[1] == 'X')) ? (uint32_t)std::stoul(name.substr(2), nullptr, 16)
                                                                                            : (uint32_t)std::stoul(name.substr(1));
                put_utf8(o, cp == 0xA0 ? ' ' : cp);
                done = true;
            } catch (...) {
            }
        } else {
            for (auto& [n, cp] : named)
                if (name == n) {
                    if (cp) put_utf8(o, cp);
                    done = true;
                    break;
                }
        }
        if (done) i = e;
        else o += '&';
    }
    return o;
}

std::string strip_tags(const std::string& html) {
    std::string o;
    bool tag = false;
    for (char c : html) {
        if (c == '<') tag = true;
        else if (c == '>') tag = false;
        else if (!tag) o += c;
    }
    o = html_decode_entities(o);
    std::string r;
    bool sp = false;
    for (char c : o) {
        if (std::isspace((unsigned char)c)) { sp = true; continue; }
        if (sp && !r.empty()) r += ' ';
        sp = false;
        r += c;
    }
    return r;
}

static std::string html_to_text_all(const std::string& html);

std::string html_to_text(const std::string& html, std::string* title) {
    const std::string lo = lower(html);
    if (title) {
        title->clear();
        const size_t a = lo.find("<title"), b = lo.find("</title>");
        if (a != std::string::npos && b != std::string::npos && b > a) {
            const size_t gt = lo.find('>', a);
            if (gt < b) *title = strip_tags(html.substr(gt + 1, b - gt - 1));
        }
    }
    // the page's main content when it marks it (<main>, else <article>), unless that is nearly empty
    for (const char* tag : {"main", "article"}) {
        const size_t a = lo.find(std::string("<") + tag), b = lo.rfind(std::string("</") + tag + ">");
        if (a == std::string::npos || b == std::string::npos || b <= a) continue;
        const char next = a + 1 + std::strlen(tag) < lo.size() ? lo[a + 1 + std::strlen(tag)] : '>';
        if (next != '>' && next != ' ') continue;  // e.g. <mainframe>
        std::string t = html_to_text_all(html.substr(a, b - a));
        if (t.size() > 500) return t;
    }
    return html_to_text_all(html);
}

static std::string html_to_text_all(const std::string& html) {
    const std::string lo = lower(html);
    std::string* title = nullptr;
    if (title) {
        title->clear();
        const size_t a = lo.find("<title"), b = lo.find("</title>");
        if (a != std::string::npos && b != std::string::npos && b > a) {
            const size_t gt = lo.find('>', a);
            if (gt < b) *title = strip_tags(html.substr(gt + 1, b - gt - 1));
        }
    }
    static const char* skip[] = {"script", "style", "noscript", "svg", "template", "head", "iframe", "nav", "aside", "footer",
                                 "button", "select"};
    static const char* blocks[] = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
                                   "header", "footer", "table", "ul", "ol", "pre", "blockquote", "dt", "dd", "hr", "main", "nav"};
    std::string out;
    int pre = 0;
    size_t i = 0;
    while (i < html.size()) {
        if (html[i] != '<') {
            const size_t e = html.find('<', i);
            std::string t = html_decode_entities(html.substr(i, (e == std::string::npos ? html.size() : e) - i));
            if (!pre) {
                std::string r;
                bool sp = !out.empty() && out.back() != '\n' && out.back() != ' ' ? false : true;
                for (char c : t) {
                    if (std::isspace((unsigned char)c)) {
                        if (!sp) r += ' ';
                        sp = true;
                    } else {
                        r += c;
                        sp = false;
                    }
                }
                t = r;
            }
            out += t;
            i = e == std::string::npos ? html.size() : e;
            continue;
        }
        if (lo.compare(i, 4, "<!--") == 0) {
            const size_t e = lo.find("-->", i + 4);
            i = e == std::string::npos ? html.size() : e + 3;
            continue;
        }
        // end of the tag, skipping '>' inside quoted attribute values
        size_t gt = i + 1;
        for (char q = 0; gt < lo.size(); gt++) {
            const char ch = lo[gt];
            if (q) { if (ch == q) q = 0; }
            else if (ch == '"' || ch == '\'') { if (gt > 0 && lo[gt - 1] == '=') q = ch; }
            else if (ch == '>') break;
        }
        if (gt >= lo.size()) break;
        const bool closing = i + 1 < lo.size() && lo[i + 1] == '/';
        size_t ns = i + (closing ? 2 : 1), ne = ns;
        while (ne < gt && (std::isalnum((unsigned char)lo[ne]))) ne++;
        const std::string name = lo.substr(ns, ne - ns);
        i = gt + 1;
        if (!closing && std::find_if(std::begin(skip), std::end(skip), [&](const char* s) { return name == s; }) != std::end(skip)) {
            const size_t e = lo.find("</" + name, i);
            i = e == std::string::npos ? html.size() : lo.find('>', e) + 1;
            if (i == 0) i = html.size();
            continue;
        }
        if (name == "pre") pre += closing ? -1 : 1;
        if (pre < 0) pre = 0;
        if (std::find_if(std::begin(blocks), std::end(blocks), [&](const char* s) { return name == s; }) != std::end(blocks)) {
            while (!out.empty() && out.back() == ' ') out.pop_back();
            if (!out.empty() && out.back() != '\n') out += '\n';
            if (!closing && name == "li") out += "- ";
            if (!closing && name.size() == 2 && name[0] == 'h' && name[1] >= '1' && name[1] <= '6') out += std::string(name[1] - '0', '#') + " ";
        } else if (name == "td" || name == "th") {
            if (!closing && !out.empty() && out.back() != '\n') out += " | ";
        }
    }
    // at most one blank line in a row
    std::string r;
    int nl = 0;
    for (char c : out) {
        if (c == '\n') {
            if (++nl > 2) continue;
            while (!r.empty() && r.back() == ' ') r.pop_back();
        } else if (c != ' ' || !r.empty()) {
            nl = 0;
        }
        r += c;
    }
    const size_t b = r.find_first_not_of(" \n");
    return b == std::string::npos ? "" : r.substr(b);
}
