#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "json.h"
#include "unicode_tables.h"

namespace {

// ---------------------------------------------------------------------------------------------
// UTF-8
// ---------------------------------------------------------------------------------------------
std::vector<uint32_t> utf8_decode(const std::string& s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        uint8_t c = (uint8_t)s[i];
        uint32_t cp;
        int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 6) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 14) { cp = c & 0x0F; len = 3; }
        else if ((c >> 3) == 30) { cp = c & 0x07; len = 4; }
        else { out.push_back(0xFFFD); i++; continue; }
        if (i + len > s.size()) { out.push_back(0xFFFD); i++; continue; }
        bool ok = true;
        for (int k = 1; k < len; k++) {
            uint8_t cc = (uint8_t)s[i + k];
            if ((cc >> 6) != 2) { ok = false; break; }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (!ok) { out.push_back(0xFFFD); i++; continue; }
        out.push_back(cp);
        i += len;
    }
    return out;
}

void utf8_append(std::string& out, uint32_t cp) {
    if (cp < 0x80) out += (char)cp;
    else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
        out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
    } else {
        out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F));
        out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
    }
}

// ---------------------------------------------------------------------------------------------
// Unicode properties
// ---------------------------------------------------------------------------------------------
template <size_t N>
bool in_ranges(const uint32_t (&r)[N][2], uint32_t cp) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < r[mid][0]) hi = mid;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}
bool is_L(uint32_t c) { return c < 0x80 ? ((c | 0x20) >= 'a' && (c | 0x20) <= 'z') : in_ranges(kLetter, c); }
bool is_M(uint32_t c) { return c >= 0x300 && in_ranges(kMark, c); }
bool is_N(uint32_t c) { return c < 0x80 ? (c >= '0' && c <= '9') : in_ranges(kNumber, c); }
bool is_S(uint32_t c) { return in_ranges(kWhiteSpace, c); }
bool is_nl(uint32_t c) { return c == '\r' || c == '\n'; }

// ---------------------------------------------------------------------------------------------
// NFC normalization
// ---------------------------------------------------------------------------------------------
constexpr uint32_t SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7;
constexpr uint32_t LCount = 19, VCount = 21, TCount = 28, NCount = VCount * TCount, SCount = LCount * NCount;

uint8_t ccc_of(uint32_t cp) {
    if (cp < 0x300) return 0;
    auto it = std::lower_bound(std::begin(kCcc), std::end(kCcc), cp, [](const UCcc& a, uint32_t v) { return a.cp < v; });
    return (it != std::end(kCcc) && it->cp == cp) ? it->ccc : 0;
}

void decompose(uint32_t cp, std::vector<uint32_t>& out) {
    if (cp >= SBase && cp < SBase + SCount) {
        uint32_t s = cp - SBase;
        out.push_back(LBase + s / NCount);
        out.push_back(VBase + (s % NCount) / TCount);
        if (s % TCount) out.push_back(TBase + s % TCount);
        return;
    }
    if (cp >= 0xC0) {
        auto it = std::lower_bound(std::begin(kDecomp), std::end(kDecomp), cp,
                                   [](const UDecomp& a, uint32_t v) { return a.cp < v; });
        if (it != std::end(kDecomp) && it->cp == cp) {
            decompose(it->a, out);
            if (it->b) decompose(it->b, out);
            return;
        }
    }
    out.push_back(cp);
}

uint32_t compose_pair(uint32_t a, uint32_t b) {
    if (a >= LBase && a < LBase + LCount && b >= VBase && b < VBase + VCount)
        return SBase + ((a - LBase) * VCount + (b - VBase)) * TCount;
    if (a >= SBase && a < SBase + SCount && (a - SBase) % TCount == 0 && b > TBase && b < TBase + TCount)
        return a + (b - TBase);
    auto it = std::lower_bound(std::begin(kComp), std::end(kComp), std::make_pair(a, b),
                               [](const UComp& x, const std::pair<uint32_t, uint32_t>& v) {
                                   return x.a < v.first || (x.a == v.first && x.b < v.second);
                               });
    return (it != std::end(kComp) && it->a == a && it->b == b) ? it->c : 0;
}

}  // namespace

std::string Tokenizer::nfc(const std::string& s) {
    bool ascii = true;
    for (unsigned char c : s)
        if (c >= 0x80) { ascii = false; break; }
    if (ascii) return s;

    std::vector<uint32_t> d;
    for (uint32_t cp : utf8_decode(s)) decompose(cp, d);
    // canonical ordering: stable sort runs of non-starters by combining class
    for (size_t i = 0; i < d.size();) {
        if (ccc_of(d[i]) == 0) { i++; continue; }
        size_t j = i;
        while (j < d.size() && ccc_of(d[j]) != 0) j++;
        std::stable_sort(d.begin() + i, d.begin() + j, [](uint32_t a, uint32_t b) { return ccc_of(a) < ccc_of(b); });
        i = j;
    }
    // canonical composition
    std::vector<uint32_t> out;
    int starter = -1;
    int last_ccc = -1;
    for (uint32_t cp : d) {
        int cc = ccc_of(cp);
        if (starter >= 0) {
            bool blocked = last_ccc != -1 && (last_ccc == 0 || last_ccc >= cc);
            if (!blocked) {
                if (uint32_t c = compose_pair(out[starter], cp)) { out[starter] = c; continue; }
            }
        }
        if (cc == 0) { starter = (int)out.size(); last_ccc = -1; }
        else last_ccc = cc;
        out.push_back(cp);
    }
    std::string r;
    for (uint32_t cp : out) utf8_append(r, cp);
    return r;
}

// Hand-written equivalent of the Qwen pre-tokenizer regex:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//   |\s*[\r\n]+|\s+(?!\S)|\s+
std::vector<std::string> Tokenizer::pretokenize(const std::string& text) {
    std::vector<uint32_t> cps = utf8_decode(text);
    const size_t n = cps.size();
    auto LM = [&](size_t i) { return i < n && (is_L(cps[i]) || is_M(cps[i])); };
    auto other = [&](size_t i) { return i < n && !is_S(cps[i]) && !is_L(cps[i]) && !is_M(cps[i]) && !is_N(cps[i]); };
    auto lower = [&](size_t i) { return i < n && cps[i] < 0x80 ? (uint32_t)(cps[i] | 0x20) : 0u; };

    std::vector<std::string> pieces;
    size_t i = 0;
    while (i < n) {
        size_t end = 0;
        // 1. contractions
        if (cps[i] == '\'') {
            uint32_t a = lower(i + 1), b = lower(i + 2);
            if (a == 's' || a == 't' || a == 'm' || a == 'd') end = i + 2;
            else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) end = i + 3;
        }
        // 2. optional non-letter/number prefix + letters/marks
        if (!end) {
            uint32_t c = cps[i];
            bool ok = true;
            size_t j = i;
            if (!is_nl(c) && !is_L(c) && !is_N(c) && LM(i + 1)) j = i + 1;
            else ok = LM(i);
            if (ok) {
                while (LM(j)) j++;
                end = j;
            }
        }
        // 3. single number
        if (!end && is_N(cps[i])) end = i + 1;
        // 4. optional space + run of "other" + trailing newlines
        if (!end) {
            bool ok = true;
            size_t j = i;
            if (cps[i] == ' ' && other(i + 1)) j = i + 1;
            else ok = other(i);
            if (ok) {
                while (other(j)) j++;
                while (j < n && is_nl(cps[j])) j++;
                end = j;
            }
        }
        if (!end && is_S(cps[i])) {
            size_t j = i;
            while (j < n && is_S(cps[j])) j++;
            // 5. \s*[\r\n]+  -> up to and including the last newline in the run
            size_t last_nl = SIZE_MAX;
            for (size_t k = i; k < j; k++)
                if (is_nl(cps[k])) last_nl = k;
            if (last_nl != SIZE_MAX) end = last_nl + 1;
            // 6. \s+(?!\S)
            else if (j == n) end = j;
            else if (j - 1 > i) end = j - 1;
            // 7. \s+
            else end = j;
        }
        if (!end) end = i + 1;
        std::string p;
        for (size_t k = i; k < end; k++) utf8_append(p, cps[k]);
        pieces.push_back(std::move(p));
        i = end;
    }
    return pieces;
}

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void Tokenizer::load(const std::string& dir) {
    // GPT-2 byte <-> unicode mapping
    std::vector<uint32_t> unicode_to_byte(512, 0xFFFF);
    int extra = 0;
    for (int b = 0; b < 256; b++) {
        bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        uint32_t cp = printable ? (uint32_t)b : 256u + extra++;
        byte_to_unicode_[b].clear();
        utf8_append(byte_to_unicode_[b], cp);
        unicode_to_byte[cp] = (uint32_t)b;
    }

    json::Value t = json::parse(read_file(dir + "/tokenizer.json"));
    const json::Value& model = t["model"];
    for (auto& [tok, id] : model["vocab"].obj) {
        int i = (int)id.as_int();
        vocab_[tok] = i;
        if ((int)id_to_token_.size() <= i) id_to_token_.resize(i + 1);
        std::string bytes;
        for (uint32_t cp : utf8_decode(tok)) bytes += (char)(cp < 512 && unicode_to_byte[cp] != 0xFFFF ? unicode_to_byte[cp] : '?');
        id_to_token_[i] = bytes;
    }
    int rank = 0;
    for (auto& m : model["merges"].arr) {
        std::string a, b;
        if (m.type == json::Value::String) {
            size_t sp = m.as_str().find(' ');
            a = m.as_str().substr(0, sp);
            b = m.as_str().substr(sp + 1);
        } else {
            a = m[0].as_str();
            b = m[1].as_str();
        }
        auto ia = vocab_.find(a), ib = vocab_.find(b), ic = vocab_.find(a + b);
        if (ia != vocab_.end() && ib != vocab_.end() && ic != vocab_.end())
            merges_[((uint64_t)ia->second << 32) | (uint32_t)ib->second] = {rank, ic->second};
        rank++;
    }
    for (auto& a : t["added_tokens"].arr) {
        int id = (int)a["id"].as_int();
        const std::string& s = a["content"].as_str();
        added_.push_back({s, id});
        if ((int)id_to_token_.size() <= id) id_to_token_.resize(id + 1);
        id_to_token_[id] = s;
        vocab_[s] = id;
    }
    std::sort(added_.begin(), added_.end(), [](auto& x, auto& y) { return x.first.size() > y.first.size(); });
}

int Tokenizer::token_id(const std::string& s) const {
    auto it = vocab_.find(s);
    return it == vocab_.end() ? -1 : it->second;
}

std::string Tokenizer::decode(int id) const {
    return (id >= 0 && id < (int)id_to_token_.size()) ? id_to_token_[id] : std::string();
}

void Tokenizer::bpe(const std::string& piece, std::vector<int>& out) const {
    auto cached = cache_.find(piece);
    if (cached != cache_.end()) {
        out.insert(out.end(), cached->second.begin(), cached->second.end());
        return;
    }
    std::vector<int> sym;
    for (unsigned char c : piece) sym.push_back(vocab_.at(byte_to_unicode_[c]));
    for (;;) {
        int best_rank = INT_MAX, best_i = -1, best_id = -1;
        for (size_t i = 0; i + 1 < sym.size(); i++) {
            auto it = merges_.find(((uint64_t)sym[i] << 32) | (uint32_t)sym[i + 1]);
            if (it != merges_.end() && it->second.first < best_rank) {
                best_rank = it->second.first;
                best_i = (int)i;
                best_id = it->second.second;
            }
        }
        if (best_i < 0) break;
        sym[best_i] = best_id;
        sym.erase(sym.begin() + best_i + 1);
    }
    if (cache_.size() < 200000) cache_[piece] = sym;
    out.insert(out.end(), sym.begin(), sym.end());
}

std::vector<int> Tokenizer::encode(const std::string& text, bool parse_special) const {
    std::vector<int> ids;
    auto encode_plain = [&](const std::string& seg) {
        if (seg.empty()) return;
        for (const std::string& p : pretokenize(nfc(seg))) bpe(p, ids);
    };
    size_t start = 0, i = 0;
    while (i < text.size()) {
        int matched = -1;
        size_t mlen = 0;
        if (parse_special) {
            for (auto& [s, id] : added_)
                if (text.compare(i, s.size(), s) == 0) { matched = id; mlen = s.size(); break; }
        }
        if (matched >= 0) {
            encode_plain(text.substr(start, i - start));
            ids.push_back(matched);
            i += mlen;
            start = i;
        } else {
            i++;
        }
    }
    encode_plain(text.substr(start));
    return ids;
}
