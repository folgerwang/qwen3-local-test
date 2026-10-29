// Minimal JSON parser (enough for config.json, tokenizer.json, safetensors headers).
#pragma once
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace json {

struct Value {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj;

    bool is_null() const { return type == Null; }
    const Value* find(std::string_view key) const {
        for (auto& kv : obj)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
    const Value& operator[](std::string_view key) const {
        const Value* v = find(key);
        if (!v) throw std::runtime_error("json: missing key " + std::string(key));
        return *v;
    }
    const Value& operator[](size_t i) const { return arr.at(i); }
    size_t size() const { return type == Array ? arr.size() : obj.size(); }
    int64_t as_int() const { return (int64_t)num; }
    double as_num() const { return num; }
    const std::string& as_str() const { return str; }
    // Convenience accessors with defaults.
    int64_t get_int(std::string_view key, int64_t def) const {
        const Value* v = find(key);
        return (v && v->type == Number) ? v->as_int() : def;
    }
    double get_num(std::string_view key, double def) const {
        const Value* v = find(key);
        return (v && v->type == Number) ? v->num : def;
    }
};

class Parser {
public:
    explicit Parser(std::string_view s) : s_(s) {}
    Value parse() {
        Value v = value();
        ws();
        if (p_ != s_.size()) fail("trailing characters");
        return v;
    }

private:
    std::string_view s_;
    size_t p_ = 0;

    [[noreturn]] void fail(const char* msg) {
        throw std::runtime_error(std::string("json: ") + msg + " at offset " + std::to_string(p_));
    }
    void ws() {
        while (p_ < s_.size() && (s_[p_] == ' ' || s_[p_] == '\n' || s_[p_] == '\r' || s_[p_] == '\t')) p_++;
    }
    char peek() { return p_ < s_.size() ? s_[p_] : '\0'; }
    void expect(char c) {
        if (peek() != c) fail("unexpected character");
        p_++;
    }
    bool lit(std::string_view w) {
        if (s_.substr(p_, w.size()) == w) { p_ += w.size(); return true; }
        return false;
    }

    Value value() {
        ws();
        Value v;
        char c = peek();
        if (c == '{') {
            v.type = Value::Object;
            p_++;
            ws();
            if (peek() == '}') { p_++; return v; }
            for (;;) {
                ws();
                std::string k = string();
                ws();
                expect(':');
                v.obj.emplace_back(std::move(k), value());
                ws();
                if (peek() == ',') { p_++; continue; }
                expect('}');
                return v;
            }
        }
        if (c == '[') {
            v.type = Value::Array;
            p_++;
            ws();
            if (peek() == ']') { p_++; return v; }
            for (;;) {
                v.arr.push_back(value());
                ws();
                if (peek() == ',') { p_++; continue; }
                expect(']');
                return v;
            }
        }
        if (c == '"') { v.type = Value::String; v.str = string(); return v; }
        if (lit("true")) { v.type = Value::Bool; v.b = true; return v; }
        if (lit("false")) { v.type = Value::Bool; v.b = false; return v; }
        if (lit("null")) return v;
        if (c == '-' || (c >= '0' && c <= '9')) {
            const char* begin = s_.data() + p_;
            char* end = nullptr;
            v.type = Value::Number;
            v.num = std::strtod(begin, &end);
            if (end == begin) fail("bad number");
            p_ += end - begin;
            return v;
        }
        fail("unexpected token");
    }

    static void put_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
        }
    }
    uint32_t hex4() {
        if (p_ + 4 > s_.size()) fail("bad \\u escape");
        uint32_t v = 0;
        for (int i = 0; i < 4; i++) {
            char c = s_[p_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= c - '0';
            else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
            else fail("bad hex digit");
        }
        return v;
    }
    std::string string() {
        expect('"');
        std::string out;
        for (;;) {
            if (p_ >= s_.size()) fail("unterminated string");
            char c = s_[p_++];
            if (c == '"') return out;
            if (c != '\\') { out += c; continue; }
            char e = s_[p_++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && s_.substr(p_, 2) == "\\u") {
                        p_ += 2;
                        uint32_t lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }
};

inline Value parse(std::string_view s) { return Parser(s).parse(); }

}  // namespace json
