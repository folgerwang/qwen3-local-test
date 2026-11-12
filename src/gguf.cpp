#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <stdexcept>

const char* gtype_name(GType t) {
    switch (t) {
        case GType::F32: return "F32";
        case GType::F16: return "F16";
        case GType::Q5_1: return "Q5_1";
        case GType::Q8_0: return "Q8_0";
        case GType::Q4_K: return "Q4_K";
        case GType::Q5_K: return "Q5_K";
        case GType::Q6_K: return "Q6_K";
        case GType::IQ4_NL: return "IQ4_NL";
        case GType::BF16: return "BF16";
        default: return "unsupported";
    }
}

int gtype_block(GType t) {
    switch (t) {
        case GType::F32: case GType::F16: case GType::BF16: return 1;
        case GType::Q5_1: case GType::Q8_0: case GType::IQ4_NL: return 32;
        case GType::Q4_K: case GType::Q5_K: case GType::Q6_K: return 256;
        default: return 0;
    }
}

int gtype_block_bytes(GType t) {
    switch (t) {
        case GType::F32: return 4;
        case GType::F16: case GType::BF16: return 2;
        case GType::Q5_1: return 24;    // d, m (f16), qh[4], qs[16]
        case GType::Q8_0: return 34;    // d (f16), qs[32]
        case GType::IQ4_NL: return 18;  // d (f16), qs[16]
        case GType::Q4_K: return 144;   // d, dmin (f16), scales[12], qs[128]
        case GType::Q5_K: return 176;   // d, dmin (f16), scales[12], qh[32], qs[128]
        case GType::Q6_K: return 210;   // ql[128], qh[64], scales[16], d (f16)
        default: return 0;
    }
}

namespace {

struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    template <class T>
    T rd() {
        if ((size_t)(end - p) < sizeof(T)) throw std::runtime_error("gguf: truncated header");
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = rd<uint64_t>();
        if ((uint64_t)(end - p) < n) throw std::runtime_error("gguf: truncated string");
        std::string s((const char*)p, (size_t)n);
        p += n;
        return s;
    }
};

// GGUF metadata value types.
enum : uint32_t { T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 };

bool is_float(uint32_t t) { return t == T_F32 || t == T_F64; }

void read_scalar(Reader& r, uint32_t t, double& num, uint64_t& u) {
    switch (t) {
        case T_U8: u = r.rd<uint8_t>(); num = (double)u; break;
        case T_I8: { int8_t v = r.rd<int8_t>(); u = (uint64_t)(int64_t)v; num = v; break; }
        case T_U16: u = r.rd<uint16_t>(); num = (double)u; break;
        case T_I16: { int16_t v = r.rd<int16_t>(); u = (uint64_t)(int64_t)v; num = v; break; }
        case T_U32: u = r.rd<uint32_t>(); num = (double)u; break;
        case T_I32: { int32_t v = r.rd<int32_t>(); u = (uint64_t)(int64_t)v; num = v; break; }
        case T_F32: num = r.rd<float>(); u = (uint64_t)num; break;
        case T_BOOL: u = r.rd<uint8_t>(); num = (double)u; break;
        case T_U64: u = r.rd<uint64_t>(); num = (double)u; break;
        case T_I64: { int64_t v = r.rd<int64_t>(); u = (uint64_t)v; num = (double)v; break; }
        case T_F64: num = r.rd<double>(); u = (uint64_t)num; break;
        default: throw std::runtime_error("gguf: bad metadata type");
    }
}

GValue read_value(Reader& r, uint32_t t) {
    GValue v;
    if (t == T_STR) {
        v.kind = GValue::Str;
        v.str = r.str();
    } else if (t == T_ARR) {
        const uint32_t et = r.rd<uint32_t>();
        const uint64_t n = r.rd<uint64_t>();
        if (et == T_STR) {
            v.kind = GValue::StrArr;
            v.strs.reserve((size_t)n);
            for (uint64_t i = 0; i < n; i++) v.strs.push_back(r.str());
        } else if (et == T_ARR) {
            throw std::runtime_error("gguf: nested arrays are not supported");
        } else {
            v.kind = GValue::NumArr;
            v.ints.resize((size_t)n);
            if (is_float(et)) v.nums.resize((size_t)n);
            for (uint64_t i = 0; i < n; i++) {
                double d;
                uint64_t u;
                read_scalar(r, et, d, u);
                v.ints[(size_t)i] = u;
                if (is_float(et)) v.nums[(size_t)i] = d;
            }
        }
    } else {
        read_scalar(r, t, v.num, v.u64);
    }
    return v;
}

}  // namespace

bool Gguf::is_gguf_path(const std::string& path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (fs::is_regular_file(path, ec)) return fs::path(path).extension() == ".gguf";
    if (!fs::is_directory(path, ec)) return false;
    for (auto& e : fs::directory_iterator(path, ec))
        if (e.path().extension() == ".gguf") return true;
    return false;
}

Gguf::Gguf(const std::string& path) {
    namespace fs = std::filesystem;
    fs::path dir = fs::is_directory(path) ? fs::path(path) : fs::path(path).parent_path();
    if (dir.empty()) dir = ".";
    dir_ = dir.string();
    std::vector<std::string> files;
    if (fs::is_directory(path)) {
        for (auto& e : fs::directory_iterator(dir))
            if (e.path().extension() == ".gguf") files.push_back(e.path().string());
    } else {
        // one shard given: "<base>-00001-of-00004.gguf" pulls in every "<base>-*-of-00004.gguf"
        const std::string name = fs::path(path).filename().string();
        const size_t of = name.rfind("-of-");
        if (of != std::string::npos && of >= 6 && name[of - 6] == '-') {
            const std::string base = name.substr(0, of - 5), tail = name.substr(of);
            for (auto& e : fs::directory_iterator(dir)) {
                const std::string n = e.path().filename().string();
                if (n.size() == name.size() && n.compare(0, base.size(), base) == 0 &&
                    n.compare(of, std::string::npos, tail) == 0)
                    files.push_back(e.path().string());
            }
        } else {
            files.push_back(path);
        }
    }
    if (files.empty()) throw std::runtime_error("no .gguf files in " + path);
    std::sort(files.begin(), files.end());
    for (auto& f : files) add_file(f);
}

void Gguf::check_complete() const {
    if (const GValue* n = find_kv("split.tensors.count"); n && n->u64 != tensors_.size())
        throw std::runtime_error("gguf: found " + std::to_string(tensors_.size()) + " of " + std::to_string(n->u64) +
                                 " tensors in " + dir_ + " (missing shards? run setup.bat)");
}

void Gguf::add_file(const std::string& path) {
    files_.push_back(std::make_unique<MappedFile>(path));
    const MappedFile& mf = *files_.back();
    Reader r{mf.data(), mf.data() + mf.size()};
    if (mf.size() < 24 || std::memcmp(r.p, "GGUF", 4) != 0) throw std::runtime_error("not a GGUF file: " + path);
    r.p += 4;
    const uint32_t version = r.rd<uint32_t>();
    if (version < 2 || version > 3) throw std::runtime_error("unsupported GGUF version in " + path);
    const uint64_t n_tensors = r.rd<uint64_t>(), n_kv = r.rd<uint64_t>();

    uint64_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        std::string key = r.str();
        const uint32_t t = r.rd<uint32_t>();
        GValue v = read_value(r, t);
        if (key == "general.alignment") alignment = v.u64;
        // shard 0 carries the model metadata; later shards only repeat the split.* keys
        kv_.emplace(std::move(key), std::move(v));
    }

    struct Info { std::string name; GTensor t; uint64_t off; };
    std::vector<Info> infos((size_t)n_tensors);
    for (auto& in : infos) {
        in.name = r.str();
        const uint32_t nd = r.rd<uint32_t>();
        if (nd > 4) throw std::runtime_error("gguf: bad tensor rank: " + in.name);
        for (uint32_t d = 0; d < nd; d++) in.t.dims.push_back((int64_t)r.rd<uint64_t>());
        const uint32_t ty = r.rd<uint32_t>();
        in.t.type = (GType)ty;
        if (gtype_block(in.t.type) == 0) in.t.type = GType::Unknown;
        in.off = r.rd<uint64_t>();
    }
    const uint64_t header = (uint64_t)(r.p - mf.data());
    const uint64_t base = (header + alignment - 1) / alignment * alignment;
    for (auto& in : infos) {
        int64_t numel = 1;
        for (auto d : in.t.dims) numel *= d;
        if (in.t.type != GType::Unknown) {
            const int blk = gtype_block(in.t.type);
            if (in.t.dims.empty() || in.t.dims[0] % blk) throw std::runtime_error("gguf: row not block aligned: " + in.name);
            in.t.nbytes = (size_t)(numel / blk) * gtype_block_bytes(in.t.type);
        }
        if (base + in.off + in.t.nbytes > mf.size())
            throw std::runtime_error("gguf: tensor " + in.name + " exceeds file " + path + " (incomplete download?)");
        in.t.data = mf.data() + base + in.off;
        tensors_[in.name] = std::move(in.t);
    }
}

const GValue* Gguf::find_kv(const std::string& key) const {
    auto it = kv_.find(key);
    return it == kv_.end() ? nullptr : &it->second;
}
const GValue& Gguf::kv(const std::string& key) const {
    const GValue* v = find_kv(key);
    if (!v) throw std::runtime_error("gguf: missing metadata key " + key);
    return *v;
}
double Gguf::kv_num(const std::string& key, double def) const {
    const GValue* v = find_kv(key);
    return v ? v->num : def;
}
const GTensor* Gguf::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}
const GTensor& Gguf::get(const std::string& name) const {
    const GTensor* t = find(name);
    if (!t) throw std::runtime_error("gguf: missing tensor " + name);
    return *t;
}
