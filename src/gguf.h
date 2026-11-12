// Memory-mapped GGUF reader (v2/v3, split files). Tensors are exposed as raw pointers into the mappings.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "safetensors.h"  // MappedFile

// ggml tensor types used by the GGUF quantizations we can run.
enum class GType : uint32_t {
    F32 = 0, F16 = 1, Q5_1 = 7, Q8_0 = 8, Q4_K = 12, Q5_K = 13, Q6_K = 14, IQ4_NL = 20, BF16 = 30, Unknown = 0xFFFF
};
const char* gtype_name(GType t);
// Weights per quantization block and bytes per block (1/element size for the float types, 0/0 if unsupported).
int gtype_block(GType t);
int gtype_block_bytes(GType t);

struct GTensor {
    GType type = GType::Unknown;
    std::vector<int64_t> dims;  // ggml order: dims[0] is the contiguous (row) dimension
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
};

// One metadata value. Arrays of numbers land in `nums`, arrays of strings in `strs`.
struct GValue {
    enum Kind { Num, Str, NumArr, StrArr } kind = Num;
    double num = 0;       // scalars (also bools); exact for |v| < 2^53
    uint64_t u64 = 0;     // scalars as an integer (exact)
    std::string str;
    std::vector<uint64_t> ints;   // integer arrays (exact)
    std::vector<double> nums;     // float arrays
    std::vector<std::string> strs;
};

class Gguf {
public:
    // `path` is a directory holding *.gguf shards or one .gguf file (its sibling shards are loaded too).
    explicit Gguf(const std::string& path);
    static bool is_gguf_path(const std::string& path);
    // Throws unless every shard of a split model was found (the metadata alone needs only the first one).
    void check_complete() const;

    const GValue* find_kv(const std::string& key) const;
    const GValue& kv(const std::string& key) const;
    uint64_t kv_u64(const std::string& key) const { return kv(key).u64; }
    double kv_num(const std::string& key, double def) const;
    const std::string& kv_str(const std::string& key) const { return kv(key).str; }

    const GTensor* find(const std::string& name) const;
    const GTensor& get(const std::string& name) const;
    size_t count() const { return tensors_.size(); }
    const std::string& dir() const { return dir_; }

private:
    void add_file(const std::string& path);
    std::string dir_;
    std::vector<std::unique_ptr<MappedFile>> files_;
    std::unordered_map<std::string, GValue> kv_;
    std::unordered_map<std::string, GTensor> tensors_;
};
