// Memory-mapped safetensors reader. Tensors are exposed as raw pointers into the mapping.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

enum class DType { F32, BF16, F16, F8_E4M3, I64, Unknown };

size_t dtype_size(DType t);

struct TensorView {
    DType dtype = DType::Unknown;
    std::vector<int64_t> shape;
    const uint8_t* data = nullptr;
    size_t nbytes = 0;
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

class MappedFile {
public:
    explicit MappedFile(const std::string& path);
    ~MappedFile();
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

private:
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
#ifdef _WIN32
    void* file_ = nullptr;
    void* mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// All *.safetensors files in a model directory, indexed by tensor name.
class SafeTensors {
public:
    explicit SafeTensors(const std::string& dir);
    const TensorView& get(const std::string& name) const;
    const TensorView* find(const std::string& name) const;
    size_t count() const { return tensors_.size(); }

private:
    void add_file(const std::string& path);
    std::vector<std::unique_ptr<MappedFile>> files_;
    std::unordered_map<std::string, TensorView> tensors_;
};
