#include "safetensors.h"

#include <cstring>
#include <filesystem>
#include <stdexcept>

#include "json.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

size_t dtype_size(DType t) {
    switch (t) {
        case DType::F32: return 4;
        case DType::BF16: return 2;
        case DType::F16: return 2;
        case DType::F8_E4M3: return 1;
        case DType::I64: return 8;
        default: return 0;
    }
}

static DType parse_dtype(const std::string& s) {
    if (s == "F32") return DType::F32;
    if (s == "BF16") return DType::BF16;
    if (s == "F16") return DType::F16;
    if (s == "F8_E4M3") return DType::F8_E4M3;
    if (s == "I64") return DType::I64;
    return DType::Unknown;
}

#ifdef _WIN32
MappedFile::MappedFile(const std::string& path) {
    std::wstring wpath = std::filesystem::path(path).wstring();
    HANDLE f = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path);
    LARGE_INTEGER sz;
    GetFileSizeEx(f, &sz);
    size_ = (size_t)sz.QuadPart;
    HANDLE m = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!m) { CloseHandle(f); throw std::runtime_error("cannot map " + path); }
    data_ = (const uint8_t*)MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!data_) { CloseHandle(m); CloseHandle(f); throw std::runtime_error("cannot map view " + path); }
    file_ = f;
    mapping_ = m;
}
MappedFile::~MappedFile() {
    if (data_) UnmapViewOfFile(data_);
    if (mapping_) CloseHandle((HANDLE)mapping_);
    if (file_) CloseHandle((HANDLE)file_);
}
#else
MappedFile::MappedFile(const std::string& path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open " + path);
    struct stat st;
    fstat(fd_, &st);
    size_ = (size_t)st.st_size;
    void* p = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) throw std::runtime_error("cannot mmap " + path);
    data_ = (const uint8_t*)p;
}
MappedFile::~MappedFile() {
    if (data_) munmap((void*)data_, size_);
    if (fd_ >= 0) close(fd_);
}
#endif

SafeTensors::SafeTensors(const std::string& dir) {
    for (auto& e : std::filesystem::directory_iterator(dir))
        if (e.path().extension() == ".safetensors") add_file(e.path().string());
    if (tensors_.empty()) throw std::runtime_error("no safetensors found in " + dir);
}

void SafeTensors::add_file(const std::string& path) {
    auto f = std::make_unique<MappedFile>(path);
    if (f->size() < 8) throw std::runtime_error("truncated " + path);
    uint64_t hlen;
    std::memcpy(&hlen, f->data(), 8);
    if (8 + hlen > f->size()) throw std::runtime_error("bad header " + path);
    json::Value h = json::parse(std::string_view((const char*)f->data() + 8, hlen));
    const uint8_t* base = f->data() + 8 + hlen;
    for (auto& [name, v] : h.obj) {
        if (name == "__metadata__") continue;
        TensorView t;
        t.dtype = parse_dtype(v["dtype"].as_str());
        for (auto& d : v["shape"].arr) t.shape.push_back(d.as_int());
        size_t b = (size_t)v["data_offsets"][0].as_int(), e = (size_t)v["data_offsets"][1].as_int();
        if (8 + hlen + e > f->size()) throw std::runtime_error("tensor out of range: " + name);
        t.data = base + b;
        t.nbytes = e - b;
        tensors_[name] = std::move(t);
    }
    files_.push_back(std::move(f));
}

void SafeTensors::release_pages() const {
    for (auto& f : files_) {
#ifdef _WIN32
        // VirtualUnlock on pages that are not locked takes them out of the working set: they stay in the file
        // cache (standby) and come back on the next access, but no longer count as this process's memory.
        VirtualUnlock((LPVOID)f->data(), f->size());
#else
        madvise((void*)f->data(), f->size(), MADV_DONTNEED);
#endif
    }
}

const TensorView* SafeTensors::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const TensorView& SafeTensors::get(const std::string& name) const {
    const TensorView* t = find(name);
    if (!t) throw std::runtime_error("missing tensor: " + name);
    return *t;
}
