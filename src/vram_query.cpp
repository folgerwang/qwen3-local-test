// Free VRAM as the driver sees it, over all processes.
//
// Vulkan's memory budget (VK_EXT_memory_budget) only knows this process's share: with other programs holding
// VRAM it still reports most of the card as free, and allocating that much overcommits the card. NVML (part of
// the NVIDIA driver) reports what is really used. Returns 0 when it cannot tell (not Windows + NVIDIA, no
// nvml.dll, or no GPU with that name).
#include <cstdint>
#include <cstring>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

uint64_t qw_system_free_vram(const char* device_name) {
    HMODULE h = LoadLibraryA("nvml.dll");
    if (!h) return 0;
    struct Mem { unsigned long long total, free, used; };
    using Init = int (*)();
    using Count = int (*)(unsigned*);
    using Handle = int (*)(unsigned, void**);
    using Name = int (*)(void*, char*, unsigned);
    using Info = int (*)(void*, Mem*);
    const Init init = (Init)GetProcAddress(h, "nvmlInit_v2"), shutdown = (Init)GetProcAddress(h, "nvmlShutdown");
    const Count count = (Count)GetProcAddress(h, "nvmlDeviceGetCount_v2");
    const Handle handle = (Handle)GetProcAddress(h, "nvmlDeviceGetHandleByIndex_v2");
    const Name name = (Name)GetProcAddress(h, "nvmlDeviceGetName");
    const Info info = (Info)GetProcAddress(h, "nvmlDeviceGetMemoryInfo");
    uint64_t free = 0;
    if (init && shutdown && count && handle && name && info && init() == 0) {
        unsigned n = 0;
        if (count(&n) == 0)
            for (unsigned i = 0; i < n; i++) {
                void* dev = nullptr;
                char nm[96] = {};
                Mem m{};
                if (handle(i, &dev) || name(dev, nm, sizeof nm) || info(dev, &m)) continue;
                // the Vulkan name is the NVML name, sometimes with a prefix; one GPU needs no match
                if (n == 1 || std::strstr(device_name, nm) || std::strstr(nm, device_name)) {
                    free = m.free;
                    break;
                }
            }
        shutdown();
    }
    FreeLibrary(h);
    return free;
}
#else
uint64_t qw_system_free_vram(const char*) { return 0; }
#endif
