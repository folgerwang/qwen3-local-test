#include "sysmem.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <cstdio>
#include <mutex>
#pragma comment(lib, "psapi.lib")
#pragma comment(lib, "pdh.lib")
#endif

long long gpu_dedicated_used(unsigned luid_low, long luid_high) {
#ifdef _WIN32
    // The counter Task Manager shows ("Dedicated GPU memory"), one instance per adapter LUID.
    static std::mutex mu;
    static PDH_HQUERY query = nullptr;
    static PDH_HCOUNTER counter = nullptr;
    static bool failed = false;
    std::lock_guard<std::mutex> lock(mu);
    if (failed) return -1;
    if (!query) {
        wchar_t path[160];
        std::swprintf(path, 160, L"\\GPU Adapter Memory(luid_0x%08X_0x%08X_phys_0)\\Dedicated Usage", (unsigned)luid_high, luid_low);
        if (PdhOpenQueryW(nullptr, 0, &query) != ERROR_SUCCESS || PdhAddEnglishCounterW(query, path, 0, &counter) != ERROR_SUCCESS) {
            failed = true;
            return -1;
        }
    }
    PDH_FMT_COUNTERVALUE v{};
    if (PdhCollectQueryData(query) != ERROR_SUCCESS ||
        PdhGetFormattedCounterValue(counter, PDH_FMT_LARGE, nullptr, &v) != ERROR_SUCCESS || v.CStatus != PDH_CSTATUS_VALID_DATA)
        return -1;
    return v.largeValue;
#else
    (void)luid_low;
    (void)luid_high;
    return -1;
#endif
}

MemoryStats memory_stats() {
    MemoryStats m;
#ifdef _WIN32
    constexpr double GB = 1073741824.0;
    PROCESS_MEMORY_COUNTERS_EX pc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pc, sizeof(pc))) {
        m.working_set = pc.WorkingSetSize / GB;
        m.private_bytes = pc.PrivateUsage / GB;
    }
    MEMORYSTATUSEX ms{sizeof(ms)};
    if (GlobalMemoryStatusEx(&ms)) {
        m.available = ms.ullAvailPhys / GB;
        m.total = ms.ullTotalPhys / GB;
    }
#endif
    return m;
}
