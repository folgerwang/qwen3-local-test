#include "sysmem.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

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
