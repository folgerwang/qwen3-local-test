// Process and system memory figures (GB), for /info.
#pragma once

struct MemoryStats {
    double working_set = 0;  // physical memory charged to this process right now
    double private_bytes = 0;  // committed private memory (includes pinned copies)
    double available = 0;    // system RAM available to other programs (free + standby)
    double total = 0;        // installed RAM
};

MemoryStats memory_stats();

// Dedicated video memory in use by ALL programs on the adapter with this LUID (bytes), or -1 if unknown.
// (Vulkan's memory budget only reports this process's use; this is what tells us how full the card really is.)
long long gpu_dedicated_used(unsigned luid_low, long luid_high);
