// Process and system memory figures (GB), for /info.
#pragma once

struct MemoryStats {
    double working_set = 0;  // physical memory charged to this process right now
    double private_bytes = 0;  // committed private memory (includes pinned copies)
    double available = 0;    // system RAM available to other programs (free + standby)
    double total = 0;        // installed RAM
};

MemoryStats memory_stats();
