// tegra.cpp — Tegra platform utilities
//
// See tegra.h for interface documentation.

#include "tegra.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

namespace deusridet {

size_t read_memavail_kb() {
    size_t avail_kb = 0;
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "MemAvailable:", 13) == 0) {
            sscanf(line + 13, " %zu", &avail_kb);
            break;
        }
    }
    fclose(f);
    return avail_kb;
}

bool drop_page_caches() {
    // Attempt 1: direct write (works if running as root or sysctl configured)
    FILE* f = fopen("/proc/sys/vm/drop_caches", "w");
    if (f) {
        fprintf(f, "3\n");
        fclose(f);
        return true;
    }
    // Attempt 2: passwordless sudo (common on Tegra dev boards)
    int rc = system("sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null");
    return (rc == 0);
}

void tegra_cleanup() {
    size_t before_kb = read_memavail_kb();

    cudaDeviceReset();

    if (!drop_page_caches()) {
        fprintf(stderr, "[WARN] Cannot write /proc/sys/vm/drop_caches "
                        "(need root). CMA pages may not be reclaimed.\n");
    }

    size_t after_kb = read_memavail_kb();
    if (before_kb > 0 && after_kb > 0) {
        long delta_mb = ((long)after_kb - (long)before_kb) / 1024;
        fprintf(stderr, "[Tegra] MemAvailable: %zu MB → %zu MB (%+ld MB)\n",
                before_kb / 1024, after_kb / 1024, delta_mb);
    }
}

} // namespace deusridet
