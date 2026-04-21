/**
 * @file tegra.h
 * @philosophical_role Declaration of the Tegra introspection surface. Keeps hardware-specific knowledge funneled through one header so that subsystems never grep sysfs directly.
 * @serves Vigilia, bench tools, awaken startup.
 */
// tegra.h — Tegra platform utilities
//
// Memory status, page cache management, and cleanup for Jetson iGPU.
// The NvMap CMA driver on L4T does not immediately return freed GPU pages
// to the system pool after cudaFree/cudaDeviceReset — drop_page_caches()
// triggers kernel reclaim.  tegra_cleanup() combines cudaDeviceReset +
// drop_page_caches for a clean exit.

#pragma once

#include <cstddef>

namespace deusridet {

// Read MemAvailable from /proc/meminfo (returns kB, 0 on failure).
size_t read_memavail_kb();

// Write "3" to /proc/sys/vm/drop_caches.
// Tries direct write first, falls back to passwordless sudo -n.
bool drop_page_caches();

// cudaDeviceReset + drop_page_caches, with MemAvailable delta report.
void tegra_cleanup();

} // namespace deusridet
