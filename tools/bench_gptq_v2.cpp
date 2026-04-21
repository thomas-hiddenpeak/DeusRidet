/**
 * @file bench_gptq_v2.cpp
 * @philosophical_role Developer instrument — exercises the v2 GPTQ kernel
 *         family in isolation against Marlin baselines. Not an Actus:
 *         pure kernel measurement, no entity-level behaviour.
 * @serves performance regression tracking for `src/machina/gptq_gemm_v2.{h,cu}`.
 */
#include "machina/gptq_gemm_v2.h"
#include "communis/tegra.h"

int main(int /*argc*/, char** /*argv*/) {
    deusridet::bench_gptq_v2_kernels();
    deusridet::tegra_cleanup();
    return 0;
}
