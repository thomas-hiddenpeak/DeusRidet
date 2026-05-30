// @role: DiariZenClustering AHC + VBx EM + assignment (P2b-2..4). STUB — to be
//        implemented in the next increment; present so the --fea harness links.
#include "diarizen_clustering.h"

namespace deusridet {
namespace orator {

void DiarizenClustering::agglomerative_(const std::vector<double>&, int, int,
                                        std::vector<int>&) const {}

bool DiarizenClustering::debug_ahc(const float*, int, int, std::vector<int>&) {
    return false;  // TODO(P2b-2)
}

bool DiarizenClustering::cluster(const float*, int, int, int, const float*, int,
                                 std::vector<std::int8_t>&) {
    return false;  // TODO(P2b-4)
}

}  // namespace orator
}  // namespace deusridet
