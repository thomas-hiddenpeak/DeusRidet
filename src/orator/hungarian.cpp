/**
 * @file hungarian.cpp
 * @philosophical_role Munkres assignment in plain C++. No external deps,
 *     deterministic, exact.
 * @serves OratorReclusterer (`orator_reclusterer.cpp`).
 *
 * @role Pure algorithmic primitive. Padded-square Kuhn-Munkres.
 *     Source: classical formulation, see e.g. Bourgeois & Lassalle 1971.
 */
#include "hungarian.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace deusridet::orator {

namespace {

constexpr double kInf = 1e18;

// Hungarian algorithm for a square N*N matrix with non-negative costs.
// Returns assignment[row] = col for each of the N rows.
std::vector<int> hungarian_square(std::vector<std::vector<double>> a) {
    const int n = static_cast<int>(a.size());
    // u, v are dual potentials; p[j] = row matched to column j (1-indexed),
    // way[j] traces augmenting path during column update.
    std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
    std::vector<int>    p(n + 1, 0),   way(n + 1, 0);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, kInf);
        std::vector<char>   used(n + 1, 0);
        do {
            used[j0] = 1;
            int    i0 = p[j0];
            int    j1 = -1;
            double delta = kInf;
            for (int j = 1; j <= n; ++j) {
                if (!used[j]) {
                    double cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j]  = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1    = j;
                    }
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j]    -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            int j1 = way[j0];
            p[j0]  = p[j1];
            j0     = j1;
        } while (j0 != 0);
    }

    std::vector<int> assignment(n, -1);
    for (int j = 1; j <= n; ++j) {
        if (p[j] >= 1 && p[j] <= n) {
            assignment[p[j] - 1] = j - 1;
        }
    }
    return assignment;
}

} // namespace

std::vector<int> solve_assignment(const std::vector<double>& cost,
                                  int N_rows,
                                  int N_cols) {
    if (N_rows <= 0 || N_cols <= 0) return std::vector<int>(std::max(0, N_rows), -1);

    const int N = std::max(N_rows, N_cols);

    // Find a finite ceiling to pad missing rows/cols with so the dummy
    // entries never out-compete a real one.
    double cmax = 0.0;
    for (int i = 0; i < N_rows; ++i) {
        for (int j = 0; j < N_cols; ++j) {
            const double c = cost[i * N_cols + j];
            if (c > cmax && c < kInf) cmax = c;
        }
    }
    const double pad = cmax * 10.0 + 1.0;

    std::vector<std::vector<double>> a(N, std::vector<double>(N, pad));
    for (int i = 0; i < N_rows; ++i) {
        for (int j = 0; j < N_cols; ++j) {
            a[i][j] = cost[i * N_cols + j];
        }
    }

    auto sq = hungarian_square(std::move(a));

    std::vector<int> out(N_rows, -1);
    for (int i = 0; i < N_rows; ++i) {
        const int j = sq[i];
        out[i] = (j >= 0 && j < N_cols) ? j : -1;
    }
    return out;
}

} // namespace deusridet::orator
