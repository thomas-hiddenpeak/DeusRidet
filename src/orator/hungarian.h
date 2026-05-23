/**
 * @file hungarian.h
 * @philosophical_role The principled assignment — when N local cluster
 *     centroids must be paired with M global speakers, fairness across the
 *     whole matrix beats greedy nearest neighbour. The Munkres algorithm
 *     guarantees the best total cost.
 * @serves OratorReclusterer's persistent-ID linking step.
 */
#pragma once

#include <vector>

namespace deusridet::orator {

// Solves a rectangular assignment problem.
//
// cost     : row-major N_rows * N_cols cost matrix. Lower cost = better match.
// N_rows   : number of rows.
// N_cols   : number of columns.
//
// Returns vector<int> of length max(N_rows, N_cols) giving the assigned
// column for each row (-1 if the row is unassigned, which happens when
// N_rows > N_cols and that row is the surplus one). Output length is
// always N_rows; entries beyond N_rows are not produced.
//
// Implementation: Kuhn-Munkres on a padded square matrix with O(N^3) time
// and O(N^2) memory. N here is max(N_rows, N_cols), which for the Orator
// reclusterer is bounded by cfg.max_k (≤ 8), so the absolute cost is
// negligible.
std::vector<int> solve_assignment(const std::vector<double>& cost,
                                  int N_rows,
                                  int N_cols);

} // namespace deusridet::orator
