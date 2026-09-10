// Exposure-profile aggregation (swaps/xva/exposure.hpp): EPE/ENE/PFE from a book-value grid. Tests the pure
// aggregation math on synthetic per-swap NPV grids (no model needed) — the invariants a desk relies on.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Core>

#include "swaps/xva/exposure.hpp"

namespace xva = swaps::xva;

namespace {
// Build a node-major per-swap NPV grid: n_swaps rows, columns (path p, node j) at j*n_paths + p.
Eigen::MatrixXd grid(int n_swaps, int n_paths, int n_nodes, double seed) {
  Eigen::MatrixXd g(n_swaps, n_paths * n_nodes);
  for (int j = 0; j < n_nodes; ++j)
    for (int p = 0; p < n_paths; ++p)
      for (int s = 0; s < n_swaps; ++s)
        // node 0 deterministic (no path spread); later nodes fan out with a signed, path/swap-dependent value.
        g(s, j * n_paths + p) = 1.0e5 * (s + 1) * (0.5 - (j == 0 ? 0.5 : double((p * 7 + s * 3 + int(seed)) % 11) / 10.0)) * (1 + j);
  return g;
}
}  // namespace

TEST(ExposureProfile, InvariantsAndDeterministicToday) {
  const int nS = 3, nP = 200, nN = 40;
  const Eigen::MatrixXd g = grid(nS, nP, nN, 4.0);
  std::vector<double> t(nN);
  for (int j = 0; j < nN; ++j) t[j] = 0.5 * j;
  const xva::ExposureProfile pr = xva::exposure_profile(g, nP, nN, t, 0.975);

  const Eigen::RowVectorXd v = g.colwise().sum();  // netted book value per state
  for (int j = 0; j < nN; ++j) {
    // EPE >= 0, ENE <= 0.
    EXPECT_GE(pr.epe[j], 0.0) << "node " << j;
    EXPECT_LE(pr.ene[j], 0.0) << "node " << j;
    // Core identity: max(v,0)+min(v,0)=v  =>  epe + ene == mean_paths(v).
    double mean = 0.0, lo = 1e300, hi = -1e300;
    for (int p = 0; p < nP; ++p) {
      const double x = v[j * nP + p];
      mean += x;
      lo = std::min(lo, x);
      hi = std::max(hi, x);
    }
    mean /= nP;
    EXPECT_NEAR(pr.epe[j] + pr.ene[j], mean, 1e-6 * std::max(1.0, std::abs(mean))) << "node " << j;
    // PFE is the q-quantile of the POSITIVE exposure max(v, 0) with the lower-index convention: sort the nP
    // values, take index floor(q (nP - 1)). (E5.2 2026-09-10: the old `lo <= pfe <= hi` accepted the mean, the
    // median or any wrong quantile.)
    std::vector<double> pos(nP);
    for (int p = 0; p < nP; ++p) pos[p] = std::max(v[j * nP + p], 0.0);
    std::sort(pos.begin(), pos.end());
    const int qidx = static_cast<int>(std::floor(0.975 * (nP - 1)));
    EXPECT_DOUBLE_EQ(pr.pfe[j], pos[qidx]) << "node " << j;
    EXPECT_GE(pr.pfe[j], 0.0);
    EXPECT_LE(pr.pfe[j], std::max(hi, 0.0));
  }
  // Node 0 is deterministic (all paths equal) -> EPE(0)=max(V,0), ENE(0)=min(V,0), PFE(0)=V exactly.
  const double v0 = v[0];
  EXPECT_DOUBLE_EQ(pr.epe[0], std::max(v0, 0.0));
  EXPECT_DOUBLE_EQ(pr.ene[0], std::min(v0, 0.0));
  EXPECT_DOUBLE_EQ(pr.pfe[0], std::max(v0, 0.0));  // PFE is a quantile of the POSITIVE exposure (2026-09-10)
  // The profile fans out: a later node's PFE strictly exceeds the (deterministic) node-0 exposure.
  EXPECT_GT(pr.pfe[nN - 1], pr.pfe[0]);
}
