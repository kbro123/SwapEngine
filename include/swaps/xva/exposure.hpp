#pragma once
// Counterparty-exposure profiles (EPE / ENE / PFE) from a simulated book-value grid — the aggregation layer
// on top of the MC-exposure kernel (portfolio/compiled.hpp CompiledPortfolio::npv_grid, which reprices the
// whole book at every simulated curve-state at ~2M states/sec). This header is MODEL-AGNOSTIC: it takes the
// per-swap NPV grid a caller already produced from its path model (LGM/HW1F, or a Gaussian curve-state proxy),
// nets it, and reduces each time-node's cross-path distribution to the profiles a desk quotes:
//
//   EPE(t) = E_paths[ max(V_net(t), 0) ]     (expected positive exposure — drives CVA)
//   ENE(t) = E_paths[ min(V_net(t), 0) ]     (expected negative exposure — drives DVA)
//   PFE_q(t) = q-quantile of V_net(t)         (potential future exposure, e.g. q=0.975 — the limit line)
//
// State layout: the grid has n_paths*n_nodes columns; column for (path p, node j) lives at j*n_paths + p, so
// all paths at a node are a contiguous block (node-major) — the same layout the benchmark/model fills.
#include <algorithm>
#include <cmath>
#include <vector>

#include <Eigen/Core>

namespace swaps::xva {

struct ExposureProfile {
  std::vector<double> node_time;  // t_j
  std::vector<double> epe;        // expected positive exposure per node
  std::vector<double> ene;        // expected negative exposure per node (<= 0)
  std::vector<double> pfe;        // q-quantile (potential future exposure) per node
};

// `npv_grid` is n_swaps x (n_paths*n_nodes) — the per-swap NPVs from CompiledPortfolio::npv_grid. Netting set =
// the whole book: sum the swaps to a book value per state. `pfe_q` in (0,1). Node-major column layout (above).
inline ExposureProfile exposure_profile(const Eigen::MatrixXd& npv_grid, int n_paths, int n_nodes,
                                        const std::vector<double>& node_time, double pfe_q = 0.975) {
  // Net the book: one value per simulated state (colwise sum over swaps).
  const Eigen::RowVectorXd v_net = npv_grid.colwise().sum();
  ExposureProfile pr;
  pr.node_time = node_time;
  pr.epe.resize(n_nodes);
  pr.ene.resize(n_nodes);
  pr.pfe.resize(n_nodes);
  std::vector<double> col(n_paths);  // reused per-node scratch for the quantile
  const std::size_t qidx = std::min<std::size_t>(n_paths - 1,
                                                 static_cast<std::size_t>(std::floor(pfe_q * (n_paths - 1))));
  for (int j = 0; j < n_nodes; ++j) {
    const int base = j * n_paths;
    double sp = 0.0, sn = 0.0;
    for (int p = 0; p < n_paths; ++p) {
      const double v = v_net[base + p];
      sp += v > 0.0 ? v : 0.0;
      sn += v < 0.0 ? v : 0.0;
      col[static_cast<std::size_t>(p)] = v;
    }
    pr.epe[j] = sp / n_paths;
    pr.ene[j] = sn / n_paths;
    std::nth_element(col.begin(), col.begin() + qidx, col.end());  // O(n) PFE quantile
    pr.pfe[j] = col[qidx];
  }
  return pr;
}

}  // namespace swaps::xva
