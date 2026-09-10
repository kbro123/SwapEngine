#pragma once
// GenerateRisk: one portfolio, N curve bundles, ALL risk implied off the FIRST bundle's discount factors.
//
// The book is priced once on bundle[0]'s calibrated curve C0 (the anchor). Every other bundle is RE-LEVELED
// onto C0 — its quotes are set to the levels C0 implies, so it describes the SAME curve, not its own saved
// market — and then the book's curve risk is re-expressed in that bundle's instrument basis. So NPV and the
// parallel-curve DV01 are identical in every output (they are properties of C0 and the book), and only the
// per-instrument decomposition differs. There is only ever ONE curve, so the ladders are internally
// consistent by construction and their totals reconcile (no second curve, no DV01 leak).
//
// Rank completion (no smoothing): a bundle with more knots than quotes (e.g. a knot-only meeting) has a
// singular quote->knot map, which is what makes a naive transform blow up. Instead of a curvature penalty
// (which biases the real pillars), we complete the basis the way a desk would: the null direction(s) of the
// calibration Jacobian are self-quoted — pinned at their calibrated value — as SYNTHETIC pillars. That makes
// the operator well-posed, preserves the calibrated curve, and yields an interpretable risk row for each
// implied pillar instead of a ± blow-up. A fully-determined bundle has no null space, so it is untouched.
//
// This subsumes the pairwise "risk transform" (that is just N=2). QuantLib-free; lives in the api/ seam.

#include <string>
#include <vector>

#include <Eigen/Core>

#include "swaps/api/bundle_api.hpp"  // BundleSession, RegSpec, bundle_from_json/book_from_json

namespace swaps::api {

namespace cal = swaps::calibration;

// One bundle's risk, expressed in ITS OWN instruments, off the anchor curve.
struct ConsistentBundleRisk {
  Eigen::VectorXd ladder;             // dP/dq over this bundle's REAL calibration instruments (n_residuals)
  Eigen::VectorXd synthetic;          // dP/dq over the self-quoted null pillars (implied free knots), if any
  std::vector<int> synthetic_knot;    // the dominant knot index of each synthetic pillar (for labelling)
  double npv = 0.0;                   // book NPV on the anchor curve (identical across bundles)
  double pv01 = 0.0;                  // parallel-curve DV01 on the anchor curve (identical across bundles)
  double ladder_dv01 = 0.0;          // 1bp · Σ(ladder + synthetic) — reconciles to pv01 up to the basis gap
  int n_residuals = 0, n_synthetic = 0;
};

// The full result: the shared invariants plus one ConsistentBundleRisk per input bundle (in input order).
struct ConsistentRisk {
  double npv = 0.0, pv01 = 0.0;       // the invariants (from the anchor bundle) — the same for every row
  int n = 0;                          // number of positions
  std::vector<ConsistentBundleRisk> bundles;
};

// Price `book` on bundles[0]'s curve and re-express its risk in every bundle's instrument basis. `reg` is
// used only to make the RE-LEVELING CALIBRATION of an under-determined bundle well-posed (a light default is
// applied when reg is off); the RISK operator itself is rank-completed by self-quoting, not smoothed.
ConsistentRisk generate_risk(const swaps::portfolio::MultiCurveBook& book,
                             const std::vector<cal::BundleProblem>& bundles, const RegSpec& reg = {});

// JSON seam: request = {"book": {...}, "bundles": [ {curves,instruments}, ... ], "regularize": {...}? }.
// Returns {npv, pv01, n, bundles:[{ladder, synthetic, synthetic_knot, npv, pv01, ladder_dv01,
// n_residuals, n_synthetic}, ...]}. Parses via bundle_from_json/book_from_json.
std::string generate_risk_json(const std::string& request);

// The rank-completed risk operator applied to a curve gradient (exposed for its test): null directions of J
// at the engine's kRankThreshold are self-quoted as unit-pinned rows; returns the length-(n_res + n_null)
// ladder and each synthetic pillar's dominant knot in `syn_knot`.
Eigen::VectorXd null_completed_ladder(const Eigen::MatrixXd& J, const Eigen::VectorXd& g, std::vector<int>& syn_knot);

}  // namespace swaps::api
