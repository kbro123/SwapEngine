#pragma once
// AadBlock -- prices the NON-W-cacheable instruments of a bundle (FX/MtM, or a Portfolio that contains
// one) via the templated / forward-AAD path, so a HYBRID engine can keep the cacheable majority on the
// W-cache instead of dropping the whole bundle to AAD.
//
// This is the higher-cost path, so it is optimised for the fact that only the QUOTES change tick to tick
// -- the instruments, curves and knot structure are fixed at construction:
//   * WIDTH REDUCTION (the dominant saving). The Dual is Eigen::AutoDiffScalar<VectorXd> -- a DYNAMIC
//     gradient whose width, and every per-op allocation, scales with the number of seeded knots. So we
//     seed AAD only over the knots these instruments actually TOUCH (their curves + spread ancestors).
//     A couple of FX trades touching two curves differentiate w.r.t. ~20 knots, not the bundle's 100 --
//     the Dual pass and its allocations shrink proportionally, and the block Jacobian is scattered back
//     to those knots' columns.
//   * BUFFER REUSE. The seed vector's gradient basis is CONSTANT (it is the identity over the touched
//     knots), so it is built ONCE; each call only overwrites the seed VALUES. The sub-problem, the
//     touched-column map and the output targets are all sized once. (The per-operation gradient
//     allocations inside Eigen's AutoDiffScalar remain -- eliminating those needs a pooled Dual type,
//     a separate future change.)

#include <Eigen/Core>

#include <algorithm>
#include <set>
#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/problem.hpp"

namespace swaps::calibration {

class AadBlock {
 public:
  AadBlock() = default;
  bool empty() const { return rows_.empty(); }
  int size() const { return static_cast<int>(rows_.size()); }
  const std::vector<int>& rows() const { return rows_; }

  // Build from the bundle's curves, the non-cacheable instruments, their GLOBAL residual rows, and the
  // global knot count. Determines the touched knots and pre-sizes the reusable width-reduced seed.
  void init(const std::vector<BundleCurveSpec>& curves, std::vector<Instrument> instruments,
            std::vector<int> rows, int n_knots) {
    n_knots_ = n_knots;
    rows_ = std::move(rows);
    sub_.curves = curves;
    sub_.instruments = std::move(instruments);
    if (rows_.empty()) return;

    // Touched knots = every knot of every curve any sub-instrument references, closed under spread
    // ancestry (a spread curve's DFs also depend on its base's knots).
    std::set<int> curves_used;
    for (const auto& ins : sub_.instruments) collect_curves(ins, curves_used);
    std::set<int> closed;
    for (int c : curves_used) {
      for (int a = c; a >= 0; a = sub_.curves[a].base) closed.insert(a);
    }
    std::set<int> knots;
    for (int c : closed) {
      const int off = sub_.offset(c), nk = sub_.curves[c].n_knots();
      for (int i = 0; i < nk; ++i) knots.insert(off + i);
    }
    touched_.assign(knots.begin(), knots.end());  // sorted

    // Build the width-reduced seed ONCE: gradient e_j on touched_[j], zero elsewhere. Only values change.
    const int w = static_cast<int>(touched_.size());
    xd_.resize(n_knots_);
    for (int k = 0; k < n_knots_; ++k) xd_[k].derivatives() = Eigen::VectorXd::Zero(w);
    for (int j = 0; j < w; ++j) xd_[touched_[j]].derivatives()[j] = 1.0;
  }

  // Model quotes of the block's instruments (doubles), written to out[global_row]. (Used only to fill the
  // uniform model_rates vector; a bundle with a non-cacheable instrument does not stream, so these rows
  // are never the driver of a frozen-Newton reprice.)
  void model_rates_into(const Eigen::VectorXd& x, Eigen::VectorXd& out) const {
    if (rows_.empty()) return;
    const auto C = build_bundle_curves<double>(sub_.curves, [&](int c, int i) { return x[sub_.offset(c) + i]; });
    const auto curve_of = [&C](int i) -> const CurveHandle<double>& { return *C[i]; };
    for (int j = 0; j < size(); ++j)
      out[rows_[j]] = instrument_model_quote<double>(sub_.instruments[j], curve_of);
  }

  // True residuals (doubles) of the block's instruments against their stored markets, into out[global_row].
  void residuals_into(const Eigen::VectorXd& x, Eigen::VectorXd& out) const {
    if (rows_.empty()) return;
    const Eigen::VectorXd r = sub_.residuals<double>(x);
    for (int j = 0; j < size(); ++j) out[rows_[j]] = r[j];
  }

  // Jacobian rows d(residual)/dx via WIDTH-REDUCED AAD, into J.row(global_row) of an (n_res x n_knots) J.
  // Only the touched columns are nonzero; the rest stay whatever the caller pre-zeroed.
  void jacobian_into(const Eigen::VectorXd& x, Eigen::MatrixXd& J) const {
    if (rows_.empty()) return;
    for (int k = 0; k < n_knots_; ++k) xd_[k].value() = x[k];   // reuse the seed: values only
    const auto rd = sub_.residuals<ad::Dual>(xd_);
    const int w = static_cast<int>(touched_.size());
    for (int j = 0; j < size(); ++j) {
      J.row(rows_[j]).setZero();
      const auto& g = rd[j].derivatives();
      if (g.size() == w)
        for (int t = 0; t < w; ++t) J(rows_[j], touched_[t]) = g[t];
    }
  }

  // --- Streaming (frozen-Newton) forms: residual/Jacobian against a LIVE market q instead of the stored
  // mids, so a mixed FX/MtM bundle streams on the SAME hybrid engine (cacheable rows W-cache, these rows
  // AAD) rather than recalibrating each tick. residuals_vs_into runs every tick (cheap doubles);
  // jacobian_vs_into runs only on a Jacobian REFRESH (staleness), so the AAD sweep is off the hot path.

  // True residuals against the live market q, into out[global_row]: instrument_residual with q as the
  // target (FX gets ln F_model − ln q[row]; a banded row gets w(q_model)·(q_model − q[row])).
  void residuals_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::VectorXd& out) const {
    if (rows_.empty()) return;
    const auto C = build_bundle_curves<double>(sub_.curves, [&](int c, int i) { return x[sub_.offset(c) + i]; });
    const auto curve_of = [&C](int i) -> const CurveHandle<double>& { return *C[i]; };
    for (int j = 0; j < size(); ++j)
      out[rows_[j]] = instrument_residual<double>(sub_.instruments[j], curve_of, q[rows_[j]]);
  }

  // Jacobian of residuals_vs_into via WIDTH-REDUCED AAD, into the touched columns of J.row(global_row).
  // Consistent with residuals_vs_into by construction: AAD differentiates the SAME instrument_residual
  // (so the band chain-rule term (q_model − q) and the FX 1/F_model factor fall out automatically).
  void jacobian_vs_into(const Eigen::VectorXd& x, const Eigen::VectorXd& q, Eigen::MatrixXd& J) const {
    if (rows_.empty()) return;
    for (int k = 0; k < n_knots_; ++k) xd_[k].value() = x[k];   // reuse the seed: values only
    const auto C = build_bundle_curves<ad::Dual>(sub_.curves, [&](int c, int i) { return xd_[sub_.offset(c) + i]; });
    const auto curve_of = [&C](int i) -> const CurveHandle<ad::Dual>& { return *C[i]; };
    const int w = static_cast<int>(touched_.size());
    for (int j = 0; j < size(); ++j) {
      const ad::Dual rj = instrument_residual<ad::Dual>(sub_.instruments[j], curve_of, q[rows_[j]]);
      J.row(rows_[j]).setZero();
      const auto& g = rj.derivatives();
      if (g.size() == w)
        for (int t = 0; t < w; ++t) J(rows_[j], touched_[t]) = g[t];
    }
  }

 private:
  // Curve roles an instrument reads (recursing through Portfolio components).
  static void collect_curves(const Instrument& ins, std::set<int>& s) {
    auto leg = [&](const FloatLeg& l) {
      s.insert(l.forecast); s.insert(l.discount);
      if (l.reset_num >= 0) s.insert(l.reset_num);
      if (l.reset_den >= 0) s.insert(l.reset_den);
    };
    switch (ins.quote) {
      case QuoteKind::Rate: s.insert(ins.forecast); break;
      case QuoteKind::FxForward:
        if (ins.fx_num >= 0) s.insert(ins.fx_num);
        if (ins.fx_den >= 0) s.insert(ins.fx_den);
        break;
      case QuoteKind::Portfolio:
        for (const auto& c : ins.combination) collect_curves(c.instrument, s);
        break;
      default:  // ParRate / ParSpread / XccyMtmBasis
        leg(ins.fwd); leg(ins.bench); leg(ins.mtm); s.insert(ins.fixed.discount);
        break;
    }
  }

  BundleProblem sub_;          // the non-cacheable instruments over the SAME curves (built once)
  std::vector<int> rows_;      // block index -> global residual row
  std::vector<int> touched_;   // global knot indices these instruments differentiate w.r.t. (sorted)
  int n_knots_ = 0;
  mutable Eigen::Matrix<ad::Dual, Eigen::Dynamic, 1> xd_;  // reused width-reduced seed (values updated)
};

}  // namespace swaps::calibration
