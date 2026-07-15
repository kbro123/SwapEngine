#pragma once
// Compiled (W-cache) residual for the multi-curve BundleProblem -- the bundle analogue of
// CompiledResidual, built on the multi-curve pricing/compiled_book.hpp engine. Produces model rates /
// residuals and the ANALYTIC block Jacobian (no AAD in the hot loop), so the warm & streaming
// re-calibrators drive the bundle on the microsecond fast path instead of an AAD sweep per refresh.
//
// Same factorization as the single-curve CompiledResidual: DF_all = exp(-W_all x) once, then the cheap
// per-type transform (futures DF-ratios, swap par rate = float_pv/annuity, basis = (pv_bench-pv_fwd)/
// annuity), and J = -(dr/dDF diag(DF)) W_all with dr/dDF the analytic per-instrument sensitivity
// scattered into the global DF columns. Residual order matches BundleProblem::residuals:
// avg futures, comp futures, swaps, bases.

#include <Eigen/Core>

#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/pricing/compiled_book.hpp"

namespace swaps::calibration {

class CompiledBundleResidual {
 public:
  explicit CompiledBundleResidual(const BundleProblem& p)
      : n_avg_(static_cast<int>(p.avg_futs.size())),
        n_comp_(static_cast<int>(p.comp_futs.size())),
        n_swap_(static_cast<int>(p.swaps.size())),
        n_basis_(static_cast<int>(p.bases.size())),
        market_(p.market()) {
    std::vector<pricing::CurveStructure> specs;
    specs.reserve(p.curves.size());
    for (const auto& c : p.curves) specs.push_back({c.meeting, c.back, c.base});
    cs_.init(specs);

    // Register instruments in residual order (avg, comp, swaps, bases), each on its role curves.
    for (const auto& a : p.avg_futs) avg_.add(cs_, a.forecast, a.sched, a.convexity);
    for (const auto& c : p.comp_futs) comp_.add(cs_, c.forecast, c.sched, c.convexity);
    for (const auto& s : p.swaps) {
      swap_float_.add(cs_, s.forecast, s.discount, s.sched);
      swap_fixed_.add(cs_, s.discount, s.sched);
    }
    for (const auto& b : p.bases) {
      basis_bench_.add(cs_, b.benchmark, b.discount, b.sched);
      basis_fwd_.add(cs_, b.forecast, b.discount, b.sched);
      basis_fixed_.add(cs_, b.discount, b.sched);
    }

    cs_.finalize();  // builds W_all now that every (curve, time) is registered
    avg_.finalize();
    comp_.finalize();
    swap_float_.finalize();
    swap_fixed_.finalize();
    basis_bench_.finalize();
    basis_fwd_.finalize();
    basis_fixed_.finalize();
  }

  int n_residuals() const { return n_avg_ + n_comp_ + n_swap_ + n_basis_; }
  int n_times() const { return cs_.n_times(); }

  // Model rates in residual order: averaged/compounded future rates, swap par rates, basis spreads.
  Eigen::VectorXd model_rates(const Eigen::VectorXd& x) const {
    const Eigen::VectorXd DF = cs_.df(x);
    Eigen::VectorXd out(n_residuals());
    int o = 0;
    if (n_avg_) out.segment(o, n_avg_) = avg_.rate(DF), o += n_avg_;
    if (n_comp_) out.segment(o, n_comp_) = comp_.rate(DF), o += n_comp_;
    if (n_swap_) {
      out.segment(o, n_swap_) = swap_float_.pv(DF).array() / swap_fixed_.annuity(DF).array();
      o += n_swap_;
    }
    if (n_basis_) {
      const Eigen::VectorXd ann = basis_fixed_.annuity(DF);
      out.segment(o, n_basis_) = (basis_bench_.pv(DF) - basis_fwd_.pv(DF)).array() / ann.array();
    }
    return out;
  }

  Eigen::VectorXd residuals(const Eigen::VectorXd& x) const { return model_rates(x) - market_; }

  // Analytic block Jacobian J = dr/dx = -(dr/dDF diag(DF)) W_all. dr/dDF (G) is the per-instrument
  // pricing sensitivity scattered into the global DF columns; the W_all matmul maps DF-space back to
  // the stacked knot space. Block structure (an instrument touches only its role curves' knots) falls
  // out automatically because those are the only nonzero W_all columns for its DF entries.
  Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const {
    const Eigen::VectorXd DF = cs_.df(x);
    const int N = n_residuals(), T = cs_.n_times();
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(N, T);
    int o = 0;

    for (int k = 0; k < static_cast<int>(avg_.subS.size()); ++k) {
      const int j = o + avg_.fut[k], sd = avg_.subS[k], se = avg_.subE[k];
      const double ip = avg_.inv_period[avg_.fut[k]];
      G(j, sd) += ip / DF[se];
      G(j, se) += -ip * DF[sd] / (DF[se] * DF[se]);
    }
    o += n_avg_;
    for (int j = 0; j < comp_.size(); ++j) {
      const int row = o + j, s = comp_.s[j], e = comp_.e[j];
      const double it = comp_.inv_tau[j];
      G(row, s) += it / DF[e];
      G(row, e) += -it * DF[s] / (DF[e] * DF[e]);
    }
    o += n_comp_;
    if (n_swap_) {
      const Eigen::VectorXd fpv = swap_float_.pv(DF), ann = swap_fixed_.annuity(DF);
      Eigen::MatrixXd dfpv = Eigen::MatrixXd::Zero(n_swap_, T), dann = Eigen::MatrixXd::Zero(n_swap_, T);
      scatter_float(swap_float_, DF, dfpv, 1.0);
      scatter_fixed(swap_fixed_, dann);
      for (int j = 0; j < n_swap_; ++j)
        G.row(o + j) = dfpv.row(j) / ann[j] - fpv[j] * dann.row(j) / (ann[j] * ann[j]);
    }
    o += n_swap_;
    if (n_basis_) {
      const Eigen::VectorXd num = basis_bench_.pv(DF) - basis_fwd_.pv(DF);
      const Eigen::VectorXd ann = basis_fixed_.annuity(DF);
      Eigen::MatrixXd dnum = Eigen::MatrixXd::Zero(n_basis_, T), dann = Eigen::MatrixXd::Zero(n_basis_, T);
      scatter_float(basis_bench_, DF, dnum, 1.0);   // benchmark leg: +
      scatter_float(basis_fwd_, DF, dnum, -1.0);    // spread (fwd) leg: -
      scatter_fixed(basis_fixed_, dann);
      for (int j = 0; j < n_basis_; ++j)
        G.row(o + j) = dnum.row(j) / ann[j] - num[j] * dann.row(j) / (ann[j] * ann[j]);
    }
    return -((G * DF.asDiagonal()) * cs_.W());
  }

 private:
  // d(float_pv)/dDF for each coupon, scattered (with sign) into the owning instrument's row.
  static void scatter_float(const pricing::BundleFloatLegs& fl, const Eigen::VectorXd& DF,
                            Eigen::MatrixXd& d, double sign) {
    for (int i = 0; i < static_cast<int>(fl.pay.size()); ++i) {
      const int r = fl.inst[i], pay = fl.pay[i], aS = fl.accS[i], aE = fl.accE[i];
      d(r, pay) += sign * (DF[aS] / DF[aE] - 1.0);
      d(r, aS) += sign * (DF[pay] / DF[aE]);
      d(r, aE) += sign * (-DF[pay] * DF[aS] / (DF[aE] * DF[aE]));
    }
  }
  static void scatter_fixed(const pricing::BundleFixedLegs& fx, Eigen::MatrixXd& d) {
    for (int i = 0; i < static_cast<int>(fx.pay.size()); ++i)
      d(fx.inst[i], fx.pay[i]) += fx.tau[i];
  }

  int n_avg_, n_comp_, n_swap_, n_basis_;
  pricing::CompiledCurveSet cs_;
  pricing::BundleAvgFutures avg_;
  pricing::BundleCompFutures comp_;
  pricing::BundleFloatLegs swap_float_, basis_bench_, basis_fwd_;
  pricing::BundleFixedLegs swap_fixed_, basis_fixed_;
  Eigen::VectorXd market_;
};

}  // namespace swaps::calibration
