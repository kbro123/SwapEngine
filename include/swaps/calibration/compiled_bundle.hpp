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
#include "swaps/calibration/problem.hpp"
#include "swaps/pricing/compiled_book.hpp"

namespace swaps::calibration {

// A single-curve CalibrationProblem IS a 1-curve bundle: one self-discounting outright curve, its
// futures/swaps all forecasting and discounting curve 0, no basis. This lets the single-curve
// CompiledResidual be a thin delegate to CompiledBundleResidual -- one compiled engine, not two.
inline BundleProblem single_curve_bundle(const CalibrationProblem& p) {
  BundleProblem b;
  b.curves.push_back({p.meeting_times, p.back_times, -1});
  for (const auto& a : p.avg_futs) b.avg_futs.push_back({0, a.sched, a.convexity, a.market_rate});
  for (const auto& c : p.comp_futs) b.comp_futs.push_back({0, c.sched, c.convexity, c.market_rate});
  for (const auto& s : p.swaps) b.swaps.push_back({0, 0, s.sched, s.market_rate});
  for (Instrument ins : p.instruments) {  // by value: force every role onto the single curve, so the
    ins.fwd.forecast = ins.fwd.discount = 0;  // compiled path cannot disagree with price_residuals(),
    ins.bench.forecast = ins.bench.discount = 0;  // which resolves every role to the one curve.
    ins.fixed.discount = 0;
    ins.forecast = 0;
    b.instruments.push_back(std::move(ins));
  }
  return b;
}

class CompiledBundleResidual {
 public:
  explicit CompiledBundleResidual(const BundleProblem& p)
      : n_avg_(static_cast<int>(p.avg_futs.size())),
        n_comp_(static_cast<int>(p.comp_futs.size())),
        n_swap_(static_cast<int>(p.swaps.size())),
        n_basis_(static_cast<int>(p.bases.size())),
        n_gen_(static_cast<int>(p.instruments.size())),
        market_(p.market()) {
    std::vector<pricing::CurveStructure> specs;
    specs.reserve(p.curves.size());
    for (const auto& c : p.curves) specs.push_back({c.meeting, c.back, c.base});
    cs_.init(specs);

    // Register instruments in residual order (avg, comp, swaps, bases), each on its role curves.
    for (const auto& a : p.avg_futs) avg_.add_future(cs_, a.forecast, a.sched, a.convexity);
    for (const auto& c : p.comp_futs) comp_.add_future(cs_, c.forecast, c.sched, c.convexity);
    for (const auto& s : p.swaps) {
      swap_float_.add(cs_, s.forecast, s.discount, s.sched);
      swap_fixed_.add(cs_, s.discount, s.sched);
    }
    for (const auto& b : p.bases) {
      basis_bench_.add(cs_, b.benchmark, b.discount, b.sched);
      basis_fwd_.add(cs_, b.forecast, b.discount, b.sched);
      basis_fixed_.add(cs_, b.discount, b.sched);
    }
    register_generic(p);

    cs_.finalize();  // builds W_all now that every (curve, time) is registered
    avg_.finalize();
    comp_.finalize();
    swap_float_.finalize();
    swap_fixed_.finalize();
    basis_bench_.finalize();
    basis_fwd_.finalize();
    basis_fixed_.finalize();
    gen_pos_.finalize();
    gen_neg_.finalize();
    gen_fixed_.finalize();
    gen_rate_.finalize();
  }

  int n_residuals() const { return n_avg_ + n_comp_ + n_swap_ + n_basis_ + n_gen_; }
  int n_times() const { return cs_.n_times(); }

  // Model rates in BundleProblem's residual order: averaged/compounded future rates, swap par rates,
  // basis spreads, then the generic instruments in insertion order.
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
      o += n_basis_;
    }
    // Generic block: batched by quote kind, then scattered back to the instrument's insertion-order row.
    if (!q_rows_.empty()) {
      const Eigen::VectorXd ann = gen_fixed_.annuity(DF);
      const Eigen::VectorXd v = ((gen_pos_.pv(DF) - gen_neg_.pv(DF)).array() / ann.array()).matrix();
      for (std::size_t j = 0; j < q_rows_.size(); ++j) out[q_rows_[j]] = v[static_cast<int>(j)];
    }
    if (!r_rows_.empty()) {
      const Eigen::VectorXd v = gen_rate_.rate(DF);
      for (std::size_t j = 0; j < r_rows_.size(); ++j) out[r_rows_[j]] = v[static_cast<int>(j)];
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

    // Futures rows ARE the batch's rate rows -- scatter straight into G at the segment offset.
    avg_.d_rate(DF, G, o);
    o += n_avg_;
    comp_.d_rate(DF, G, o);
    o += n_comp_;
    // Swap/basis rows are a QUOTIENT of a float PV and an annuity, so each leg's dPV/dDF is gathered
    // into a local block first and combined by the quotient rule.
    if (n_swap_) {
      const Eigen::VectorXd fpv = swap_float_.pv(DF), ann = swap_fixed_.annuity(DF);
      Eigen::MatrixXd dfpv = Eigen::MatrixXd::Zero(n_swap_, T), dann = Eigen::MatrixXd::Zero(n_swap_, T);
      swap_float_.d_pv(DF, dfpv, 0, 1.0);
      swap_fixed_.d_annuity(dann, 0);
      for (int j = 0; j < n_swap_; ++j)  // d(fpv/ann) = dfpv/ann - fpv·dann/ann²
        G.row(o + j) = dfpv.row(j) / ann[j] - fpv[j] * dann.row(j) / (ann[j] * ann[j]);
    }
    o += n_swap_;
    if (n_basis_) {
      const Eigen::VectorXd num = basis_bench_.pv(DF) - basis_fwd_.pv(DF);
      const Eigen::VectorXd ann = basis_fixed_.annuity(DF);
      Eigen::MatrixXd dnum = Eigen::MatrixXd::Zero(n_basis_, T), dann = Eigen::MatrixXd::Zero(n_basis_, T);
      basis_bench_.d_pv(DF, dnum, 0, 1.0);  // benchmark leg: +
      basis_fwd_.d_pv(DF, dnum, 0, -1.0);   // spread (fwd) leg: -
      basis_fixed_.d_annuity(dann, 0);
      for (int j = 0; j < n_basis_; ++j)
        G.row(o + j) = dnum.row(j) / ann[j] - num[j] * dann.row(j) / (ann[j] * ann[j]);
    }
    o += n_basis_;
    // Generic quotient rows: (pv_pos - pv_neg)/annuity, the ONE transform behind both ParRate (an
    // empty `neg` leg => pv_neg == 0) and ParSpread. Same quotient rule as the swap/basis rows above;
    // the only difference is that the batch row j lands on the instrument's own residual row.
    if (!q_rows_.empty()) {
      const int nq = static_cast<int>(q_rows_.size());
      const Eigen::VectorXd num = gen_pos_.pv(DF) - gen_neg_.pv(DF);
      const Eigen::VectorXd ann = gen_fixed_.annuity(DF);
      Eigen::MatrixXd dnum = Eigen::MatrixXd::Zero(nq, T), dann = Eigen::MatrixXd::Zero(nq, T);
      gen_pos_.d_pv(DF, dnum, 0, 1.0);
      gen_neg_.d_pv(DF, dnum, 0, -1.0);
      gen_fixed_.d_annuity(dann, 0);
      for (int j = 0; j < nq; ++j)
        G.row(q_rows_[j]) = dnum.row(j) / ann[j] - num[j] * dann.row(j) / (ann[j] * ann[j]);
    }
    // Generic `Rate` rows ARE the futures batch's rate rows (convexity is a constant -> zero row).
    if (!r_rows_.empty()) {
      const int nr = static_cast<int>(r_rows_.size());
      Eigen::MatrixXd dr = Eigen::MatrixXd::Zero(nr, T);
      gen_rate_.d_rate(DF, dr, 0);
      for (int j = 0; j < nr; ++j) G.row(r_rows_[j]) = dr.row(j);
    }
    return -((G * DF.asDiagonal()) * cs_.W());
  }

 private:
  // Register the generic instruments, preserving their insertion order in the RESIDUAL rows while
  // batching them by quote kind (a batch must be homogeneous, and d_pv/d_rate scatter into a
  // CONTIGUOUS row block). q_rows_/r_rows_ map a batch position back to its residual row, so a mixed
  // list of ParRate/ParSpread/Rate instruments still fills exactly the rows BundleProblem documents.
  //
  // ParRate and ParSpread share ONE pair of float batches: both are (pv_pos - pv_neg)/annuity, with
  // ParRate contributing an EMPTY `neg` leg (a leg with no coupons has no entries in R_cpn, so its
  // pv row is exactly 0.0 and its d_pv contributes nothing).
  void register_generic(const BundleProblem& p) {
    int row = n_avg_ + n_comp_ + n_swap_ + n_basis_;
    const std::vector<pricing::FloatCoupon> no_leg;
    for (const auto& ins : p.instruments) {
      if (ins.quote == QuoteKind::Rate) {
        gen_rate_.add_future(cs_, ins.forecast, ins.obs, ins.convexity);
        r_rows_.push_back(row++);
        continue;
      }
      const bool spread = (ins.quote == QuoteKind::ParSpread);
      const FloatLeg& pos = spread ? ins.bench : ins.fwd;  // ParSpread: +bench; ParRate: +fwd
      gen_pos_.add(cs_, pos.forecast, pos.discount, pos.coupons);
      if (spread)
        gen_neg_.add(cs_, ins.fwd.forecast, ins.fwd.discount, ins.fwd.coupons);  // ParSpread: -fwd
      else
        gen_neg_.add(cs_, 0, 0, no_leg);  // ParRate: nothing subtracted
      gen_fixed_.add(cs_, ins.fixed.discount, ins.fixed.coupons);
      q_rows_.push_back(row++);
    }
  }

  int n_avg_, n_comp_, n_swap_, n_basis_, n_gen_;
  pricing::CompiledCurveSet cs_;
  pricing::BundleFloatBatch avg_, comp_, swap_float_, basis_bench_, basis_fwd_;
  pricing::BundleFixedLegs swap_fixed_, basis_fixed_;
  // Generic block (design §3).
  pricing::BundleFloatBatch gen_pos_, gen_neg_, gen_rate_;
  pricing::BundleFixedLegs gen_fixed_;
  std::vector<int> q_rows_, r_rows_;  // batch position -> residual row
  Eigen::VectorXd market_;
};

}  // namespace swaps::calibration
