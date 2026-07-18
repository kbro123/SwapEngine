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

// A single-curve CalibrationProblem IS a 1-curve bundle: one self-discounting outright curve, every
// instrument forecasting and discounting curve 0. This lets the single-curve CompiledResidual be a
// thin delegate to CompiledBundleResidual -- one compiled engine, not two.
inline BundleProblem single_curve_bundle(const CalibrationProblem& p) {
  BundleProblem b;
  b.curves.push_back({p.meeting_times, p.back_times, -1});
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
      : n_gen_(static_cast<int>(p.instruments.size())), market_(p.market()) {
    std::vector<pricing::CurveStructure> specs;
    specs.reserve(p.curves.size());
    for (const auto& c : p.curves) specs.push_back({c.meeting, c.back, c.base});
    cs_.init(specs);

    register_generic(p);

    cs_.finalize();  // builds W_all now that every (curve, time) is registered
    gen_pos_.finalize();
    gen_neg_.finalize();
    gen_fixed_.finalize();
    gen_rate_.finalize();

    // Jacobian scratch buffers, sized ONCE here and reused (setZero) every call -- no per-iteration
    // allocation of the N×T / nq×T / nr×T dense matrices.
    const int T = cs_.n_times();
    G_.resize(n_gen_, T);
    dnum_.resize(static_cast<int>(q_rows_.size()), T);
    dann_.resize(static_cast<int>(q_rows_.size()), T);
    dr_.resize(static_cast<int>(r_rows_.size()), T);
  }

  int n_residuals() const { return n_gen_; }
  int n_times() const { return cs_.n_times(); }

  // Model rates in BundleProblem's residual order: the generic instruments in insertion order, batched
  // by quote kind internally then scattered back to each instrument's own row.
  Eigen::VectorXd model_rates(const Eigen::VectorXd& x) const {
    const Eigen::VectorXd& DF = df_at(x);
    Eigen::VectorXd out(n_residuals());
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
    const Eigen::VectorXd& DF = df_at(x);
    G_.setZero();  // reuse the scratch buffer (sized once in the ctor)
    Eigen::MatrixXd& G = G_;

    // Quotient rows: (pv_pos - pv_neg)/annuity, the ONE transform behind both ParRate (an empty `neg`
    // leg => pv_neg == 0) and ParSpread. Each batch row j lands on the instrument's own residual row.
    if (!q_rows_.empty()) {
      const int nq = static_cast<int>(q_rows_.size());
      // Compute each batch's per-coupon numerator (the sub-period gather + reduce) ONCE, then feed it
      // to BOTH the value pass (pv_from_num) and the derivative pass (d_pv_from_num) -- the gather no
      // longer runs a second time for the derivative.
      const Eigen::VectorXd num_pos = gen_pos_.num(DF);
      const Eigen::VectorXd num_neg = gen_neg_.num(DF);
      const Eigen::VectorXd num =
          gen_pos_.pv_from_num(num_pos, DF) - gen_neg_.pv_from_num(num_neg, DF);
      const Eigen::VectorXd ann = gen_fixed_.annuity(DF);
      dnum_.setZero();
      dann_.setZero();
      Eigen::MatrixXd& dnum = dnum_;
      Eigen::MatrixXd& dann = dann_;
      gen_pos_.d_pv_from_num(num_pos, DF, dnum, 0, 1.0);
      gen_neg_.d_pv_from_num(num_neg, DF, dnum, 0, -1.0);
      gen_fixed_.d_annuity(dann, 0);
      for (int j = 0; j < nq; ++j)  // d(num/ann) = dnum/ann - num·dann/ann²
        G.row(q_rows_[j]) = dnum.row(j) / ann[j] - num[j] * dann.row(j) / (ann[j] * ann[j]);
    }
    // `Rate` rows ARE the futures batch's rate rows (convexity is a constant -> zero row).
    if (!r_rows_.empty()) {
      const int nr = static_cast<int>(r_rows_.size());
      dr_.setZero();
      Eigen::MatrixXd& dr = dr_;
      gen_rate_.d_rate(DF, dr, 0);
      for (int j = 0; j < nr; ++j) G.row(r_rows_[j]) = dr.row(j);
    }
    return -((G * DF.asDiagonal()) * cs_.W());
  }

 private:
  // DF = exp(-W_all x), memoized on x. model_rates(x) and jacobian(x) are called at the SAME x within
  // an LM step (the accepted point), so they share ONE W*x + exp instead of recomputing it. The gate is
  // exact equality on x (short-circuit on size), so the returned DF is bit-identical to cs_.df(x).
  const Eigen::VectorXd& df_at(const Eigen::VectorXd& x) const {
    if (x.size() != df_x_.size() || (x.array() != df_x_.array()).any()) {
      df_ = cs_.df(x);
      df_x_ = x;
    }
    return df_;
  }

  // Register the generic instruments, preserving their insertion order in the RESIDUAL rows while
  // batching them by quote kind (a batch must be homogeneous, and d_pv/d_rate scatter into a
  // CONTIGUOUS row block). q_rows_/r_rows_ map a batch position back to its residual row, so a mixed
  // list of ParRate/ParSpread/Rate instruments still fills exactly the rows BundleProblem documents.
  //
  // ParRate and ParSpread share ONE pair of float batches: both are (pv_pos - pv_neg)/annuity, with
  // ParRate contributing an EMPTY `neg` leg (a leg with no coupons has no entries in R_cpn, so its
  // pv row is exactly 0.0 and its d_pv contributes nothing).
  void register_generic(const BundleProblem& p) {
    int row = 0;
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

  int n_gen_;
  pricing::CompiledCurveSet cs_;
  // The ONE generic block (design §3): ParRate/ParSpread share the pos/neg float pair + fixed annuity;
  // Rate futures use the rate batch.
  pricing::BundleFloatBatch gen_pos_, gen_neg_, gen_rate_;
  pricing::BundleFixedLegs gen_fixed_;
  std::vector<int> q_rows_, r_rows_;  // batch position -> residual row
  Eigen::VectorXd market_;
  // Mutable per-call scratch (② reused Jacobian buffers, ③ DF memo) -- state that only CACHES pure
  // functions of x, so const-ness of residuals()/jacobian() is preserved semantically.
  mutable Eigen::VectorXd df_, df_x_;
  mutable Eigen::MatrixXd G_, dnum_, dann_, dr_;
};

}  // namespace swaps::calibration
