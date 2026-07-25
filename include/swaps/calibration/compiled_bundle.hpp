#pragma once
// Compiled (W-cache) residual for the multi-curve BundleProblem -- the bundle analogue of
// CompiledResidual, built on the multi-curve pricing/compiled_book.hpp engine. Produces model rates /
// residuals and the ANALYTIC block Jacobian (no AAD in the hot loop), so the warm & streaming
// re-calibrators drive the bundle on the microsecond fast path instead of an AAD sweep per refresh.
//
// Same factorization as the single-curve CompiledResidual: DF_all = exp(-W_all x) once, then the cheap
// per-type transform (futures DF-ratios, swap par rate = float_pv/annuity, basis = (pv_bench-pv_fwd)/
// annuity), and J = -(dr/dDF diag(DF)) W_all with dr/dDF the analytic per-instrument sensitivity
// scattered into the global DF columns. Residual order matches BundleProblem::residuals: the generic
// instruments in insertion order (batched internally by quote kind, then scattered to each row).

#include <Eigen/Core>

#include <stdexcept>
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
    cs_.init(p.curves);  // p.curves ARE pricing::CurveStructure now (BundleCurveSpec is an alias), no copy

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
  // DF = exp(-W_all x) (memoized on x). Exposed for profiling / downstream analytics.
  const Eigen::VectorXd& discount_factors(const Eigen::VectorXd& x) const { return df_at(x); }

  // Model rates in BundleProblem's residual order: the generic instruments in insertion order, batched
  // by quote kind internally then scattered back to each instrument's own row.
  const Eigen::VectorXd& model_rates(const Eigen::VectorXd& x) const {
    const Eigen::VectorXd& DF = df_at(x);
    out_.setZero(n_residuals());  // ACCUMULATE: a portfolio row sums its components' weighted quotes; a
                                  // plain row has one entry with weight 1 (0 + 1·q == q, bit-identical).
    if (!q_rows_.empty()) {
      const Eigen::VectorXd& ann = gen_fixed_.annuity(DF);  // refs into DISTINCT batch objects,
      const Eigen::VectorXd& pp = gen_pos_.pv(DF);          // so all three are simultaneously live
      const Eigen::VectorXd& pn = gen_neg_.pv(DF);
      for (std::size_t j = 0; j < q_rows_.size(); ++j) {
        const int i = static_cast<int>(j);
        out_[q_rows_[j].row] += q_rows_[j].weight * (pp[i] - pn[i]) / ann[i];
      }
    }
    if (!r_rows_.empty()) {
      const Eigen::VectorXd& v = gen_rate_.rate(DF);
      for (std::size_t j = 0; j < r_rows_.size(); ++j)
        out_[r_rows_[j].row] += r_rows_[j].weight * v[static_cast<int>(j)];
    }
    // FX forward OUTRIGHT F = fx_spot · DF_num(T) / DF_den(T) (the model quote; its LOG-basis residual is
    // applied in residuals_vs). Just a DF ratio -- the two DFs were registered into W like any other.
    for (const auto& f : fx_rows_)
      out_[f.row] = f.fx_spot * DF[f.idx_num] / DF[f.idx_den];
    return out_;
  }

  const Eigen::VectorXd& residuals(const Eigen::VectorXd& x) const { return residuals_vs(x, market_); }

  // Residual against an ARBITRARY market q -- the live streaming feed, not the stored anchor market. With
  // no bands this is model_rates(x) - q; with a band it is the SOFT residual w(q_model)·(q_model - q).
  // This is what lets the frozen-Newton streamer solve the banded LEAST-SQUARES on the fast path (drive
  // this residual, not model_rates-q): at the soft minimum dx = J⁺·r -> 0 even though r != 0.
  const Eigen::VectorXd& residuals_vs(const Eigen::VectorXd& x, const Eigen::VectorXd& q) const {
    const Eigen::VectorXd& mr = model_rates(x);  // model_rates fills out_; res_ (a distinct member) holds r
    res_ = mr - q;
    for (const auto& b : band_)  // banded rows: r = w(q_model)·(q_model - q)
      res_[b.row] *= band_weight_d(mr[b.row], b.lower, b.upper, b.decay).first;
    // FX rows: the residual is the implied-basis discrepancy (ln F_model − ln q)/T in RATE units, NOT the
    // raw outright difference F − q. (FX rows are never banded, so this cleanly overwrites mr − q.)
    for (const auto& f : fx_rows_) {
      using std::log;
      res_[f.row] = (log(mr[f.row]) - log(q[f.row])) / f.fx_time;
    }
    return res_;
  }

  // Analytic block Jacobian J = dr/dx = -(dr/dDF diag(DF)) W_all. dr/dDF (G) is the per-instrument
  // pricing sensitivity scattered into the global DF columns; the W_all matmul maps DF-space back to
  // the stacked knot space. Block structure (an instrument touches only its role curves' knots) falls
  // out automatically because those are the only nonzero W_all columns for its DF entries.
  Eigen::MatrixXd jacobian(const Eigen::VectorXd& x) const { return jacobian_vs(x, market_); }

  // Jacobian of residuals_vs(x, q): the band chain-rule term (q_model - q) uses the SAME market q as the
  // residual, so a frozen-Newton streamer's M is consistent with the residual it drives.
  Eigen::MatrixXd jacobian_vs(const Eigen::VectorXd& x, const Eigen::VectorXd& q) const {
    const Eigen::VectorXd& DF = df_at(x);
    // Capture the model quotes for banded rows BEFORE the batch scratch below is overwritten.
    std::vector<double> qb;
    if (!band_.empty()) {
      const Eigen::VectorXd& mr = model_rates(x);
      qb.reserve(band_.size());
      for (const auto& b : band_) qb.push_back(mr[b.row]);
    }
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
      for (int j = 0; j < nq; ++j)  // d(num/ann) = dnum/ann - num·dann/ann²; accumulate (portfolio rows)
        G.row(q_rows_[j].row) +=
            q_rows_[j].weight * (dnum.row(j) / ann[j] - num[j] * dann.row(j) / (ann[j] * ann[j]));
    }
    // `Rate` rows ARE the futures batch's rate rows (convexity is a constant -> zero row).
    if (!r_rows_.empty()) {
      const int nr = static_cast<int>(r_rows_.size());
      dr_.setZero();
      Eigen::MatrixXd& dr = dr_;
      gen_rate_.d_rate(DF, dr, 0);
      for (int j = 0; j < nr; ++j) G.row(r_rows_[j].row) += r_rows_[j].weight * dr.row(j);
    }
    // FX rows: r = (ln F − ln q)/T with F = fx_spot·DF_num/DF_den. Only TWO dr/dDF entries per row:
    //   dr/dDF_num = +1/(DF_num·T),  dr/dDF_den = −1/(DF_den·T).
    // The -(G·diag(DF))·W matmul below then yields the constant (W_den − W_num)/T row (ln F is affine in
    // x). FX rows are disjoint from the batch/band rows, so G starts at zero here.
    for (const auto& f : fx_rows_) {
      G(f.row, f.idx_num) += 1.0 / (DF[f.idx_num] * f.fx_time);
      G(f.row, f.idx_den) += -1.0 / (DF[f.idx_den] * f.fx_time);
    }
    // Band chain rule: r = w(q)·(q-market) => dr/dx = (w + w'·(q-market))·dq/dx. Scale each banded row's
    // dr/dDF (G) by that scalar before the W matmul (the matmul is linear, so scaling commutes).
    for (std::size_t k = 0; k < band_.size(); ++k) {
      const Band& b = band_[k];
      const std::pair<double, double> wd = band_weight_d(qb[k], b.lower, b.upper, b.decay);
      G.row(b.row) *= (wd.first + wd.second * (qb[k] - q[b.row]));  // (q_model - q), q = the live market
    }
    return -((G * DF.asDiagonal()) * cs_.W());
  }

 private:
  // DF = exp(-W_all x), memoized on x. model_rates(x) and jacobian(x) are called at the SAME x within
  // an LM step (the accepted point), so they share ONE W*x + exp instead of recomputing it. The gate is
  // exact equality on x (short-circuit on size), so the returned DF is bit-identical to cs_.df(x).
  const Eigen::VectorXd& df_at(const Eigen::VectorXd& x) const {
    if (x.size() != df_x_.size() || (x.array() != df_x_.array()).any()) {
      cs_.df_into(x, df_);  // allocation-free recompute into the df_ scratch
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
    for (int row = 0; row < static_cast<int>(p.instruments.size()); ++row) {
      const Instrument& ins = p.instruments[row];
      // A bid/offer band re-weights this row's residual (r = w(q)·(q-market)); the weight is a per-row
      // scalar post-transform, so the row stays on the W-cache path. Captured here, applied below.
      if (ins.band_upper > ins.band_lower)
        band_.push_back({row, ins.band_lower, ins.band_upper, ins.band_decay});
      register_at(ins, row, 1.0);
    }
  }

  // Register one instrument's legs into the batches, targeting residual `row` with `weight`. A Portfolio
  // recurses -- each component registers onto the SAME row with the product of weights -- so a butterfly
  // of par swaps becomes three weighted batch entries summed into one row, fully on the W-cache path.
  // Only a genuinely non-W-cacheable LEAF (FX/MtM, here or nested in a portfolio) forces the AAD engine.
  void register_at(const Instrument& ins, int row, double weight) {
    static const std::vector<pricing::FloatCoupon> no_leg;
    if (ins.quote == QuoteKind::Portfolio) {
      for (const auto& comp : ins.combination) register_at(comp.instrument, row, weight * comp.weight);
      return;
    }
    if (ins.quote == QuoteKind::FxForward) {
      // FX forward F = fx_spot·DF_num(T)/DF_den(T). ln F is AFFINE in x (ln DF = -W·x), so the residual
      // (ln F − ln q)/T rides the W-cache: register the two DFs and remember them; dr/dDF is two entries.
      // Only STANDALONE FX (weight 1) -- a Σ of FX log-residuals inside a Portfolio isn't this transform.
      if (weight != 1.0)
        throw std::invalid_argument("CompiledBundleResidual: FX-forward inside a Portfolio is not W-cacheable; use the AAD engine");
      fx_rows_.push_back({row, cs_.reg(ins.fx_num, ins.fx_time), cs_.reg(ins.fx_den, ins.fx_time),
                          ins.fx_spot, ins.fx_time});
      return;
    }
    if (ins.quote == QuoteKind::XccyMtmBasis)
      throw std::invalid_argument(
          "CompiledBundleResidual: MtM-xccy quote (a curve-dependent FX-reset notional) is not yet "
          "W-cacheable; use the AAD engine");
    if (ins.quote == QuoteKind::Rate) {
      gen_rate_.add_future(cs_, ins.forecast, ins.obs, ins.convexity);
      r_rows_.push_back({row, weight});
      return;
    }
    const bool spread = (ins.quote == QuoteKind::ParSpread);
    const FloatLeg& pos = spread ? ins.bench : ins.fwd;  // ParSpread: +bench; ParRate: +fwd
    gen_pos_.add(cs_, pos.forecast, pos.discount, pos.coupons);
    if (spread)
      gen_neg_.add(cs_, ins.fwd.forecast, ins.fwd.discount, ins.fwd.coupons);  // ParSpread: -fwd
    else
      gen_neg_.add(cs_, 0, 0, no_leg);  // ParRate: nothing subtracted
    gen_fixed_.add(cs_, ins.fixed.discount, ins.fixed.coupons);
    q_rows_.push_back({row, weight});
  }

  int n_gen_;
  pricing::CompiledCurveSet cs_;
  // The ONE generic block (design §3): ParRate/ParSpread share the pos/neg float pair + fixed annuity;
  // Rate futures use the rate batch.
  pricing::BundleFloatBatch gen_pos_, gen_neg_, gen_rate_;
  pricing::BundleFixedLegs gen_fixed_;
  // Batch position -> (residual row, weight). A plain instrument is one batch entry with weight 1 on its
  // own row; a Portfolio's components are several batch entries that ACCUMULATE (weighted) onto the ONE
  // portfolio row -- which is exactly why a portfolio of W-cacheable components stays W-cacheable.
  struct Scatter { int row; double weight; };
  std::vector<Scatter> q_rows_, r_rows_;
  // FX-forward rows: F = fx_spot·DF[idx_num]/DF[idx_den] at time fx_time; residual (ln F − ln q)/fx_time.
  // Affine in x (ln DF = −Wx), so it rides the W-cache with a constant Jacobian row -- no AAD needed.
  struct Fx { int row, idx_num, idx_den; double fx_spot, fx_time; };
  std::vector<Fx> fx_rows_;
  // Bid/offer bands: a residual row whose value + Jacobian get the w(q) post-transform (see residuals /
  // jacobian). Empty for a plain bundle, so the fast path is untouched when no instrument is banded.
  struct Band { int row; double lower, upper, decay; };
  std::vector<Band> band_;
  Eigen::VectorXd market_;
  // Mutable per-call scratch (② reused Jacobian buffers, ③ DF memo) -- state that only CACHES pure
  // functions of x, so const-ness of residuals()/jacobian() is preserved semantically.
  mutable Eigen::VectorXd df_, df_x_;
  mutable Eigen::VectorXd out_, res_;  // model_rates / residuals result scratch (const-ref returns)
  mutable Eigen::MatrixXd G_, dnum_, dann_, dr_;
};

}  // namespace swaps::calibration
