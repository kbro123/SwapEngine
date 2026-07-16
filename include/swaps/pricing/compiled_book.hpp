#pragma once
// Multi-curve vectorized discount engine (Stage 3 W-cache). Generalizes pricing/compiled.hpp from ONE
// curve to a BUNDLE of curves calibrated over a stacked x = [x_0; x_1; ...], WITHOUT leaving the linear
// W-cache fast path.
//
// The key structural fact (CLAUDE.md §2): every curve's log-discount is LINEAR in the stacked x, even a
// spread curve (integral_spread just ADDS to integral_base). So if we CONCATENATE every curve's DF
// vector into one global DF and give each cashflow a global index, then
//     DF_all = exp(-W_all x)
// with a block-structured W_all whose row for (curve c, time t) is the log-DF weight of c at t (c's own
// knot weights, PLUS the base's weights if c is a spread curve). The single-curve gather/reduce
// primitives and the analytic Jacobian then carry over UNCHANGED -- the multi-curve-ness lives entirely
// in the global indices and W_all's block structure. One curve is the trivial special case.
//
// This header holds the curve-agnostic pricing primitives: the multi-curve DF engine (CompiledCurveSet)
// and role-aware leg batches that gather from the global DF. Consumers (calibration residual, portfolio
// NPV) are thin final transforms on pv / annuity / rate -- see calibration/compiled_bundle.hpp.
//
// There is exactly ONE float primitive (BundleFloatBatch, design §4): the compiled form of the generic
// FloatCoupon/RateObservation model. Legs and futures, compounded and averaged, spread and no spread,
// any day-count basis are all the same batch with different DATA.

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cassert>
#include <map>
#include <utility>
#include <vector>

#include "swaps/pricing/cashflows.hpp"
#include "swaps/pricing/compiled.hpp"  // integral_weight_matrix, detail::to_vec

namespace swaps::pricing {

// A curve's knot structure + parameterization, mirroring BundleProblem::CurveSpec but with no
// calibration dependency (so this header stays in the pricing layer).
struct CurveStructure {
  std::vector<double> meeting, back;
  int base = -1;  // -1 = outright; else this curve = curves[base] + spread (spread knots)
  int n_knots() const { return static_cast<int>(meeting.size() + back.size()); }
};

// DF_all = exp(-W_all x) over every (curve, time) registered, concatenated into one global vector.
// reg(curve, t) returns a stable GLOBAL index in registration order; finalize() builds W_all by
// scattering each curve's log-DF weight rows (own knots + spread-base ancestry) into their global rows.
class CompiledCurveSet {
 public:
  void init(const std::vector<CurveStructure>& specs) {
    specs_ = specs;
    knot_offset_.assign(specs.size(), 0);
    int o = 0;
    for (std::size_t c = 0; c < specs.size(); ++c) {
      knot_offset_[c] = o;
      o += specs[c].n_knots();
    }
    n_knots_ = o;
  }

  int reg(int curve, double t) {
    const auto key = std::make_pair(curve, t);
    auto it = idx_.find(key);
    if (it != idx_.end()) return it->second;
    const int g = static_cast<int>(pts_.size());
    idx_.emplace(key, g);
    pts_.push_back(key);
    return g;
  }

  void finalize() {
    W_ = Eigen::MatrixXd::Zero(static_cast<int>(pts_.size()), n_knots_);
    // Group registered times by curve (one integral_weight_matrix call per curve), then scatter each
    // computed weight row back to its GLOBAL row index.
    std::vector<std::vector<int>> rows(specs_.size());
    std::vector<std::vector<double>> tms(specs_.size());
    for (int g = 0; g < static_cast<int>(pts_.size()); ++g) {
      rows[pts_[g].first].push_back(g);
      tms[pts_[g].first].push_back(pts_[g].second);
    }
    for (int c = 0; c < static_cast<int>(specs_.size()); ++c) {
      if (tms[c].empty()) continue;
      const Eigen::MatrixXd Wc = logdf_weight(c, tms[c]);
      for (int k = 0; k < static_cast<int>(rows[c].size()); ++k) W_.row(rows[c][k]) = Wc.row(k);
    }
  }

  Eigen::VectorXd df(const Eigen::VectorXd& x) const { return (-(W_ * x).array()).exp(); }
  const Eigen::MatrixXd& W() const { return W_; }
  int n_times() const { return static_cast<int>(pts_.size()); }
  int n_knots() const { return n_knots_; }

 private:
  // Log-DF weight rows of curve c at `times`: own knot weights in c's block, PLUS the base's log-DF
  // weights (recursively) if c is a spread curve. integral_spread(t) adds to integral_base(t), so the
  // total stays linear in the stacked x -- the whole reason a spread curve keeps the W-cache.
  Eigen::MatrixXd logdf_weight(int c, const std::vector<double>& times) const {
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(static_cast<int>(times.size()), n_knots_);
    W.middleCols(knot_offset_[c], specs_[c].n_knots()) =
        integral_weight_matrix(specs_[c].meeting, specs_[c].back, times);
    if (specs_[c].base >= 0) W += logdf_weight(specs_[c].base, times);
    return W;
  }

  std::vector<CurveStructure> specs_;
  std::vector<int> knot_offset_;
  int n_knots_ = 0;
  std::map<std::pair<int, double>, int> idx_;
  std::vector<std::pair<int, double>> pts_;  // global index -> (curve, time)
  Eigen::MatrixXd W_;
};

// THE float batch (design §4) -- the compiled form of the generic FloatCoupon/RateObservation model.
// N instruments, each a list of coupons, each coupon an observation with ANY number of weighted
// sub-periods. Two sparse reductions:
//
//   sub   = DF[s_k]/DF[e_k] - 1                      (per sub-period, forecast curve)
//   num   = R_sub * sub                              (sub-periods -> per-coupon numerator; w_k are
//                                                     R_sub's VALUES, so no separate w-multiply)
//   pv    = R_cpn * ( DF[pay] * (num + konst) * k )  (coupons -> per-instrument leg PV)
//
// in the k-form of pricing/cashflows.hpp float_coupon_pv (konst = realized + spread*tau_index,
// k = tau_pay/tau_index), which is what makes the one-sub-period/no-spread/tau_pay==tau_index shape
// reduce BIT-EXACTLY to the pre-generalization DF[pay]*(DF[accS]/DF[accE]-1).
//
// A FUTURE is the same batch stopping earlier: rate = (num + realized)*inv_tau + convexity. So the
// old BundleFloatLegs, BundleCompFutures and BundleAvgFutures are all THIS type -- "1M averaged SOFR
// future" vs "3M compounded future" vs "IBOR coupon" is data (how many sub-periods, what weights),
// not a type (design §1).
//
// All curve indices are GLOBAL into the CompiledCurveSet.
struct BundleFloatBatch {
  // Per sub-period (every coupon of every instrument, flattened).
  Eigen::VectorXi subS, subE, sub_cpn;  // sub_cpn = owning coupon
  Eigen::VectorXd sub_w;                // w_k (also carried as R_sub's values)
  Eigen::SparseMatrix<double> R_sub;    // n_coupons x n_subs, values = w_k
  // Per coupon.
  Eigen::VectorXi pay, inst;                       // inst = owning instrument; pay < 0 for futures
  Eigen::VectorXd konst, k, realized, inv_tau, convexity;
  Eigen::SparseMatrix<double> R_cpn;               // n_inst x n_coupons, 0/1
  int n_inst = 0;
  // True iff R_sub is EXACTLY the identity (one unit-weight sub-period per coupon, in order) -- the
  // standard compounded-OIS / IBOR / single-period-future shape. Then R_sub*v == v and we skip the
  // reduction entirely, so generalizing costs the hot path nothing.
  bool sub_is_identity = false;
  // True iff EVERY coupon has konst == 0 and k == 1 -- no spread, nothing realized, and
  // tau_pay == tau_index: the standard OIS / portfolio shape. The "+ konst" and "* k" passes are then
  // pure overhead over the whole coupon vector (~1.28x on a 1000-swap book, measured), so pv() fuses
  // to the minimal pre-generalization expression. Genericity must cost the hot path nothing (design §4).
  bool cpn_is_plain = false;

  int n_coupons() const { return n_cpn_; }
  int size() const { return n_inst; }

  // --- generic registration -------------------------------------------------------------------
  // One instrument = one float leg: coupons forecast `fc`, discount `dc`.
  void add(CompiledCurveSet& cs, int fc, int dc, const std::vector<FloatCoupon>& leg) {
    for (const auto& c : leg) {
      push_obs(cs, fc, c.obs);
      push_coupon(cs.reg(dc, c.pay), c.obs.realized + c.spread * c.obs.tau_index,
                  c.tau_pay / c.obs.tau_index, c.obs.realized, 1.0 / c.obs.tau_index, 0.0);
    }
    ++n_inst;
  }
  // One instrument = one future on `obs` forecasting `fc`. No discounting: the terminal transform is
  // rate(), not pv(). `convexity` is an INPUT NUMBER (design §3) -- the model lives in tests.
  void add_future(CompiledCurveSet& cs, int fc, const RateObservation& o, double conv) {
    push_obs(cs, fc, o);
    push_coupon(-1, 0.0, 0.0, o.realized, 1.0 / o.tau_index, conv);
    ++n_inst;
  }

  // --- legacy adapters ------------------------------------------------------------------------
  // Call sites still hold OisSwap / CompoundedFuture / AveragedFuture (design §3 migrates them to
  // legs+quote-transforms in a later pass). Each maps onto the generic form above; the mapping is
  // chosen so the legacy shape lands on the identity/unit-weight path and reprices bit-exactly.
  void add(CompiledCurveSet& cs, int fc, int dc, const OisSwap& s) {
    for (std::size_t i = 0; i < s.float_pay.size(); ++i) {
      // The legacy OIS coupon amount IS the DF ratio minus one (the accrual is already inside the
      // compounded growth factor), so tau_index == tau_pay == 1 => k == 1.0 exactly, konst == 0.
      push_sub(cs, fc, s.float_acc_start[i], s.float_acc_end[i], 1.0);
      push_coupon(cs.reg(dc, s.float_pay[i]), 0.0, 1.0, 0.0, 1.0, 0.0);
    }
    ++n_inst;
  }
  void add_future(CompiledCurveSet& cs, int fc, const CompoundedFuture& f, double conv) {
    push_sub(cs, fc, f.start, f.end, 1.0);            // one sub-period: the compounding telescopes
    push_coupon(-1, 0.0, 0.0, 0.0, 1.0 / f.accrual, conv);
    ++n_inst;
  }
  void add_future(CompiledCurveSet& cs, int fc, const AveragedFuture& f, double conv) {
    for (std::size_t i = 0; i < f.sub_start.size(); ++i)  // one sub-period per forward business day
      push_sub(cs, fc, f.sub_start[i], f.sub_end[i], 1.0);
    push_coupon(-1, 0.0, 0.0, f.realized_sum, 1.0 / f.period_yf, conv);
    ++n_inst;
  }

  void finalize() {
    subS = detail::to_vec(ss_);
    subE = detail::to_vec(se_);
    sub_cpn = detail::to_vec(sc_);
    sub_w = detail::to_vec(sw_);
    pay = detail::to_vec(p_);
    inst = detail::to_vec(row_);
    konst = detail::to_vec(konst_);
    k = detail::to_vec(k_);
    realized = detail::to_vec(rz_);
    inv_tau = detail::to_vec(it_);
    convexity = detail::to_vec(cv_);
    R_sub = build(sc_, sw_, n_cpn_);
    R_cpn = build(row_, std::vector<double>(row_.size(), 1.0), n_inst);
    sub_is_identity = (static_cast<int>(sc_.size()) == n_cpn_);
    for (std::size_t j = 0; sub_is_identity && j < sc_.size(); ++j)
      sub_is_identity = (sc_[j] == static_cast<int>(j) && sw_[j] == 1.0);
    cpn_is_plain = true;
    for (std::size_t j = 0; cpn_is_plain && j < konst_.size(); ++j)
      cpn_is_plain = (konst_[j] == 0.0 && k_[j] == 1.0);
  }

  // --- pricing --------------------------------------------------------------------------------
  // Per-coupon numerator num = sum_k w_k (DF[s_k]/DF[e_k] - 1). Materialized (PERF RULE).
  Eigen::VectorXd num(const Eigen::VectorXd& DF) const {
    const Eigen::VectorXd sub = (DF(subS).array() / DF(subE).array() - 1.0).matrix();
    if (sub_is_identity) return sub;  // R_sub == I: the reduction is a bitwise no-op
    return R_sub * sub;
  }

  // Per-instrument float-leg PV.
  Eigen::VectorXd pv(const Eigen::VectorXd& DF) const {
    // PERF RULE: what reaches R_cpn must be a materialized VectorXd -- handing Eigen's sparse*dense an
    // unevaluated gather/divide re-does that work per access (~1.28x slower, measured). On the identity
    // path we fuse the sub-period gather straight into the coupon vector, so the standard shape costs
    // exactly one materialized pass, as it did before this generalization.
    assert((pay.size() == 0 || pay.minCoeff() >= 0) && "pv() needs pay dates: this is a futures batch");
    Eigen::VectorXd coupon;
    if (sub_is_identity && cpn_is_plain)  // standard shape: identical work to pre-generalization
      coupon = (DF(pay).array() * (DF(subS).array() / DF(subE).array() - 1.0)).matrix();
    else if (sub_is_identity)
      coupon = (DF(pay).array() * (DF(subS).array() / DF(subE).array() - 1.0 + konst.array()) *
                k.array())
                   .matrix();
    else
      coupon = (DF(pay).array() * (num(DF).array() + konst.array()) * k.array()).matrix();
    return R_cpn * coupon;
  }

  // Per-instrument (per-future) rate = (num + realized)*inv_tau + convexity.
  Eigen::VectorXd rate(const Eigen::VectorXd& DF) const {
    if (sub_is_identity)
      return ((DF(subS).array() / DF(subE).array() - 1.0 + realized.array()) * inv_tau.array() +
              convexity.array())
          .matrix();
    return ((num(DF).array() + realized.array()) * inv_tau.array() + convexity.array()).matrix();
  }

  // --- analytic sensitivities (the ANALYTIC Jacobian's dr/dDF; no AAD in the hot loop) ----------
  // d(pv)/dDF, accumulated (with `sign`) into rows [row0, row0+n_inst) of `d`. For the generic coupon
  // pv = DF[pay]·A·k with A = Σ_k w_k(DF[s_k]/DF[e_k] − 1) + konst (the k-form):
  //     d pv / d DF[pay] = A·k
  //     d pv / d DF[s_k] = + DF[pay]·k·w_k / DF[e_k]
  //     d pv / d DF[e_k] = − DF[pay]·k·w_k·DF[s_k] / DF[e_k]²
  // tau_index vanishes into k, so at k=1, w=1, konst=0 (one sub-period, no spread,
  // tau_pay == tau_index) these are EXACTLY the pre-generalization partials.
  //
  // TWO passes, not one: d pv/d DF[pay] = A·k depends on the coupon's WHOLE numerator, i.e. on OTHER
  // DF entries, so A must be materialized before anything can be scattered to the pay column.
  //
  // `+=` (never `=`) is load-bearing -- two structural aliases are live and MUST accumulate:
  // consecutive averaged sub-periods share a registered time (e_k == s_{k+1}), and pay == e_k when
  // the forecast and discount curves coincide with no payment lag.
  void d_pv(const Eigen::VectorXd& DF, Eigen::MatrixXd& d, int row0, double sign) const {
    assert((pay.size() == 0 || pay.minCoeff() >= 0) && "d_pv() needs pay dates: this is a futures batch");
    const Eigen::VectorXd A = (num(DF).array() + konst.array()).matrix();
    for (int i = 0; i < n_cpn_; ++i) d(row0 + inst[i], pay[i]) += sign * A[i] * k[i];
    for (int j = 0; j < static_cast<int>(subS.size()); ++j) {
      const int i = sub_cpn[j], s = subS[j], e = subE[j];
      const double f = sign * DF[pay[i]] * k[i] * sub_w[j];
      d(row0 + inst[i], s) += f / DF[e];
      d(row0 + inst[i], e) += -f * DF[s] / (DF[e] * DF[e]);
    }
  }
  // d(rate)/dDF for a futures batch (realized and convexity are constants -> zero derivative):
  //     d rate / d DF[s_k] = + w_k·inv_tau / DF[e_k]
  //     d rate / d DF[e_k] = − w_k·inv_tau·DF[s_k] / DF[e_k]²
  void d_rate(const Eigen::VectorXd& DF, Eigen::MatrixXd& d, int row0) const {
    for (int j = 0; j < static_cast<int>(subS.size()); ++j) {
      const int i = sub_cpn[j], s = subS[j], e = subE[j], r = row0 + inst[i];
      const double f = inv_tau[i] * sub_w[j];
      d(r, s) += f / DF[e];
      d(r, e) += -f * DF[s] / (DF[e] * DF[e]);
    }
  }

 private:
  void push_sub(CompiledCurveSet& cs, int fc, double s_t, double e_t, double w) {
    ss_.push_back(cs.reg(fc, s_t));
    se_.push_back(cs.reg(fc, e_t));
    sc_.push_back(n_cpn_);  // the coupon about to be pushed
    sw_.push_back(w);
  }
  void push_obs(CompiledCurveSet& cs, int fc, const RateObservation& o) {
    const bool weighted = !o.weight.empty();
    for (std::size_t j = 0; j < o.sub_start.size(); ++j)
      push_sub(cs, fc, o.sub_start[j], o.sub_end[j], weighted ? o.weight[j] : 1.0);
  }
  void push_coupon(int pay_idx, double konst, double kk, double rz, double inv_tau, double conv) {
    p_.push_back(pay_idx);
    konst_.push_back(konst);
    k_.push_back(kk);
    rz_.push_back(rz);
    it_.push_back(inv_tau);
    cv_.push_back(conv);
    row_.push_back(n_inst);
    ++n_cpn_;
  }
  static Eigen::SparseMatrix<double> build(const std::vector<int>& row, const std::vector<double>& val,
                                           int n_rows) {
    std::vector<Eigen::Triplet<double>> trip;
    trip.reserve(row.size());
    for (int j = 0; j < static_cast<int>(row.size()); ++j) trip.emplace_back(row[j], j, val[j]);
    Eigen::SparseMatrix<double> M(n_rows, static_cast<int>(row.size()));
    M.setFromTriplets(trip.begin(), trip.end());
    return M;
  }

  int n_cpn_ = 0;
  std::vector<int> ss_, se_, sc_, p_, row_;
  std::vector<double> sw_, konst_, k_, rz_, it_, cv_;
};

// Fixed-leg annuities of N instruments discounting curve `dc`: ann = sum(tau*DF[pay]).
struct BundleFixedLegs {
  Eigen::VectorXi pay, inst;
  Eigen::VectorXd tau;
  Eigen::SparseMatrix<double> R;
  int n_inst = 0;

  void add(CompiledCurveSet& cs, int dc, const OisSwap& s) {
    for (std::size_t i = 0; i < s.fixed_pay.size(); ++i) {
      p_.push_back(cs.reg(dc, s.fixed_pay[i]));
      t_.push_back(s.fixed_accrual[i]);
      row_.push_back(n_inst);
    }
    ++n_inst;
  }
  void finalize() {
    pay = detail::to_vec(p_);
    tau = detail::to_vec(t_);
    inst = detail::to_vec(row_);
    std::vector<Eigen::Triplet<double>> trip;
    for (int k = 0; k < static_cast<int>(row_.size()); ++k) trip.emplace_back(row_[k], k, 1.0);
    R.resize(n_inst, static_cast<int>(row_.size()));
    R.setFromTriplets(trip.begin(), trip.end());
  }
  Eigen::VectorXd annuity(const Eigen::VectorXd& DF) const {
    const Eigen::VectorXd disc = tau.array() * DF(pay).array();  // materialize before sparse reduction
    return R * disc;
  }
  // d(annuity)/dDF[pay_i] = tau_i, accumulated into rows [row0, row0+n_inst).
  void d_annuity(Eigen::MatrixXd& d, int row0) const {
    for (int i = 0; i < static_cast<int>(pay.size()); ++i) d(row0 + inst[i], pay[i]) += tau[i];
  }

 private:
  std::vector<int> p_, row_;
  std::vector<double> t_;
};

}  // namespace swaps::pricing
