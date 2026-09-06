#pragma once
// CompiledMultiCurveBook -- the COMPILED multi-curve book reprice kernel (audit item U2).
//
// BundleSession::price_portfolio prices a MultiCurveBook through templated VIRTUAL CurveHandle calls:
// every coupon's discount(t) walks the region list (and, on a spread chain, recurses through every
// ancestor's integral) -- ~ms for a desk-scale book. But a vanilla multi-curve swap's NPV is exactly the
// shape the Stage-3 W-cache already compiles for calibration (pricing/compiled_book.hpp): every DF it
// touches is exp(-W_all x) with W_all built ONCE from the bundle's curve structures, so a reprice is
//     DF_all     = exp(-W_all x)                                   (one matvec + vectorized exp)
//     float_pv_p = R_cpn · (DF[pay]·(DF[accS]/DF[accE]-1))          (gathered coupons, sparse reduce)
//     annuity_p  = R_fix · (tau·DF[pay])
//     NPV        = Σ notional_p·float_pv_p − Σ (notional_p·fixed_rate_p)·annuity_p
// with each position's OWN (forecast, discount, fixed) roles as global curve indices -- the batches are
// already role-aware, so the multi-curve-ness costs nothing over the single-curve CompiledPortfolio.
//
// NOT every position is W-cacheable, and -- mirroring HybridBundleResidual's split -- the ones that are
// not ride the EXISTING templated path instead of degrading the whole book:
//   * Kind::Xccy: the MtM leg's FX-reset notional is a DF RATIO of two curves, so its coupon PV is a
//     product of exponentials, not a single exp(-Wx) (the §2 linear-map guard). Priced through
//     MultiCurveBook::position_value over reusable bundle curve handles.
//   * a Swap whose float leg carries a COMPOUNDED (RFR lookback/lockout) observation -- the batch's
//     arithmetic Σ cannot represent the product (compiled_book push_obs throws on it);
//   * a Swap on the MOMENT path (fixing_step > 0) -- the batch has no ∫f² correction and would silently
//     drop it;
//   * every position, when a curve uses a value-dependent scheme (curves_are_noncacheable): then NO
//     curve has a constant W at all.
// npv(x) == MultiCurveBook::value<double> over build_bundle_curves at the same x, to rounding (the same
// per-coupon evaluation order; only exp(-w·x) vs the region integral differs, ~1e-15 relative).
//
// PERF RULE (same as pricing/compiled_book.hpp): the hot path allocates NOTHING after construction --
// DF and NPV land in reused mutable scratch, the batches return const refs into their own scratch, and
// the fallback handles are built once and set_forwards'd in place (BundleCurveSet reuses its buffers).

#include <Eigen/Core>

#include <vector>

#include "swaps/ad/dual.hpp"                       // ad::Dual -- the fallback PV01 forward-AAD pass
#include "swaps/calibration/bundle_problem.hpp"   // BundleCurveSet / CurveHandle (the fallback path)
#include "swaps/calibration/hybrid_residual.hpp"  // curves_are_noncacheable -- THE curve-level W guard
#include "swaps/portfolio/portfolio.hpp"          // MultiCurveBook
#include "swaps/pricing/compiled_book.hpp"        // CompiledCurveSet + BundleFloatBatch/BundleFixedLegs

namespace swaps::portfolio {

class CompiledMultiCurveBook {
 public:
  // `curves` is the bundle's curve topology (the SAME vector BundleProblem::curves holds); the state
  // vector x passed to npv() is the stacked per-curve state [interp knots | turn δ's] in curve order,
  // exactly what BundleSession calibrates. Both inputs are copied -- neither needs to outlive this.
  CompiledMultiCurveBook(const std::vector<pricing::CurveStructure>& curves, const MultiCurveBook& book)
      : specs_(curves) {
    // Per-curve state offsets into the stacked x (CurveStructure::n_knots = interp knots + turn δ's) --
    // the same layout CompiledCurveSet::init and BundleProblem::offset use.
    off_.assign(specs_.size(), 0);
    int o = 0;
    for (std::size_t c = 0; c < specs_.size(); ++c) {
      off_[c] = o;
      o += specs_[c].n_knots();
    }
    n_knots_ = o;

    // A value-dependent interpolation scheme anywhere means NO curve has a constant W (the same bundle
    // property HybridBundleResidual keys off) -- every position then rides the templated fallback.
    const bool nonlinear = calibration::curves_are_noncacheable(specs_);

    cs_.init(specs_);
    std::vector<double> notional, rate;
    for (const auto& p : book.positions) {
      if (!nonlinear && p.kind == MultiCurveBook::Kind::Swap && swap_is_compilable(p)) {
        notional.push_back(p.notional);
        rate.push_back(p.fixed_rate);
        float_.add(cs_, p.fwd_curve, p.disc_curve, p.float_coupons);
        // An EMPTY fixed leg registers an annuity row of 0, so notional·(pv − rate·0) reproduces the
        // templated float-only branch exactly (x − rate·0 == x bitwise).
        fixed_.add(cs_, p.fixed_curve, p.fixed_coupons);
      } else {
        fallback_.positions.push_back(p);
      }
    }
    cs_.finalize();
    float_.finalize();
    fixed_.finalize();

    // Per-registered-time row sum of W_all (Σ_j W[k,j]) -- CONSTANT for a fixed topology. Under a parallel
    // knot shift x -> x + ε·1, log-DF_k gains ε·rowsum_k, so DF_k's tangent is dDF_k/dε = -rowsum_k·DF_k.
    // This is all pv01()'s compiled half needs to turn the batch's dr/dDF partials into a d(NPV)/dε.
    if (cs_.n_times() > 0) rowsum_ = cs_.W().rowwise().sum();

    const int n = static_cast<int>(notional.size());
    notional_ = Eigen::Map<const Eigen::VectorXd>(notional.data(), n);
    nf_ = notional_.array() * Eigen::Map<const Eigen::VectorXd>(rate.data(), n).array();

    // Fallback curve handles: built ONCE (zero forwards), values overwritten in place per npv() call.
    if (!fallback_.positions.empty()) fb_curves_.build(specs_);
  }

  int n_positions() const { return n_compiled() + n_fallback(); }
  int n_compiled() const { return static_cast<int>(notional_.size()); }
  int n_fallback() const { return static_cast<int>(fallback_.positions.size()); }
  int n_times() const { return cs_.n_times(); }
  int n_knots() const { return n_knots_; }

  // Total book NPV at the stacked knot state x. Allocation-free after construction (reused scratch on
  // both halves). Matches MultiCurveBook::value<double> over build_bundle_curves(specs, x) to rounding.
  double npv(const Eigen::VectorXd& x) const {
    double total = 0.0;
    if (n_compiled() > 0) {
      cs_.df_into(x, df_);  // DF into scratch; pv/annuity return refs into the batches' own scratch
      const Eigen::VectorXd& pv = float_.pv(df_);
      const Eigen::VectorXd& ann = fixed_.annuity(df_);
      npv_ = (notional_.array() * pv.array() - nf_.array() * ann.array()).matrix();
      total = npv_.sum();
    }
    if (!fallback_.positions.empty()) {
      // Overwrite the handles' forwards in place (BundleCurveSet reuses its per-curve buffers), then
      // price the non-cacheable minority through the SAME templated kernel BundleSession uses today.
      fb_curves_.update([&](int c, int i) { return x[off_[c] + i]; });
      const auto cof = [this](int i) -> const calibration::CurveHandle<double>& { return fb_curves_[i]; };
      total += fallback_.value<double>(cof);
    }
    return total;
  }

  // The book's +1bp PARALLEL-shift PV01: 1e-4 · Σⱼ ∂NPV/∂xⱼ, matching the templated forward-AAD PV01
  // (BundleSession::price_portfolio) to rounding. Allocation-free on an all-compilable book (the streaming
  // twin's hot path); the non-cacheable minority adds a single templated AAD pass only when present.
  //
  // The COMPILED half is analytic -- no AAD, no per-call allocation. NPV depends on x only through
  // DF = exp(-W_all x), and under x -> x+ε·1 the tangent of DF_k is dDF_k = -rowsum_k·DF_k (rowsum_ is
  // constant). The directional derivative is then the batch's OWN dr/dDF partials (compiled_book.hpp)
  // contracted with that tangent, position order shared with notional_/nf_:
  //     ∂pv/∂DF[pay]   = (num+konst)·k                          (per coupon)
  //     ∂pv/∂DF[s],[e] = ±DF[pay]·k·w / DF[e]{,·DF[s]/DF[e]}    (per sub-period)
  //     ∂(−nf·ann)/∂DF[pay] = −nf·τ                             (per fixed coupon)
  // Summing weight·tangent[·] per contribution (not forming ∂NPV/∂DF first) handles the structural DF
  // aliasing (e_k == s_{k+1}, pay == e_k) for free -- each contribution adds its own term.
  double pv01(const Eigen::VectorXd& x) const {
    double g = 0.0;  // Σⱼ ∂NPV/∂xⱼ = the directional derivative along the all-ones knot direction
    if (n_compiled() > 0) {
      cs_.df_into(x, df_);
      t_ = (-rowsum_.array() * df_.array()).matrix();  // parallel-shift DF tangent (reused scratch)
      const Eigen::VectorXd& num = float_.num(df_);              // per-coupon Σ w·(DF[s]/DF[e]−1)
      const double* __restrict DF = df_.data();
      const double* __restrict tt = t_.data();
      // Float coupons: the pay-column partial (num+konst)·k, weighted by the owning position's notional.
      for (int c = 0; c < float_.n_coupons(); ++c) {
        const int i = float_.inst[c];
        g += notional_[i] * (num[c] + float_.konst[c]) * float_.k[c] * tt[float_.pay[c]];
      }
      // Float sub-periods: the forecast-curve start/end partials.
      for (int j = 0; j < static_cast<int>(float_.subS.size()); ++j) {
        const int c = float_.sub_cpn[j], i = float_.inst[c];
        const int s = float_.subS[j], e = float_.subE[j];
        const double f = notional_[i] * DF[float_.pay[c]] * float_.k[c] * float_.sub_w[j];
        g += f * (tt[s] / DF[e] - tt[e] * DF[s] / (DF[e] * DF[e]));
      }
      // Fixed annuities enter NPV as −(notional·fixed_rate)·Σ τ·DF[pay]: ∂/∂DF[pay] = −nf·τ.
      for (int i = 0; i < static_cast<int>(fixed_.pay.size()); ++i)
        g += -nf_[fixed_.inst[i]] * fixed_.tau[i] * tt[fixed_.pay[i]];
    }
    if (!fallback_.positions.empty()) g += fallback_directional(x);
    return 1e-4 * g;
  }

  // True iff a Swap position fits the batch's arithmetic Σ w·(DF/DF−1) model with a SINGLE scalar fixed-rate
  // row-scale (nf_ = notional⊙fixed_rate): no compounded (product-form) observation, no moment-path
  // (fixing_step) coupon, no STEPPED per-coupon fixed rate (which the scalar nf_ cannot express), and no
  // principal-exchange cashflows (which the batch has no row for). Spreads, FX scale, realized constants,
  // weighted sub-periods and empty (fully fixed) observations are all representable. A stepped / principal
  // position rides the templated fallback exactly like Xccy / compounded, so the compiled hot path is
  // untouched (nf_ stays a single scalar per compiled position) and the 12 perf metrics are unaffected.
  static bool swap_is_compilable(const MultiCurveBook::Position& p) {
    if (!p.fixed_rates.empty() || !p.principal_flows.empty()) return false;
    for (const auto& c : p.float_coupons)
      if (c.obs.compounded || c.obs.fixing_step > 0.0) return false;
    return true;
  }

 private:
  // The fallback minority's contribution to Σⱼ ∂NPV/∂xⱼ: ONE templated forward-AAD pass (seed every knot
  // with derivative 1, read the summed derivative). Heap-allocating (full-width ad::Dual), but reached
  // ONLY for a book carrying Xccy/compounded/moment positions -- never the all-compilable streaming path.
  double fallback_directional(const Eigen::VectorXd& x) const {
    using Dual = swaps::ad::Dual;
    Eigen::Matrix<Dual, Eigen::Dynamic, 1> xd(n_knots_);
    for (int k = 0; k < n_knots_; ++k) {
      xd[k].value() = x[k];
      xd[k].derivatives() = Eigen::VectorXd::Unit(n_knots_, k);
    }
    const auto C = calibration::build_bundle_curves<Dual>(
        specs_, [&](int c, int i) { return xd[off_[c] + i]; });
    const auto cof = [&C](int i) -> const calibration::CurveHandle<Dual>& { return *C[i]; };
    const Dual npv = fallback_.value<Dual>(cof);
    return npv.derivatives().size() ? npv.derivatives().sum() : 0.0;
  }

  std::vector<pricing::CurveStructure> specs_;  // owned copy (fb_curves_ references it; ctor arg may die)
  std::vector<int> off_;                        // curve -> offset of its state block in the stacked x
  int n_knots_ = 0;

  // Compiled half: the multi-curve W-cache and role-aware batches (one row per compiled position).
  pricing::CompiledCurveSet cs_;
  pricing::BundleFloatBatch float_;
  pricing::BundleFixedLegs fixed_;
  Eigen::VectorXd notional_, nf_;  // nf_ = notional ⊙ fixed_rate (annuity row-scale)

  // Templated half: the non-cacheable positions (Xccy / compounded / moment), priced through the
  // existing virtual-handle kernel off handles that are reused (values overwritten) every call.
  MultiCurveBook fallback_;
  mutable calibration::BundleCurveSet<double> fb_curves_;

  Eigen::VectorXd rowsum_;             // per-registered-time Σ_j W[k,j] (constant): parallel-shift DF tangent scale
  mutable Eigen::VectorXd df_, npv_, t_;  // reusable per-reprice scratch (df_/npv_ npv(); t_ pv01()'s DF tangent)
};

}  // namespace swaps::portfolio
