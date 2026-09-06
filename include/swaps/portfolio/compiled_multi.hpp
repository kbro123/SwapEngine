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

  // True iff a Swap position's float leg fits the batch's arithmetic Σ w·(DF/DF−1) model: no compounded
  // (product-form) observation and no moment-path (fixing_step) coupon. Spreads, FX scale, realized
  // constants, weighted sub-periods and empty (fully fixed) observations are all representable.
  static bool swap_is_compilable(const MultiCurveBook::Position& p) {
    for (const auto& c : p.float_coupons)
      if (c.obs.compounded || c.obs.fixing_step > 0.0) return false;
    return true;
  }

 private:
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

  mutable Eigen::VectorXd df_, npv_;  // reusable per-reprice scratch (sized on first call)
};

}  // namespace swaps::portfolio
