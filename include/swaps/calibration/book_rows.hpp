#pragma once
// A BOOK IS ROWS OF THE ONE ENGINE (architecture review item 1/4, 2026-10-06).
//
// Until 2026-10-06 a multi-curve book had a W-cache TWIN (portfolio/compiled_multi.hpp CompiledMultiCurveBook):
// its own CompiledCurveSet and batches, a hand-written parallel-shift PV01 contracting the batch arrays a third
// time, two "is this position compilable" predicates, a whole-set veto on value-dependent curves and a templated
// fallback half for everything the predicates refused (a stepped fixed rate, a principal exchange, an xccy
// position with a spot time, a seasoned MtM leg, a compounded observation). None of that is a book property --
// it is exactly what the calibration residual's row model, router and tiers already decide for a calibration
// row. So a position IS a row now:
//
//     Instrument{quote = Npv}            a position per unit notional (problem.hpp: + fwd − bench + mtm − fixed)
//     Portfolio{weight = notional, ...}  the position's notional, as the row's one term weight
//     BundleProblem{curves, rows}        the book, over the bundle's own curve structures
//     HybridBundleResidual               the engine: the W-cache for every row it can express, the AAD block for
//                                        the rest, the piecewise-linear tier on a value-dependent curve
//
// and the book's analytics are the engine's: NPV = model_rates (one DF = exp(-W_all x) matvec + the gathered
// reduce), the parallel PV01 = directional_into along pricing::parallel_direction (the batches' partials contracted
// with the DF tangent as they are scattered -- the SAME partials the Jacobian uses, no second implementation),
// per-position key-rate risk = jacobian. A stepped fixed rate and a principal exchange are DATA in the fixed leg's
// τ·scale; an xccy position is a bench (domestic) leg, an MtM leg and its two notional exchanges. Nothing here
// decides routing: Instrument::noncacheable does, as for any row.
//
// position_instrument() is the one conversion from the templated book's Position to a row; MultiCurveBook::
// position_value stays as the one-shot templated kernel (and the parity oracle, tests/book_rows_test.cpp).

#include <Eigen/Core>

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/portfolio/fx_pairs.hpp"
#include "swaps/portfolio/portfolio.hpp"
#include "swaps/pricing/curve_spec.hpp"

namespace swaps::calibration {

// A position per UNIT notional as an Npv instrument. The notional is applied by the caller (book_problem wraps
// it as a Portfolio weight). Term for term MultiCurveBook::position_value:
//   Swap: + float leg (fwd), − Σ DF·τ·scale·rate_i (the scalar fixed_rate or the stepped schedule folded into
//         each fixed coupon's scale), + principal flows;
//   Xccy: + the resetting MtM leg (fx_spot, spot time), − the domestic float leg (as the bench leg), − its
//         notional exchanges (+DF(e_N) − DF(s_0), each only while unsettled; or the explicit principal_flows).
// Principal flows discount on disc_curve: they join the fixed leg when it discounts there too (or is empty), else
// they ride the bench leg as fully-fixed coupons (realized = −amount) so the Npv row still prices them exactly.
inline Instrument position_instrument(const portfolio::MultiCurveBook::Position& p) {
  using Kind = portfolio::MultiCurveBook::Kind;
  Instrument ins;
  ins.quote = QuoteKind::Npv;
  const auto flows_into = [&](const std::vector<std::pair<double, double>>& flows, double sign) {
    // sign = +1: the flows ADD to the row (a swap's principal_pv); −1: they SUBTRACT (an xccy domestic leg's).
    if (flows.empty()) return;
    if (ins.fixed.coupons.empty() || ins.fixed.discount == p.disc_curve) {
      if (ins.fixed.coupons.empty()) ins.fixed.discount = p.disc_curve;
      for (const auto& [t, amount] : flows) ins.fixed.coupons.push_back({t, 1.0, -sign * amount});  // fixed enters −
    } else {
      for (const auto& [t, amount] : flows) {  // a dated amount as a fully-fixed coupon on the bench leg (enters −)
        pricing::FloatCoupon c;
        c.obs.tau_index = 1.0;
        c.obs.realized = -sign * amount;
        c.pay = t;
        c.tau_pay = 1.0;
        ins.bench.coupons.push_back(c);
      }
      ins.bench.forecast = p.fwd_curve;
      ins.bench.discount = p.disc_curve;
    }
  };
  if (p.kind == Kind::Xccy) {
    ins.bench = {p.float_coupons, p.fwd_curve, p.disc_curve};  // the domestic leg, paid: enters −
    ins.mtm = {p.mtm_coupons, p.mtm_fwd_curve, p.mtm_disc_curve};
    ins.mtm.reset_num = p.mtm_reset_num;
    ins.mtm.reset_den = p.mtm_reset_den;
    ins.mtm.fx_spot = p.fx_spot;
    ins.mtm.fx_spot_time = p.fx_spot_time;
    ins.fixed.discount = p.disc_curve;
    if (!p.principal_flows.empty()) {
      flows_into(p.principal_flows, -1.0);
    } else if (!p.float_coupons.empty()) {
      const auto& fc0 = p.float_coupons.front();
      const auto& fcN = p.float_coupons.back();
      if ((!fc0.accrual_set && fc0.obs.sub_start.empty()) || (!fcN.accrual_set && fcN.obs.sub_end.empty()))
        throw std::runtime_error(
            "Xccy position: a fully-fixed domestic coupon has no accrual period or observation window to place "
            "the notional exchange on; supply principal_flows explicitly or set accrual_start/accrual_end");
      const double s0 = fc0.accrual_set ? fc0.accrual_start : fc0.obs.sub_start.front();
      const double eN = fcN.accrual_set ? fcN.accrual_end : fcN.obs.sub_end.back();
      // Each exchange only while unsettled (>= 0: dated today still pays). PN2b: a book rolled into the last coupon's
      // payment-lag window keeps the coupon (interest owed) after its final exchange settled (e_N < 0).
      if (eN >= 0.0) ins.fixed.coupons.push_back({eN, 1.0, 1.0});  // the final exchange, unless already settled (PN2b)
      if (s0 >= 0.0) ins.fixed.coupons.push_back({s0, 1.0, -1.0});  // the initial exchange, unless already settled
    }
    return ins;
  }
  ins.fwd = {p.float_coupons, p.fwd_curve, p.disc_curve};
  if (!p.fixed_coupons.empty()) {
    if (!p.fixed_rates.empty() && p.fixed_rates.size() != p.fixed_coupons.size())
      throw std::invalid_argument("position_instrument: fixed_rates must be empty or one per fixed coupon");
    ins.fixed.discount = p.fixed_curve;
    for (std::size_t i = 0; i < p.fixed_coupons.size(); ++i) {
      pricing::FixedCoupon c = p.fixed_coupons[i];
      c.scale *= p.fixed_rates.empty() ? p.fixed_rate : p.fixed_rates[i];  // the rate IS the coupon's amount
      ins.fixed.coupons.push_back(c);
    }
  }
  flows_into(p.principal_flows, +1.0);
  return ins;
}

// The book as a BundleProblem over `curves`: one row per position, in book order, each a Portfolio of ONE Npv
// component weighted by the notional (so the row's model value is the position's NPV in the book's raw units;
// `market` 0 -- a book row is priced, never calibrated to, unless a caller says otherwise).
inline BundleProblem book_problem(const std::vector<pricing::CurveStructure>& curves,
                                  const portfolio::MultiCurveBook& book) {
  BundleProblem b;
  b.curves = curves;
  b.instruments.reserve(book.positions.size());
  for (const auto& p : book.positions) {
    Instrument row;
    row.quote = QuoteKind::Portfolio;
    row.combination.push_back({p.notional, position_instrument(p)});
    b.instruments.push_back(std::move(row));
  }
  return b;
}

// A multi-curve (+ xccy) book bound to a bundle's curve structures for REPEATED repricing: the W-cache engine built
// once, every analytic off it. Replaces portfolio::CompiledMultiCurveBook (same call shape: npv / pv01 /
// set_fx_factors / the counts), minus everything that was its own: the twin's partials, predicates, veto and
// templated fallback are the engine's routing now. The state vector x is the stacked per-curve state in curve
// order, exactly what BundleSession calibrates. PERF: the hot path allocates nothing after the first call.
class BookRows {
 public:
  BookRows(const std::vector<pricing::CurveStructure>& curves, const portfolio::MultiCurveBook& book)
      : prob_(book_problem(curves, book)), eng_(prob_), dir_(pricing::parallel_direction(prob_.curves)) {
    for (std::size_t k = 0; k < book.positions.size(); ++k)
      if (book.positions[k].kind == portfolio::MultiCurveBook::Kind::Xccy)
        xccy_.push_back({static_cast<int>(k), book.positions[k].fx_spot});
  }

  int n_positions() const { return prob_.n_residuals(); }
  int n_rows() const { return prob_.n_residuals(); }  // one row per position
  int n_compiled() const { return eng_.n_compiled_rows(); }
  int n_fallback() const { return n_positions() - n_compiled(); }  // rows the router sent to the AAD tier
  int n_times() const { return eng_.n_times(); }
  int n_knots() const { return prob_.n_knots(); }
  const BundleProblem& problem() const { return prob_; }
  const HybridBundleResidual& engine() const { return eng_; }

  // SC2: put every xccy position's FX spot at its pair's factor (portfolio::fx_spot_under) IN PLACE -- the row's MtM
  // leg weight on the compiled engine, the leaf instrument's fx_spot on the AAD block. `slots` must come from the
  // book this was built from. Bitwise a fresh build at those spots; factors of 1 restore the constructed book
  // exactly. Allocation-free.
  void set_fx_factors(const portfolio::FxPairSlots& slots, std::span<const double> slot_factor) {
    for (const XccySpot& e : xccy_)
      eng_.set_mtm_fx_spot(e.position, portfolio::fx_spot_under(e.fx_spot, slots.slot_of[static_cast<std::size_t>(e.position)], slot_factor));
  }

  // Per-position NPVs (a const ref into the engine's scratch) and the book total, at the stacked state x.
  const Eigen::VectorXd& npvs(const Eigen::VectorXd& x) const { return eng_.model_rates(x); }
  double npv(const Eigen::VectorXd& x) const { return n_rows() == 0 ? 0.0 : eng_.model_rates(x).sum(); }

  // The +1bp PARALLEL-shift PV01: 1e-4 · (J·u) summed over the rows, u = pricing::parallel_direction (1 on outright
  // curves' interpolation knots, 0 on spread knots and turn deltas). Analytic on every tier, no J formed.
  const Eigen::VectorXd& pv01s(const Eigen::VectorXd& x) const {
    eng_.directional_into(x, dir_, d_);
    d_ *= 1e-4;
    return d_;
  }
  double pv01(const Eigen::VectorXd& x) const { return n_rows() == 0 ? 0.0 : pv01s(x).sum(); }

  // Per-position key-rate risk d(NPV)/dx (n_positions × n_knots): the engine's Jacobian, nothing book-specific.
  Eigen::MatrixXd key_rate(const Eigen::VectorXd& x) const { return eng_.jacobian(x); }

 private:
  struct XccySpot { int position; double fx_spot; };  // an xccy position's row and its base spot
  BundleProblem prob_;
  HybridBundleResidual eng_;
  Eigen::VectorXd dir_;
  std::vector<XccySpot> xccy_;
  mutable Eigen::VectorXd d_;
};

}  // namespace swaps::calibration
