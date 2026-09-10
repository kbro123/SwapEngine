#ifndef SWAPS_BUILD_CREDIT_INSTRUMENTS_HPP
#define SWAPS_BUILD_CREDIT_INSTRUMENTS_HPP
// build/credit_instruments.hpp — a par CDS (credit default swap) calibration instrument, plus the builder
// that assembles its premium schedule and protection-integration grid. Additive, QuantLib-free, header-only
// (mirrors build/inflation_instruments.hpp for the inflation swaps).
//
// CONVENTIONS chosen here (all times are curve-time year fractions, ACT/365F from the value date — the same
// convention every builder in build/instruments.hpp uses):
//
//   PREMIUM LEG — the protection buyer pays spread s on a QUARTERLY schedule (default freq = 4). The
//     coupon accrual is ACT/360; on the ACT/365F curve-time axis that day count is applied as
//         τᵢ = (tᵢ − tᵢ₋₁) · 365/360.
//     The leg's value per unit spread is the RISKY ANNUITY (a.k.a. RPV01):
//         A(s=1) = Σᵢ DF(tᵢ) · Q(tᵢ) · τᵢ,
//     i.e. each coupon is discounted (DF, nominal) AND weighted by survival to its payment date Q(tᵢ).
//     (No accrual-on-default term — premium is modelled as paid only if the name survives the period end,
//     the standard "no accrued" risky annuity; documented here so the discretisation is explicit.)
//
//   PROTECTION LEG — pays (1−R) at the moment of default. Recovery R defaults to 0.40. The exact value is
//         P = (1−R) · ∫₀ᵀ DF(u) · (−dQ(u)),
//     discretised on a fine uniform grid 0 = u₀ < u₁ < … < u_M = T with the DEFAULT-PAID-AT-PERIOD-END
//     convention (the protection cash-flow of each subinterval is discounted at the interval's RIGHT end):
//         P ≈ (1−R) · Σⱼ DF(uⱼ) · ( Q(uⱼ₋₁) − Q(uⱼ) ).
//     Q(uⱼ₋₁) − Q(uⱼ) is the default probability inside the subinterval; refining the grid (prot_steps
//     per premium period) makes the piecewise-flat-DF approximation of the integral arbitrarily accurate.
//
//   PAR SPREAD — the breakeven quote is the spread that zeroes the CDS PV:
//         s* = P / A(s=1) = protectionPV / riskyAnnuity.
//     Note s* scales with (1−R): a HIGHER recovery ⇒ a SMALLER protection leg ⇒ a LOWER par spread.
//
// The nominal DISCOUNT FACTORS DF(·) enter ONLY as BUILD-TIME WEIGHTS (doubles), captured when the
// instrument is built — exactly like the nominal weights in the YoY inflation swap. The free, calibrated
// object is the hazard/survival curve, so an AAD sweep differentiates only the Q(·) terms.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "swaps/build/day_count.hpp"
#include "swaps/calibration/credit_instrument.hpp"  // the row type (E6.4); make_cds below builds one
#include "swaps/curve/hazard.hpp"

namespace swaps::build {

// The row type lives in calibration/ (E6.4); this alias keeps `build::CdsInstrument` spelling valid.
using calibration::CdsInstrument;

// One par-CDS calibration instrument. `model_quote<Scalar>` prices its PAR SPREAD against a SurvivalCurve;
// the residual is (model_quote − market), rate units, like every other calibration instrument.
// (365/360 for ACT/360 — the ISDA standard — 1.0 for ACT/365F): credit_accrual_ratio(day_count). All of
// premium_freq / prot_steps / accrual_ratio are product DATA (credit.cds_products[]), never defaulted here.
inline double credit_accrual_ratio(const std::string& day_count) { return 365.0 / day_count_basis(day_count); }
template <class DiscFn>
inline CdsInstrument make_cds(double maturity, double spread, double recovery, DiscFn&& df,
                             int premium_freq, int prot_steps, double accrual_ratio) {
  if (!(maturity > 0.0)) throw std::invalid_argument("make_cds: maturity must be > 0");
  if (!(recovery >= 0.0 && recovery < 1.0)) throw std::invalid_argument("make_cds: recovery must be in [0, 1)");
  if (!(accrual_ratio > 0.0)) throw std::invalid_argument("make_cds: accrual_ratio must be > 0");
  if (premium_freq < 1) throw std::invalid_argument("make_cds: premium_freq must be >= 1");
  if (prot_steps < 1) throw std::invalid_argument("make_cds: prot_steps must be >= 1");

  CdsInstrument ins;
  ins.recovery = recovery;
  ins.maturity = maturity;
  ins.market = spread;

  // Premium schedule: quarterly (1/freq) coupon ends up to T, final snapped exactly to maturity. ACT/360
  // accrual on the ACT/365F axis => τ = (Δt years)·365/360.
  const double dt = 1.0 / static_cast<double>(premium_freq);
  std::vector<double> ends;
  for (int i = 1; i * dt < maturity - 1e-9; ++i) ends.push_back(i * dt);
  ends.push_back(maturity);
  double prev = 0.0;
  for (double e : ends) {
    ins.pay.push_back(e);
    ins.tau.push_back((e - prev) * accrual_ratio);
    ins.df_prem.push_back(df(e));
    prev = e;
  }

  // Protection grid: a uniform subdivision of [0, T] into (#premium periods × prot_steps) subintervals.
  const int M = static_cast<int>(ends.size()) * prot_steps;
  const double h = maturity / static_cast<double>(M);
  for (int j = 1; j <= M; ++j) {
    const double l = (j - 1) * h;
    const double r = (j == M) ? maturity : j * h;  // snap the last right-end exactly to maturity
    ins.gl.push_back(l);
    ins.gr.push_back(r);
    ins.df_prot.push_back(df(r));
  }
  return ins;
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_CREDIT_INSTRUMENTS_HPP
