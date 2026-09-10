#ifndef SWAPS_BUILD_INFLATION_INSTRUMENTS_HPP
#define SWAPS_BUILD_INFLATION_INSTRUMENTS_HPP
// build/inflation_instruments.hpp — inflation-swap calibration instruments: a Zero-Coupon Inflation Swap
// (ZCIS) and a Year-on-Year Inflation Swap (YoY), plus the builders that assemble their schedules. Additive,
// QuantLib-free, header-only (mirrors build/instruments.hpp for nominal swaps).
//
// CONVENTIONS chosen here (all times are curve-time year fractions, ACT/365F from the value date — the
// same convention every builder in build/instruments.hpp uses):
//   ZCIS  — a single index exchange at maturity T. The inflation leg pays I(T)/I(0) − 1; the fixed leg
//           pays (1+k)^T − 1. Par => (1+k)^T = I(T)/I(0), so the quoted breakeven k is ANNUALLY
//           COMPOUNDED and the model quote is the curve's `zc_breakeven(T)`. Deterministic-curve
//           breakevens are discount-curve independent (the single payment's DF cancels).
//   YoY   — annual periods [t_{i-1}, t_i]; the inflation leg pays I(t_i)/I(t_{i-1}) − 1 each period, the
//           fixed leg pays k. The market quote is the PAR fixed rate,
//               k = Σ DF_n(t_i)·R_i  /  Σ DF_n(t_i)·τ_i,      R_i = I(t_i)/I(t_{i-1}) − 1,
//           discounted on the NOMINAL curve. The nominal discount factors enter only as BUILD-TIME
//           WEIGHTS (the breakeven curve is the free object), so an AAD sweep differentiates only R_i.
//           With a flat/zero nominal it reduces to the τ-weighted average of the period returns.
//
// WHY ZCIS is not "just legs": its (1+k)^T convention is a POWER of the index ratio, not a leg-PV ratio,
// so it uses the dedicated `zc_breakeven` transform rather than a QuoteKind. YoY, by contrast, IS two
// legs / one indexed and would drop straight into the existing QuoteKind::ParRate path with the breakeven
// curve in the "forecast" role (a leg-composed helper that proved this had no consumer and was deleted in
// E6.1, 2026-09-10; InflationInstrument's YoY model_quote is the shipped form).

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "swaps/curve/inflation.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace swaps::build {

namespace px = swaps::pricing;

// One inflation calibration instrument (ZCIS or YoY). Its `model_quote<Scalar>` prices against an
// InflationIndexCurve; the residual is (model_quote − market), rate units, like every other instrument.
struct InflationInstrument {
  enum class Kind { ZCIS, YoY };
  Kind kind = Kind::ZCIS;

  // ZCIS: the single maturity.
  double maturity = 0.0;

  // YoY: per-period data. t0[i]/t1[i] are the period start/end (curve time); w_num[i] = DF_n(pay_i) and
  // w_den[i] = DF_n(pay_i)·τ_i are the build-time nominal-discount weights of the par-rate quotient.
  std::vector<double> t0, t1, w_num, w_den;

  double market = 0.0;  // quoted breakeven (ZCIS) / par fixed rate (YoY), rate units

  template <class Scalar, class InflCurve>
  Scalar model_quote(const InflCurve& c) const {
    if (kind == Kind::ZCIS) return c.zc_breakeven(maturity);
    // YoY par fixed rate = Σ w_num·R_i / Σ w_den. Seed the numerator from the FIRST curve-dependent term
    // (R_0 carries the AAD derivatives); the denominator is a pure build-time constant.
    Scalar num = Scalar(w_num[0]) * c.yoy_forward(t0[0], t1[0]);
    double den = w_den[0];
    for (std::size_t i = 1; i < t0.size(); ++i) {
      num += Scalar(w_num[i]) * c.yoy_forward(t0[i], t1[i]);
      den += w_den[i];
    }
    return num / den;
  }

  template <class Scalar, class InflCurve>
  Scalar residual(const InflCurve& c) const {
    return model_quote<Scalar>(c) - market;
  }
};

// ZCIS builder: maturity (curve-time years) + quoted annually-compounded breakeven.
inline InflationInstrument inflation_zcis(double maturity_years, double breakeven) {
  if (!(maturity_years > 0.0)) throw std::invalid_argument("inflation_zcis: maturity must be > 0");
  InflationInstrument ins;
  ins.kind = InflationInstrument::Kind::ZCIS;
  ins.maturity = maturity_years;
  ins.market = breakeven;
  return ins;
}

// YoY builder. `period_ends` are the annual reset/payment times (curve years); the first period runs from
// 0 to period_ends[0]. `nom_zero` is a flat continuously-compounded nominal zero rate used only to weight
// the par-rate quotient (default 0 => a τ-weighted average, discount-independent). `par_rate` is the quote.
inline InflationInstrument inflation_yoy(const std::vector<double>& period_ends, double par_rate,
                                         double nom_zero = 0.0) {
  if (period_ends.empty()) throw std::invalid_argument("inflation_yoy: need at least one period");
  InflationInstrument ins;
  ins.kind = InflationInstrument::Kind::YoY;
  ins.market = par_rate;
  double prev = 0.0;
  for (double e : period_ends) {
    if (!(e > prev)) throw std::invalid_argument("inflation_yoy: period_ends must be strictly increasing");
    const double tau = e - prev;
    const double df = std::exp(-nom_zero * e);  // pay at period end
    ins.t0.push_back(prev);
    ins.t1.push_back(e);
    ins.w_num.push_back(df);
    ins.w_den.push_back(df * tau);
    prev = e;
  }
  return ins;
}

// Regular annual YoY schedule out to an integer number of years (convenience wrapper).
inline InflationInstrument inflation_yoy_annual(int years, double par_rate, double nom_zero = 0.0) {
  std::vector<double> ends;
  for (int y = 1; y <= years; ++y) ends.push_back(static_cast<double>(y));
  return inflation_yoy(ends, par_rate, nom_zero);
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_INFLATION_INSTRUMENTS_HPP
