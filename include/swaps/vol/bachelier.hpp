#pragma once
// Bachelier (NORMAL-vol) analytics — the foundation of the SOFR options layer. SOFR swaptions and caps are
// quoted in normal vol (basis points), so this is the base model (not Black/lognormal): it is exact at zero
// and negative strikes, it is the SABR calibration target, and it is the analytic control variate for every
// Monte-Carlo product later. All pricing is expressed on a forward rate F, strike K, normal vol σ, expiry T,
// and an ANNUITY (the numeraire: Σ τ_i·DF_i for a swaption, or τ·DF for a caplet) — so the same function
// prices a swaption or a caplet depending on what annuity you hand it.
//
// Payer swaption = call on the swap rate:  V = A·[ (F−K)·Φ(d) + σ√T·φ(d) ],  d = (F−K)/(σ√T).
// Receiver:                                 V = A·[ (K−F)·Φ(−d) + σ√T·φ(d) ].
// Put–call parity (a unit invariant):        V_pay − V_rec = A·(F − K).
//
// Scalar-templated so `double` prices and a reverse-AD type yields Greeks from the same code (the plan's
// price/Greeks duality). Closed-form vega/delta are provided too; implied-vol inverts on double via Newton.
#include <algorithm>
#include <cmath>

#include "swaps/vol/normal.hpp"

namespace swaps::vol {

enum class Payoff { Payer, Receiver };  // Payer = call on the rate; Receiver = put on the rate.

inline double payoff_sign(Payoff cp) { return cp == Payoff::Payer ? 1.0 : -1.0; }

// Normal (Bachelier) price. `expiry` in year-fractions; `fwd`, `strike`, `vol`, `annuity` in the Scalar type.
template <class S>
inline S bachelier_price(const S& fwd, const S& strike, const S& vol, double expiry, const S& annuity,
                         Payoff cp) {
  using std::sqrt;
  const S sgn(payoff_sign(cp));
  const S stddev = vol * S(sqrt(expiry));
  if (stddev <= S(0.0)) {  // zero vol / zero expiry -> discounted intrinsic
    const S intr = sgn * (fwd - strike);
    return annuity * (intr > S(0.0) ? intr : S(0.0));
  }
  const S d = sgn * (fwd - strike) / stddev;
  return annuity * (sgn * (fwd - strike) * normal_cdf(d) + stddev * normal_pdf(d));
}

// dV/dσ (per unit normal vol). Symmetric in payer/receiver.
template <class S>
inline S bachelier_vega(const S& fwd, const S& strike, const S& vol, double expiry, const S& annuity) {
  using std::sqrt;
  const S stddev = vol * S(sqrt(expiry));
  if (stddev <= S(0.0)) return S(0.0);
  const S d = (fwd - strike) / stddev;
  return annuity * S(sqrt(expiry)) * normal_pdf(d);
}

// dV/dF (sensitivity to the forward rate).
template <class S>
inline S bachelier_delta(const S& fwd, const S& strike, const S& vol, double expiry, const S& annuity,
                         Payoff cp) {
  using std::sqrt;
  const S sgn(payoff_sign(cp));
  const S stddev = vol * S(sqrt(expiry));
  if (stddev <= S(0.0)) return annuity * ((sgn * (fwd - strike) > S(0.0)) ? sgn : S(0.0));
  const S d = sgn * (fwd - strike) / stddev;
  return annuity * sgn * normal_cdf(d);
}

// d2V/dF2 (gamma). Symmetric in payer/receiver: A·φ(d)/(σ√T).
template <class S>
inline S bachelier_gamma(const S& fwd, const S& strike, const S& vol, double expiry, const S& annuity) {
  using std::sqrt;
  const S stddev = vol * S(sqrt(expiry));
  if (stddev <= S(0.0)) return S(0.0);
  const S d = (fwd - strike) / stddev;
  return annuity * normal_pdf(d) / stddev;
}

// Invert price -> normal vol (double). Price is strictly increasing in vol above intrinsic. A plain Newton
// from the ATM seed escapes the basin for deep-OTM strikes (tiny vega), so this is a SAFEGUARDED
// Newton-bisection: keep a [lo,hi] bracket, take the Newton step when it stays inside, bisect otherwise.
// Robust for all strikes. Returns 0 for a price at/below intrinsic.
inline double bachelier_implied_vol(double price, double fwd, double strike, double expiry, double annuity,
                                    Payoff cp, double tol = 1e-12, int max_iter = 100) {
  const double intrinsic = annuity * std::max(payoff_sign(cp) * (fwd - strike), 0.0);
  if (!(price > intrinsic) || annuity <= 0.0 || expiry <= 0.0) return 0.0;
  const auto price_at = [&](double vol) {
    return bachelier_price<double>(fwd, strike, vol, expiry, annuity, cp);
  };
  // Upper bracket: the ATM formula UNDER-estimates for OTM (a given price needs a higher vol there), so grow
  // `hi` until it over-prices the target; `lo=0` always under-prices (intrinsic).
  double lo = 0.0, hi = std::max(1e-6, (price / annuity) * std::sqrt(2.0 * M_PI / expiry));
  for (int i = 0; i < 64 && price_at(hi) < price; ++i) hi *= 2.0;
  double vol = 0.5 * (lo + hi);
  for (int i = 0; i < max_iter; ++i) {
    const double diff = price_at(vol) - price;
    if (diff > 0.0) hi = vol; else lo = vol;              // tighten the bracket
    const double vega = bachelier_vega<double>(fwd, strike, vol, expiry, annuity);
    double next = (vega > 1e-16) ? vol - diff / vega : 0.5 * (lo + hi);
    if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);  // Newton left the bracket -> bisect
    if (std::abs(next - vol) < tol) return next;
    vol = next;
  }
  return vol;
}

}  // namespace swaps::vol
