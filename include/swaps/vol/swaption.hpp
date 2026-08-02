#pragma once
// Swaption pricing off a discount curve: turn the underlying swap's discount factors into a forward swap
// rate S0 and annuity A0, then price with the Bachelier normal model (optionally with a SABR smile vol at
// the strike). Physically-settled European swaption under the annuity numeraire:
//   S0 = (DF(start) - DF(maturity)) / A0,   A0 = Σ τ_i · DF(pay_i)   (fixed leg),
//   V  = A0 · Bachelier(S0, K, σ_N, T_expiry).
// Pure math on discount factors — the api/ seam samples the calibrated SOFR curve to fill these in. Keeps
// the vol/ layer QuantLib-free and independent of the frozen calc kernel.
#include <vector>

#include "swaps/vol/bachelier.hpp"
#include "swaps/vol/sabr.hpp"

namespace swaps::vol {

struct ForwardSwap {
  double rate = 0.0;      // S0 — the forward par swap rate
  double annuity = 0.0;   // A0 = Σ τ_i DF_i (the swaption numeraire / PV01 of the fixed leg)
};

// S0, A0 from the underlying's discount factors: DF at the swap start and maturity, and the fixed-leg pay-date
// DFs with their accrual factors τ. (For an OIS swap the float leg telescopes, so DF(start)-DF(end) is exact.)
inline ForwardSwap forward_swap(double df_start, double df_end, const std::vector<double>& df_pay,
                                const std::vector<double>& tau) {
  double a = 0.0;
  for (std::size_t i = 0; i < df_pay.size() && i < tau.size(); ++i) a += tau[i] * df_pay[i];
  return {a > 0.0 ? (df_start - df_end) / a : 0.0, a};
}

// Price a European swaption from its forward/annuity + a normal vol. `expiry` in years.
inline double swaption_price(double s0, double a0, double strike, double normal_vol, double expiry,
                             Payoff cp) {
  return bachelier_price<double>(s0, strike, normal_vol, expiry, a0, cp);
}

// Convenience: price directly off a SABR smile — the normal vol is read at the strike, then Bachelier.
inline double swaption_price_sabr(double s0, double a0, double strike, double expiry, const SabrParams& sabr,
                                  Payoff cp) {
  const double vol = sabr_normal_vol(s0, strike, expiry, sabr);
  return bachelier_price<double>(s0, strike, vol, expiry, a0, cp);
}

}  // namespace swaps::vol
