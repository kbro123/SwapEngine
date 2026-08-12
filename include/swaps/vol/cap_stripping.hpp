#pragma once
// Cap/floor pricing + caplet-vol STRIPPING (Phase A §1.3). A caplet is a one-period swaption: a European
// option on a single forward rate F with numeraire tau*DF_pay, priced by the same Bachelier normal model as
// the swaption layer (vol/bachelier.hpp). A cap/floor is the sum of its caplets/floorlets.
//
// Desks quote caps by a single FLAT vol per maturity (the vol that, applied to every caplet, reprices the
// cap). "Stripping" bootstraps the underlying per-caplet (spot) vols from that term structure of flat cap
// vols: piecewise-constant between quoted maturities, each bucket solved so the cap through that maturity
// reprices — the standard forward-vol bootstrap, and the input a smile/surface (§1.2) is then built on.
//
// Units are the model's decimals (rates absolute, vols normal/absolute), matching the rest of vol/.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "swaps/vol/bachelier.hpp"

namespace swaps::vol {

// One caplet's curve data: the forward rate it options, its accrual tau, the pay-date discount factor, and the
// option expiry (fixing time) in years. Strike is a cap-level input (shared across the cap's caplets).
struct CapletLeg {
  double forward = 0.0;
  double tau = 0.0;
  double df_pay = 0.0;
  double expiry = 0.0;
};

// A caplet (Payer = call on the rate) / floorlet (Receiver) at `strike` and normal `vol`. Numeraire tau*DF.
inline double caplet_price(const CapletLeg& L, double strike, double vol, Payoff cp = Payoff::Payer) {
  return bachelier_price<double>(L.forward, strike, vol, L.expiry, L.tau * L.df_pay, cp);
}

// A cap/floor priced with a per-caplet vol vector.
inline double cap_price(const std::vector<CapletLeg>& legs, double strike, const std::vector<double>& vols,
                        Payoff cp = Payoff::Payer) {
  double s = 0.0;
  for (std::size_t i = 0; i < legs.size(); ++i) s += caplet_price(legs[i], strike, vols[i], cp);
  return s;
}

// A cap/floor priced with ONE flat vol applied to every caplet (the market quote convention).
inline double cap_price_flat(const std::vector<CapletLeg>& legs, double strike, double flat_vol,
                             Payoff cp = Payoff::Payer) {
  double s = 0.0;
  for (const CapletLeg& L : legs) s += caplet_price(L, strike, flat_vol, cp);
  return s;
}

// Invert a cap price to its flat normal vol (monotone increasing in vol; safeguarded bisection). Returns 0 for
// a price at/below the cap's total intrinsic.
inline double cap_implied_flat_vol(const std::vector<CapletLeg>& legs, double strike, double price,
                                   Payoff cp = Payoff::Payer, double tol = 1e-12, int max_iter = 100) {
  double intrinsic = 0.0;
  for (const CapletLeg& L : legs)
    intrinsic += L.tau * L.df_pay * std::max(payoff_sign(cp) * (L.forward - strike), 0.0);
  if (!(price > intrinsic)) return 0.0;
  double lo = 0.0, hi = 0.02;
  for (int i = 0; i < 64 && cap_price_flat(legs, strike, hi, cp) < price; ++i) hi *= 2.0;
  for (int i = 0; i < max_iter; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (hi - lo < tol) return mid;
    (cap_price_flat(legs, strike, mid, cp) < price ? lo : hi) = mid;
  }
  return 0.5 * (lo + hi);
}

// Strip PIECEWISE-CONSTANT per-caplet vols from a term structure of FLAT cap vols. `cap_last_leg[j]` is the
// index of the last caplet in the j-th quoted cap (strictly increasing); `cap_flat_vols[j]` its flat vol. The
// caplets in bucket j (after the previous cap's last leg, through cap_last_leg[j]) share one stripped vol,
// solved so the cap through cap_last_leg[j] reprices at cap_flat_vols[j]. Returns a vol per caplet.
inline std::vector<double> strip_caplet_vols(const std::vector<CapletLeg>& legs,
                                             double strike,
                                             const std::vector<int>& cap_last_leg,
                                             const std::vector<double>& cap_flat_vols,
                                             Payoff cp = Payoff::Payer) {
  if (cap_last_leg.size() != cap_flat_vols.size())
    throw std::invalid_argument("strip_caplet_vols: cap_last_leg and cap_flat_vols length mismatch");
  std::vector<double> vols(legs.size(), 0.0);
  int prev = 0;
  double fixed = 0.0;  // running PV of the already-stripped prefix [0..prev-1] — accumulated, not re-summed
  for (std::size_t j = 0; j < cap_last_leg.size(); ++j) {
    const int e = cap_last_leg[j];
    if (e < prev || e >= static_cast<int>(legs.size()))
      throw std::invalid_argument("strip_caplet_vols: cap_last_leg must be increasing and in range");
    // Target = the whole cap [0..e] priced at its flat vol (summed in place — no per-bucket vector copy).
    // The [0..prev-1] part is already stripped, so its PV is carried in `fixed` (O(1)) instead of re-summed.
    double target = 0.0;
    for (int i = 0; i <= e; ++i) target += caplet_price(legs[i], strike, cap_flat_vols[j], cp);
    // Solve the bucket vol v so sum_{prev..e} caplet(v) = target - fixed (monotone increasing in v).
    const double residual = target - fixed;
    const auto bucket = [&](double v) {
      double s = 0.0;
      for (int i = prev; i <= e; ++i) s += caplet_price(legs[i], strike, v, cp);
      return s;
    };
    double lo = 0.0, hi = std::max(cap_flat_vols[j], 1e-6);
    for (int i = 0; i < 64 && bucket(hi) < residual; ++i) hi *= 2.0;
    for (int i = 0; i < 100; ++i) {
      const double mid = 0.5 * (lo + hi);
      if (hi - lo < 1e-13) { lo = hi = mid; break; }
      (bucket(mid) < residual ? lo : hi) = mid;
    }
    const double v = 0.5 * (lo + hi);
    // Set this bucket's vols AND fold its PV into the running prefix (same caplet(v) evals the old re-sum
    // would repeat next iteration — so `fixed` stays byte-for-byte Σ_{0..prev-1} caplet(stripped_i)).
    for (int i = prev; i <= e; ++i) { vols[i] = v; fixed += caplet_price(legs[i], strike, v, cp); }
    prev = e + 1;
  }
  return vols;
}

}  // namespace swaps::vol
