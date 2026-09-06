#pragma once
// Garman-Kohlhagen / Black-Scholes LOGNORMAL vanilla FX option — the engine's first lognormal model
// (vol/bachelier.hpp is NORMAL, for rates). FX vanillas are quoted lognormal (Black) on a spot S with TWO
// rates: domestic r_d (the payoff/numeraire currency) and foreign r_f (the asset currency; a continuous
// "dividend yield" on the foreign unit). The forward is F = S·e^{(r_d−r_f)T} and every price is the Black
// forward price discounted at the DOMESTIC rate:
//
//   d1 = [ ln(F/K) + ½σ²T ] / (σ√T),   d2 = d1 − σ√T
//   Call = e^{−r_d T} [ F·Φ(d1) − K·Φ(d2) ] = S·e^{−r_f T}Φ(d1) − K·e^{−r_d T}Φ(d2)
//   Put  = e^{−r_d T} [ K·Φ(−d2) − F·Φ(−d1) ]
//   Put–call parity:  Call − Put = e^{−r_d T}(F − K) = S·e^{−r_f T} − K·e^{−r_d T}.
//
// Prices are per ONE unit of FOREIGN notional (a EURUSD call on 1 EUR), i.e. in domestic currency per
// foreign unit. Greeks: spot delta/gamma (∂/∂S), vega (∂/∂σ, per 1.00 = 100 vol points), theta (∂/∂t =
// −∂/∂T, per year), and BOTH rate rhos (∂/∂r_d, ∂/∂r_f) — the two-rate structure that distinguishes FX
// from single-curve Black. Scalar-templated so `double` prices and a reverse-AD type (ad::Dual)
// differentiates the same closed forms; implied vol inverts on double via a safeguarded Newton-bisection.
// QuantLib-free; header-only; consumes only vol/normal.hpp.
#include <algorithm>
#include <cmath>

#include "swaps/vol/normal.hpp"

namespace swaps::vol {

enum class CallPut { Call, Put };  // FX vanilla right (call/put ON the FX rate, foreign per domestic quote).

inline double cp_sign(CallPut cp) { return cp == CallPut::Call ? 1.0 : -1.0; }

// Forward under Garman-Kohlhagen: F = S·e^{(r_d−r_f)T}.
template <class S>
inline S gk_forward(const S& spot, double expiry, const S& r_dom, const S& r_for) {
  using std::exp;
  return spot * exp((r_dom - r_for) * S(expiry));
}

// ------------------------------------------------------------------------------------------------------
// FORWARD form — the numeraire-agnostic core. Black price off a forward F and a DOMESTIC discount factor
// df_dom = e^{−r_d T}. Reused by the spot form below and by the surface's strike<->delta conversions.
template <class S>
inline S black_price_fwd(const S& fwd, const S& strike, const S& vol, double expiry, const S& df_dom,
                         CallPut cp) {
  using std::log;
  using std::sqrt;
  const S sgn(cp_sign(cp));
  const S stddev = vol * S(sqrt(expiry));
  if (stddev <= S(0.0)) {  // zero vol / zero expiry -> discounted intrinsic on the forward
    const S intr = sgn * (fwd - strike);
    return df_dom * (intr > S(0.0) ? intr : S(0.0));
  }
  const S d1 = (log(fwd / strike) + S(0.5) * stddev * stddev) / stddev;
  const S d2 = d1 - stddev;
  return df_dom * sgn * (fwd * normal_cdf(sgn * d1) - strike * normal_cdf(sgn * d2));
}

// ------------------------------------------------------------------------------------------------------
// SPOT form (Garman-Kohlhagen). Price per unit foreign notional, in domestic currency.
template <class S>
inline S gk_price(const S& spot, const S& strike, const S& vol, double expiry, const S& r_dom,
                  const S& r_for, CallPut cp) {
  using std::exp;
  const S df_dom = exp(-r_dom * S(expiry));
  const S fwd = gk_forward(spot, expiry, r_dom, r_for);
  return black_price_fwd(fwd, strike, vol, expiry, df_dom, cp);
}

// Spot delta ∂V/∂S (UNADJUSTED / not premium-adjusted): call e^{−r_f T}Φ(d1), put −e^{−r_f T}Φ(−d1).
template <class S>
inline S gk_delta(const S& spot, const S& strike, const S& vol, double expiry, const S& r_dom,
                  const S& r_for, CallPut cp) {
  using std::exp;
  using std::log;
  using std::sqrt;
  const S sgn(cp_sign(cp));
  const S df_for = exp(-r_for * S(expiry));
  const S stddev = vol * S(sqrt(expiry));
  const S fwd = gk_forward(spot, expiry, r_dom, r_for);
  if (stddev <= S(0.0)) return df_for * sgn * ((sgn * (fwd - strike) > S(0.0)) ? S(1.0) : S(0.0));
  const S d1 = (log(fwd / strike) + S(0.5) * stddev * stddev) / stddev;
  return df_for * sgn * normal_cdf(sgn * d1);
}

// Gamma ∂²V/∂S² = e^{−r_f T}φ(d1)/(S σ√T). Symmetric in call/put.
template <class S>
inline S gk_gamma(const S& spot, const S& strike, const S& vol, double expiry, const S& r_dom,
                  const S& r_for) {
  using std::exp;
  using std::log;
  using std::sqrt;
  const S df_for = exp(-r_for * S(expiry));
  const S stddev = vol * S(sqrt(expiry));
  if (stddev <= S(0.0)) return S(0.0);
  const S fwd = gk_forward(spot, expiry, r_dom, r_for);
  const S d1 = (log(fwd / strike) + S(0.5) * stddev * stddev) / stddev;
  return df_for * normal_pdf(d1) / (spot * stddev);
}

// Vega ∂V/∂σ = S·e^{−r_f T}φ(d1)√T (per unit vol, i.e. per 1.00 = 100 vol points). Symmetric in call/put.
template <class S>
inline S gk_vega(const S& spot, const S& strike, const S& vol, double expiry, const S& r_dom,
                 const S& r_for) {
  using std::exp;
  using std::log;
  using std::sqrt;
  const S df_for = exp(-r_for * S(expiry));
  const S sqrtT(sqrt(expiry));
  const S stddev = vol * sqrtT;
  if (stddev <= S(0.0)) return S(0.0);
  const S fwd = gk_forward(spot, expiry, r_dom, r_for);
  const S d1 = (log(fwd / strike) + S(0.5) * stddev * stddev) / stddev;
  return spot * df_for * normal_pdf(d1) * sqrtT;
}

// Theta ∂V/∂t = −∂V/∂T (per year): time decay carrying BOTH rate carries.
//   Call: −S·e^{−r_f T}φ(d1)σ/(2√T) + r_f·S·e^{−r_f T}Φ(d1) − r_d·K·e^{−r_d T}Φ(d2)
//   Put : −S·e^{−r_f T}φ(d1)σ/(2√T) − r_f·S·e^{−r_f T}Φ(−d1) + r_d·K·e^{−r_d T}Φ(−d2)
template <class S>
inline S gk_theta(const S& spot, const S& strike, const S& vol, double expiry, const S& r_dom,
                  const S& r_for, CallPut cp) {
  using std::exp;
  using std::log;
  using std::sqrt;
  const S sgn(cp_sign(cp));
  const S df_for = exp(-r_for * S(expiry));
  const S df_dom = exp(-r_dom * S(expiry));
  const S sqrtT(sqrt(expiry));
  const S stddev = vol * sqrtT;
  const S fwd = gk_forward(spot, expiry, r_dom, r_for);
  if (stddev <= S(0.0)) return S(0.0);
  const S d1 = (log(fwd / strike) + S(0.5) * stddev * stddev) / stddev;
  const S d2 = d1 - stddev;
  const S decay = -spot * df_for * normal_pdf(d1) * vol / (S(2.0) * sqrtT);
  return decay + sgn * r_for * spot * df_for * normal_cdf(sgn * d1) -
         sgn * r_dom * strike * df_dom * normal_cdf(sgn * d2);
}

// Domestic rho ∂V/∂r_d = ±K·T·e^{−r_d T}Φ(±d2)  (+ for call, − for put).
template <class S>
inline S gk_rho_dom(const S& spot, const S& strike, const S& vol, double expiry, const S& r_dom,
                    const S& r_for, CallPut cp) {
  using std::exp;
  using std::log;
  using std::sqrt;
  const S sgn(cp_sign(cp));
  const S df_dom = exp(-r_dom * S(expiry));
  const S stddev = vol * S(sqrt(expiry));
  const S fwd = gk_forward(spot, expiry, r_dom, r_for);
  if (stddev <= S(0.0)) {  // discounted-intrinsic limit: ∂/∂r_d of df_dom·max(sgn(F−K),0), F is r_d-dependent
    const S intr = sgn * (fwd - strike);
    return (intr > S(0.0)) ? S(expiry) * df_dom * (sgn * fwd - intr) : S(0.0);
  }
  const S d2 = (log(fwd / strike) - S(0.5) * stddev * stddev) / stddev;
  return sgn * strike * S(expiry) * df_dom * normal_cdf(sgn * d2);
}

// Foreign rho ∂V/∂r_f = ∓S·T·e^{−r_f T}Φ(±d1)  (− for call, + for put).
template <class S>
inline S gk_rho_for(const S& spot, const S& strike, const S& vol, double expiry, const S& r_dom,
                    const S& r_for, CallPut cp) {
  using std::exp;
  using std::log;
  using std::sqrt;
  const S sgn(cp_sign(cp));
  const S df_for = exp(-r_for * S(expiry));
  const S stddev = vol * S(sqrt(expiry));
  const S fwd = gk_forward(spot, expiry, r_dom, r_for);
  if (stddev <= S(0.0)) {
    const S intr = sgn * (fwd - strike);
    return (intr > S(0.0)) ? -S(expiry) * spot * df_for * sgn : S(0.0);
  }
  const S d1 = (log(fwd / strike) + S(0.5) * stddev * stddev) / stddev;
  return -sgn * spot * S(expiry) * df_for * normal_cdf(sgn * d1);
}

// ALL SIX Greeks (+ price) in ONE pass — the FX vol-cube hot path (analogous to bachelier_greeks). Shares
// one sqrt, one φ(d1), and the Φ(±d1)/Φ(±d2) evaluations across every output. Values are IDENTICAL to the
// individual analytics above (pinned by a consistency test).
template <class S>
struct GkGreeks {
  S price, delta, gamma, vega, theta, rho_dom, rho_for;
};

template <class S>
inline GkGreeks<S> gk_greeks(const S& spot, const S& strike, const S& vol, double expiry, const S& r_dom,
                             const S& r_for, CallPut cp) {
  using std::exp;
  using std::log;
  using std::sqrt;
  const S sgn(cp_sign(cp));
  const S df_dom = exp(-r_dom * S(expiry));
  const S df_for = exp(-r_for * S(expiry));
  const S sqrtT(sqrt(expiry));
  const S stddev = vol * sqrtT;
  const S fwd = spot * exp((r_dom - r_for) * S(expiry));
  GkGreeks<S> g;
  if (stddev <= S(0.0)) {  // zero vol / zero expiry -> discounted intrinsic; smooth Greeks vanish
    const S intr = sgn * (fwd - strike);
    const S pos = (intr > S(0.0)) ? S(1.0) : S(0.0);
    g.price = df_dom * intr * pos;
    g.delta = df_for * sgn * pos;
    g.gamma = S(0.0);
    g.vega = S(0.0);
    g.theta = S(0.0);
    g.rho_dom = (intr > S(0.0)) ? S(expiry) * df_dom * (sgn * fwd - intr) : S(0.0);
    g.rho_for = (intr > S(0.0)) ? -S(expiry) * spot * df_for * sgn : S(0.0);
    return g;
  }
  const S d1 = (log(fwd / strike) + S(0.5) * stddev * stddev) / stddev;
  const S d2 = d1 - stddev;
  const S phi1 = normal_pdf(d1);
  const S Nsd1 = normal_cdf(sgn * d1);
  const S Nsd2 = normal_cdf(sgn * d2);
  g.price = df_dom * sgn * (fwd * Nsd1 - strike * Nsd2);
  g.delta = df_for * sgn * Nsd1;
  g.gamma = df_for * phi1 / (spot * stddev);
  g.vega = spot * df_for * phi1 * sqrtT;
  g.theta = -spot * df_for * phi1 * vol / (S(2.0) * sqrtT) + sgn * r_for * spot * df_for * Nsd1 -
            sgn * r_dom * strike * df_dom * Nsd2;
  g.rho_dom = sgn * strike * S(expiry) * df_dom * Nsd2;
  g.rho_for = -sgn * spot * S(expiry) * df_for * Nsd1;
  return g;
}

// Invert price -> Black vol (double). Price is strictly increasing in vol above discounted intrinsic. A
// plain Newton from an ATM seed can escape the basin on deep wings (tiny vega), so this is a SAFEGUARDED
// Newton-bisection: keep a [lo,hi] bracket, Newton step when it stays inside, bisect otherwise. Returns 0
// for a price at/below intrinsic.
inline double gk_implied_vol(double price, double spot, double strike, double expiry, double r_dom,
                             double r_for, CallPut cp, double tol = 1e-12, int max_iter = 100) {
  const double df_dom = std::exp(-r_dom * expiry);
  const double fwd = spot * std::exp((r_dom - r_for) * expiry);
  const double intrinsic = df_dom * std::max(cp_sign(cp) * (fwd - strike), 0.0);
  if (!(price > intrinsic) || expiry <= 0.0) return 0.0;
  const auto price_at = [&](double vol) {
    return gk_price<double>(spot, strike, vol, expiry, r_dom, r_for, cp);
  };
  double lo = 0.0, hi = 1.0;  // 100% vol is a generous upper seed; grow until it over-prices the target
  for (int i = 0; i < 64 && price_at(hi) < price; ++i) hi *= 2.0;
  double vol = 0.5 * (lo + hi);
  for (int i = 0; i < max_iter; ++i) {
    const double diff = price_at(vol) - price;
    if (diff > 0.0) hi = vol; else lo = vol;
    const double vega = gk_vega<double>(spot, strike, vol, expiry, r_dom, r_for);
    double next = (vega > 1e-16) ? vol - diff / vega : 0.5 * (lo + hi);
    if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);
    if (std::abs(next - vol) < tol) return next;
    vol = next;
  }
  return vol;
}

}  // namespace swaps::vol
