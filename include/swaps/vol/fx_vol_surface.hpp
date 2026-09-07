#pragma once
// FX volatility SMILE, quoted the FX way — by DELTA, not by strike. The interbank market gives, per expiry,
// {ATM, 25d risk-reversal, 25d butterfly [, 10d RR, 10d BF]} in vol terms; this header turns those into
// (strike, vol) knots and interpolates a smooth smile in LOG-MONEYNESS x = ln(K/F).
//
//   σ(25Δ call) = ATM + BF25 + ½·RR25,   σ(25Δ put) = ATM + BF25 − ½·RR25   (the "smile"/quoted-vol BF).
//
// DELTA CONVENTION: spot delta, UNADJUSTED (not premium-adjusted) — the standard for short-dated majors.
//   Δ_call = e^{−r_f T}Φ(d1),  Δ_put = −e^{−r_f T}Φ(−d1),  d1 = [ln(F/K)+½σ²T]/(σ√T).
//   Given a target |Δ| and a vol σ:  d1 = ±Φ⁻¹(|Δ|·e^{r_f T})  (+ call, − put),
//                                    K  = F·exp(−d1·σ√T + ½σ²T).
//   ATM = DELTA-NEUTRAL STRADDLE (Δ_call+Δ_put=0 ⇒ d1=0):  K_ATM = F·exp(½·σ_ATM²·T).
//
// ALL FOUR delta conventions {spot,forward}×{unadjusted,premium-adjusted} are supported (see DeltaConv in
// fx_black.hpp), plus two ATM conventions (AtmConv): DELTA-NEUTRAL STRADDLE (default) and ATM-FORWARD (K=F).
// The unadjusted conventions invert monotonically (direct Φ⁻¹). PREMIUM-ADJUSTED delta is NON-MONOTONE in
// strike (|Δ| = disc·(K/F)·N(±d2) has an interior maximum for a call → TWO strikes share one |Δ|): we locate
// the maximum, then bisect on each branch and return the MARKET strike = the root with the smaller
// |log-moneyness| (closer to the forward = the standard OTM branch). SpotUnadj + DNS is the documented
// default, byte-identical to the historical single-convention path. To keep the smile arbitrage-aware the
// strike interp is a monotone PCHIP — no overshoot between knots — with flat extrapolation past the wings.
//
// The strike<->delta conversion runs on double (it builds the grid); the interpolated vol feeds
// vol/fx_black.hpp (Scalar-templated) for pricing/Greeks. QuantLib-free; header-only.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "swaps/vol/fx_black.hpp"
#include "swaps/vol/normal.hpp"

namespace swaps::vol {

// Inverse standard-normal CDF (Acklam's rational approximation, |abs err| < 1.15e-9). Double-only: used to
// place strikes from deltas, not to differentiate through.
inline double norm_inv(double p) {
  static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                             1.383577518672690e+02,  -3.066479806614716e+01, 2.506628277459239e+00};
  static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                             6.680131188771972e+01,  -1.328068155288572e+01};
  static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                             -2.549732539343734e+00, 4.374664141464968e+00,  2.938163982698783e+00};
  static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                             3.754408661907416e+00};
  const double plow = 0.02425, phigh = 1.0 - plow;
  if (p <= 0.0) return -HUGE_VAL;
  if (p >= 1.0) return HUGE_VAL;
  double x;
  if (p < plow) {
    const double q = std::sqrt(-2.0 * std::log(p));
    x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  } else if (p <= phigh) {
    const double q = p - 0.5, r = q * q;
    x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
  } else {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
  }
  return x;
}

// The two ATM-strike conventions. DeltaNeutral (delta-neutral straddle) is the historical default.
enum class AtmConv { DeltaNeutral, Forward };

// ATM strike under an AtmConv. DNS solves Δ_call+Δ_put=0: unadjusted ⇒ d1=0 ⇒ K=F·e^{+½σ²T};
// premium-adjusted ⇒ d2=0 ⇒ K=F·e^{−½σ²T}. ATM-forward is simply K=F. (`premium_adjusted` only matters for
// DNS.) The unadjusted-DNS expression is written to be byte-identical to the historical F·e^{½σ²T} literal.
inline double fx_atm_strike(double forward, double expiry, double vol, AtmConv atm, bool premium_adjusted) {
  if (atm == AtmConv::Forward) return forward;
  const double half = premium_adjusted ? -0.5 : 0.5;
  return forward * std::exp(half * vol * vol * expiry);
}

// Strike at a target delta magnitude |Δ|∈(0,1) under `conv`, given forward F, expiry T, df_for = e^{−r_f T},
// vol σ. UNADJUSTED conventions invert directly (|Δ| = disc·N(sgn·d1), disc = df_for spot / 1 forward, so
// d1 = sgn·Φ⁻¹(|Δ|/disc), K = F·exp(−d1·σ√T + ½σ²T)). The SpotUnadj path (default) is byte-identical to the
// historical formula. PREMIUM-ADJUSTED (|Δ| = disc·(K/F)·N(sgn·d2)) is non-monotone in K for a call: we solve
// h(x) = e^x·N(sgn·d2) = |Δ|/disc for x = ln(K/F). We first locate the interior maximum x* (bracketed sign
// change of h′; a put has none — h is monotone), then bisect each monotone branch and return the MARKET
// strike = the root with the smaller |x| (log-moneyness closer to the forward, i.e. the standard OTM branch).
inline double fx_strike_from_delta(double forward, double expiry, double df_for, double vol, double delta_mag,
                                   CallPut cp, DeltaConv conv = DeltaConv::SpotUnadj) {
  if (!(expiry > 0.0) || !(vol > 0.0)) throw std::invalid_argument("fx_strike_from_delta: need T,σ > 0");
  const double stddev = vol * std::sqrt(expiry);
  const double sgn = cp_sign(cp);
  const double disc = delta_is_spot(conv) ? df_for : 1.0;

  if (!delta_is_pa(conv)) {  // monotone: direct inversion (SpotUnadj reproduces the historical formula)
    const double d1 = sgn * norm_inv(delta_mag / disc);
    return forward * std::exp(-d1 * stddev + 0.5 * stddev * stddev);
  }

  // Premium-adjusted dual-strike solve. h(x) = e^x·N(sgn·d2), d2 = (−x − ½σ²T)/(σ√T); solve h(x) = target.
  const double target = delta_mag / disc;
  const auto d2 = [&](double x) { return (-x - 0.5 * stddev * stddev) / stddev; };
  const auto h = [&](double x) { return std::exp(x) * normal_cdf(sgn * d2(x)); };
  // sign of h′(x)/e^x = N(sgn·d2) − sgn·φ(d2)/(σ√T).
  const auto hprime = [&](double x) { return normal_cdf(sgn * d2(x)) - sgn * normal_pdf(d2(x)) / stddev; };
  const double R = std::max(1.0, 12.0 * stddev) + 2.0;  // wide log-moneyness bracket
  const double xlo = -R, xhi = R;
  const auto bisect = [&](double a, double b) {  // solve h=target on a monotone, sign-bracketed [a,b]
    double fa = h(a) - target;
    for (int i = 0; i < 200; ++i) {
      const double m = 0.5 * (a + b);
      const double fm = h(m) - target;
      if (std::abs(fm) < 1e-15 || (b - a) < 1e-15) return m;
      if ((fa < 0.0) == (fm < 0.0)) { a = m; fa = fm; } else { b = m; }
    }
    return 0.5 * (a + b);
  };

  if ((hprime(xlo) < 0.0) == (hprime(xhi) < 0.0))  // monotone (put): a single root on the full range
    return forward * std::exp(bisect(xlo, xhi));

  // Interior maximum (call): bisect h′ for x*, then solve each branch and keep the lower-|moneyness| root.
  double a = xlo, b = xhi;
  const bool inc_lo = hprime(a) > 0.0;
  for (int i = 0; i < 200 && (b - a) > 1e-14; ++i) {
    const double m = 0.5 * (a + b);
    if ((hprime(m) > 0.0) == inc_lo) a = m; else b = m;
  }
  const double xstar = 0.5 * (a + b);
  if (!(h(xstar) > target)) return forward * std::exp(xstar);  // |Δ| above the achievable max → max strike
  const double xL = bisect(xlo, xstar);   // increasing branch (deep side)
  const double xR = bisect(xstar, xhi);   // decreasing branch (OTM side)
  const double x = (std::abs(xL) <= std::abs(xR)) ? xL : xR;
  return forward * std::exp(x);
}

// The market delta quotes for ONE expiry (vol terms, absolute e.g. 0.11 = 11%). rr/bf are the standard
// risk-reversal / (smile) butterfly. The 10d wing is optional.
struct FxDeltaQuotes {
  double atm = 0.0;
  double rr25 = 0.0;
  double bf25 = 0.0;
  bool has10 = false;
  double rr10 = 0.0;
  double bf10 = 0.0;
};

// A single-expiry FX smile: (strike, vol) knots interpolated monotone-PCHIP in log-moneyness x = ln(K/F),
// flat past the wings. Holds F, T, df_for so it can convert delta<->strike for queries.
class FxVolSurface {
 public:
  // Flat smile — one knot; vol() is constant. df_for defaults to 1 (rates irrelevant for a flat vol).
  static FxVolSurface flat(double forward, double expiry, double vol, double df_for = 1.0) {
    FxVolSurface s;
    s.forward_ = forward;
    s.expiry_ = expiry;
    s.df_for_ = df_for;
    s.x_ = {0.0};
    s.vol_ = {vol};
    s.build_slopes();
    return s;
  }

  // Build from interbank delta quotes. Places the ATM knot (per AtmConv) + 25d (and optional 10d) wings under
  // the chosen delta convention. Defaults (SpotUnadj + DeltaNeutral) reproduce the historical smile exactly.
  static FxVolSurface from_delta_quotes(const FxDeltaQuotes& q, double forward, double expiry, double df_for,
                                        DeltaConv delta_conv = DeltaConv::SpotUnadj,
                                        AtmConv atm_conv = AtmConv::DeltaNeutral) {
    if (!(forward > 0.0) || !(expiry > 0.0)) throw std::invalid_argument("FxVolSurface: need F,T > 0");
    FxVolSurface s;
    s.forward_ = forward;
    s.expiry_ = expiry;
    s.df_for_ = df_for;
    std::vector<std::pair<double, double>> pts;  // (x = ln(K/F), vol)
    const auto add = [&](double K, double vol) { pts.emplace_back(std::log(K / forward), vol); };

    const double k_atm = fx_atm_strike(forward, expiry, q.atm, atm_conv, delta_is_pa(delta_conv));
    add(k_atm, q.atm);
    const double v25c = q.atm + q.bf25 + 0.5 * q.rr25;
    const double v25p = q.atm + q.bf25 - 0.5 * q.rr25;
    add(fx_strike_from_delta(forward, expiry, df_for, v25c, 0.25, CallPut::Call, delta_conv), v25c);
    add(fx_strike_from_delta(forward, expiry, df_for, v25p, 0.25, CallPut::Put, delta_conv), v25p);
    if (q.has10) {
      const double v10c = q.atm + q.bf10 + 0.5 * q.rr10;
      const double v10p = q.atm + q.bf10 - 0.5 * q.rr10;
      add(fx_strike_from_delta(forward, expiry, df_for, v10c, 0.10, CallPut::Call, delta_conv), v10c);
      add(fx_strike_from_delta(forward, expiry, df_for, v10p, 0.10, CallPut::Put, delta_conv), v10p);
    }
    std::sort(pts.begin(), pts.end());
    for (const auto& p : pts) {
      s.x_.push_back(p.first);
      s.vol_.push_back(p.second);
    }
    s.build_slopes();
    return s;
  }

  // Build directly from (strike, vol) knots (already absolute). Strikes need not be sorted.
  static FxVolSurface from_strike_vols(std::vector<double> strikes, std::vector<double> vols, double forward,
                                       double expiry, double df_for = 1.0) {
    if (strikes.size() != vols.size() || strikes.empty())
      throw std::invalid_argument("FxVolSurface: non-empty, equal-length strike/vol knots required");
    std::vector<std::pair<double, double>> pts;
    for (std::size_t i = 0; i < strikes.size(); ++i) pts.emplace_back(std::log(strikes[i] / forward), vols[i]);
    std::sort(pts.begin(), pts.end());
    FxVolSurface s;
    s.forward_ = forward;
    s.expiry_ = expiry;
    s.df_for_ = df_for;
    for (const auto& p : pts) {
      s.x_.push_back(p.first);
      s.vol_.push_back(p.second);
    }
    s.build_slopes();
    return s;
  }

  double forward() const { return forward_; }
  double expiry() const { return expiry_; }
  double df_for() const { return df_for_; }

  // Vol at log-moneyness x = ln(K/F) — monotone-PCHIP, flat outside the knot range.
  double vol_at_logm(double x) const {
    const std::size_t n = x_.size();
    if (n == 1) return vol_[0];
    if (x <= x_.front()) return vol_.front();
    if (x >= x_.back()) return vol_.back();
    const std::size_t j = static_cast<std::size_t>(std::upper_bound(x_.begin(), x_.end(), x) - x_.begin()) - 1;
    const double h = x_[j + 1] - x_[j];
    const double t = (x - x_[j]) / h;
    const double h00 = (1.0 + 2.0 * t) * (1.0 - t) * (1.0 - t);
    const double h10 = t * (1.0 - t) * (1.0 - t);
    const double h01 = t * t * (3.0 - 2.0 * t);
    const double h11 = t * t * (t - 1.0);
    return h00 * vol_[j] + h10 * h * m_[j] + h01 * vol_[j + 1] + h11 * h * m_[j + 1];
  }

  double vol_at_strike(double strike) const { return vol_at_logm(std::log(strike / forward_)); }

  // Strike at a target spot-delta magnitude |Δ| under THIS smile (vol depends on strike -> a short fixed
  // point on the smile vol; a handful of iterations to convergence).
  double strike_for_delta(double delta_mag, CallPut cp, DeltaConv conv = DeltaConv::SpotUnadj, int iters = 32,
                          double tol = 1e-12) const {
    double vol = vol_at_logm(0.0);  // seed with the ATM-ish vol
    double K = fx_strike_from_delta(forward_, expiry_, df_for_, vol, delta_mag, cp, conv);
    for (int i = 0; i < iters; ++i) {
      const double vnew = vol_at_strike(K);
      const double Knew = fx_strike_from_delta(forward_, expiry_, df_for_, vnew, delta_mag, cp, conv);
      if (std::abs(Knew - K) <= tol * K) return Knew;
      vol = vnew;
      K = Knew;
    }
    return K;
  }

  const std::vector<double>& logm_knots() const { return x_; }
  const std::vector<double>& vol_knots() const { return vol_; }

 private:
  double forward_ = 0.0, expiry_ = 0.0, df_for_ = 1.0;
  std::vector<double> x_, vol_, m_;  // x_ = ln(K/F) ascending; m_ = PCHIP slopes

  // Fritsch-Carlson monotone-safe slopes (shape-preserving; no overshoot between knots).
  void build_slopes() {
    const std::size_t n = x_.size();
    m_.assign(n, 0.0);
    if (n < 2) return;
    std::vector<double> delta(n - 1);
    for (std::size_t i = 0; i + 1 < n; ++i) delta[i] = (vol_[i + 1] - vol_[i]) / (x_[i + 1] - x_[i]);
    m_[0] = delta[0];
    m_[n - 1] = delta[n - 2];
    for (std::size_t i = 1; i + 1 < n; ++i)
      m_[i] = (delta[i - 1] * delta[i] <= 0.0) ? 0.0 : 0.5 * (delta[i - 1] + delta[i]);
    for (std::size_t i = 0; i + 1 < n; ++i) {
      if (delta[i] == 0.0) {
        m_[i] = m_[i + 1] = 0.0;
        continue;
      }
      const double a = m_[i] / delta[i], b = m_[i + 1] / delta[i];
      const double s = a * a + b * b;
      if (s > 9.0) {
        const double tau = 3.0 / std::sqrt(s);
        m_[i] = tau * a * delta[i];
        m_[i + 1] = tau * b * delta[i];
      }
    }
  }
};

// ------------------------------------------------------------------------------------------------------
// FxOption — a CURVE-INDEPENDENT vanilla definition: pair, strike OR delta, expiry, call/put, notional.
struct FxOption {
  std::string pair;             // e.g. "EURUSD" (informational; the market carries spot/rates)
  double strike = 0.0;          // absolute strike (used when by_delta == false)
  double delta = 0.0;           // spot-delta magnitude in (0,1) (used when by_delta == true)
  double expiry = 0.0;          // years (informational; the surface carries the pricing T)
  CallPut cp = CallPut::Call;
  double notional = 1.0;        // foreign units
  bool by_delta = false;        // resolve `strike` from `delta` against the smile
  DeltaConv delta_conv = DeltaConv::SpotUnadj;  // convention used when by_delta resolves the strike
};

// The market a surface prices against: spot + the two continuous rates.
struct FxSurfaceMarket {
  double spot = 0.0;
  double r_dom = 0.0;
  double r_for = 0.0;
};

// Flat SoA batch result — one row per option (analogous to vol/VolCube). Greeks per unit foreign notional
// plus notional_price = price·notional.
struct FxVolCube {
  int n = 0;
  std::vector<double> strike, vol, forward, price, delta, gamma, vega, theta, rho_dom, rho_for, notional_price;
  double price_us = 0.0;
};

// One-pass reprice of a book of FxOptions off ONE smile at the surface's expiry (the FX-cube hot path):
// resolve each strike (from delta if by_delta) once, read the smile vol, and take gk_greeks in a single
// pass into the pre-sized SoA columns.
inline FxVolCube price_fx_book(const FxSurfaceMarket& m, const FxVolSurface& surf,
                               const std::vector<FxOption>& opts) {
  const double T = surf.expiry();
  const double F = surf.forward();
  FxVolCube out;
  out.n = static_cast<int>(opts.size());
  for (std::vector<double>* col : {&out.strike, &out.vol, &out.forward, &out.price, &out.delta, &out.gamma,
                                   &out.vega, &out.theta, &out.rho_dom, &out.rho_for, &out.notional_price})
    col->reserve(opts.size());
  for (const FxOption& o : opts) {
    const double K = o.by_delta ? surf.strike_for_delta(o.delta, o.cp, o.delta_conv) : o.strike;
    const double vol = surf.vol_at_strike(K);
    const GkGreeks<double> g = gk_greeks<double>(m.spot, K, vol, T, m.r_dom, m.r_for, o.cp);
    out.strike.push_back(K);
    out.vol.push_back(vol);
    out.forward.push_back(F);
    out.price.push_back(g.price);
    out.delta.push_back(g.delta);
    out.gamma.push_back(g.gamma);
    out.vega.push_back(g.vega);
    out.theta.push_back(g.theta);
    out.rho_dom.push_back(g.rho_dom);
    out.rho_for.push_back(g.rho_for);
    out.notional_price.push_back(g.price * o.notional);
  }
  return out;
}

}  // namespace swaps::vol
