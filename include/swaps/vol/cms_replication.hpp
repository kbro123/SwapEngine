#pragma once
// Full Hagan static-replication CMS convexity (Phase A §1.4). Where vol/cms.hpp gives the linear-TSR
// first-order adjustment, this integrates the ACTUAL swaption smile against the standard-model (Hagan)
// replication weight — so the CMS convexity reflects the wings, not just the ATM vol. The G function and its
// derivatives, the integrand F(x) = (x-K)(G(x)/G(F0) - 1), and the replication (Hagan "Convexity Conundrums"
// 2003, eq. 2.17a/2.18a) are transcribed from QuantLib's NumericHaganPricer / GFunctionStandard and
// oracle-checked against it (tests/vol_cms_replication_oracle.cpp). QuantLib-free; model decimals.
#include <algorithm>
#include <cmath>
#include <functional>

#include "swaps/vol/bachelier.hpp"

namespace swaps::vol {

// Hagan standard-model weight G and its derivatives. q = fixed periods/year, delta = pay-lag as a fraction of
// the first period, swap_length = tenor in years (n = swap_length*q periods). == QuantLib GFunctionStandard.
struct GFunctionStandard {
  double q = 1.0;
  double delta = 0.0;
  double swap_length = 0.0;
  double n() const { return swap_length * q; }

  double operator()(double x) const {
    const double N = n(), a = 1.0 + x / q;
    return x / std::pow(a, delta) / (1.0 - 1.0 / std::pow(a, N));
  }
  // G' and G''. Every power in the closed forms shares the base a = 1 + x/q, so we evaluate a^N and a^delta
  // ONCE and derive the rest (a^(N-1)=aN/a, a^(N-delta-1)=aN/(aD·a), a^(2N-delta-2)=aN²/(aD·a²), …) by cheap
  // mul/div — two std::pow per call instead of the ~4/~12 the term-by-term forms issued. Algebraically
  // identical (oracle: G' 1e-9, G'' 1e-7 vs QuantLib GFunctionStandard), so only the pow count changes.
  double d1(double x) const {
    double g1, g2;
    derivs(x, g1, g2);
    return g1;
  }
  double d2(double x) const {
    double g1, g2;
    derivs(x, g1, g2);
    return g2;
  }
  // Both derivatives in one pass (the CMS Simpson integrand needs both at each node): amortises a^N and
  // a^delta across G' and G'' — two std::pow per node instead of four.
  void derivs(double x, double& g1, double& g2) const {
    const double N = n(), a = 1.0 + x / q;
    const double aN = std::pow(a, N), aD = std::pow(a, delta);
    const double a2 = a * a, AA = a - delta / q * x, dm = aN - 1.0, dm2 = dm * dm;
    const double B = (aN / (aD * a)) / dm;                       // a^(N-delta-1) / (a^N - 1)
    // G'
    const double sec = (N * x * (aN / a)) / (q * aD * dm2);      // N x a^(N-1) / (q a^delta (a^N-1)^2)
    g1 = AA * B - sec;
    // G''
    const double A1 = (1.0 - delta) / q;
    const double Num = ((1.0 + delta - N) * aN - (1.0 + delta) * aN * aN) / (aD * a2);
    const double B1 = Num / (q * dm2);
    const double C = x / aD;
    const double C1 = (1.0 - delta / q * x / a) / aD;            // (a^d - (d/q)x a^(d-1)) / a^(2d)
    const double D = (aN / a) / dm2;                             // a^(N-1) / (a^N-1)^2
    const double D1 = (aN / a2) * ((N - 1.0) * dm - 2.0 * N * aN) / (q * dm2 * dm);
    g2 = A1 * B + AA * B1 - N / q * (C1 * D + C * D1);
  }
};

// The convexity-adjusted CMS forward swap rate by static replication over the swaption smile. `vol_at(strike)`
// gives the normal vol at that strike (a SABR smile, or a constant). `forward` = the forward swap rate, `expiry`
// the fixing time. F''(x) = [2 G'(x) + (x-F0) G''(x)] / G(F0); adjustment = ∫_{F0}^{hi} F''·payer + ∫_{lo}^{F0}
// F''·receiver (Bachelier, annuity 1 — it cancels). Strikes span forward ± n_std ATM std devs, Simpson `steps`.
inline double cms_replicated_forward(double forward, double expiry, const GFunctionStandard& G,
                                     const std::function<double(double)>& vol_at,
                                     double n_std = 6.0, int steps = 1000, double lo_frac = 0.20) {
  const double G0 = G(forward);
  const double atm_sd = std::max(vol_at(forward), 1e-8) * std::sqrt(expiry);
  // The standard-model G has a NON-INTEGRABLE pole at the rate S=0 (G'' ~ -C/S^2), so the receiver leg is
  // floored at a small positive rate (`lo_frac`*forward) rather than run to/through zero — where the
  // receiver is already deep out-of-the-money and its contribution negligible. (QuantLib integrates through
  // the pole and its result there is quadrature-node-dependent; we stay on the well-posed side.)
  const double hi = forward + n_std * atm_sd;
  const double lo = std::max(forward - n_std * atm_sd, lo_frac * forward);
  const auto Fpp = [&](double x) {
    double g1, g2;
    G.derivs(x, g1, g2);  // one shared a^N/a^delta pass instead of separate d1(x)/d2(x)
    return (2.0 * g1 + (x - forward) * g2) / G0;
  };
  const auto simpson = [&](double a, double b, Payoff cp) {
    const int m = steps % 2 ? steps + 1 : steps;  // even for Simpson
    const double h = (b - a) / m;
    double s = 0.0;
    for (int i = 0; i <= m; ++i) {
      const double x = a + i * h;
      const double w = (i == 0 || i == m) ? 1.0 : (i % 2 ? 4.0 : 2.0);
      s += w * Fpp(x) * bachelier_price<double>(forward, x, vol_at(x), expiry, 1.0, cp);
    }
    return s * h / 3.0;
  };
  return forward + simpson(forward, hi, Payoff::Payer) + simpson(lo, forward, Payoff::Receiver);
}

}  // namespace swaps::vol
