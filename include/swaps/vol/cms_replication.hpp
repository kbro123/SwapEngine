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
  double d1(double x) const {
    const double N = n(), a = 1.0 + x / q, AA = a - delta / q * x;
    const double B = std::pow(a, N - delta - 1.0) / (std::pow(a, N) - 1.0);
    const double sec = (N * x * std::pow(a, N - 1.0)) /
                       (q * std::pow(a, delta) * (std::pow(a, N) - 1.0) * (std::pow(a, N) - 1.0));
    return AA * B - sec;
  }
  double d2(double x) const {
    const double N = n(), a = 1.0 + x / q, AA = a - delta / q * x, A1 = (1.0 - delta) / q;
    const double B = std::pow(a, N - delta - 1.0) / (std::pow(a, N) - 1.0);
    const double Num = (1.0 + delta - N) * std::pow(a, N - delta - 2.0) -
                       (1.0 + delta) * std::pow(a, 2.0 * N - delta - 2.0);
    const double Den = (std::pow(a, N) - 1.0) * (std::pow(a, N) - 1.0);
    const double B1 = Num / (q * Den);
    const double C = x / std::pow(a, delta);
    const double C1 =
        (std::pow(a, delta) - delta / q * x * std::pow(a, delta - 1.0)) / std::pow(a, 2.0 * delta);
    const double D = std::pow(a, N - 1.0) / ((std::pow(a, N) - 1.0) * (std::pow(a, N) - 1.0));
    const double D1 = ((N - 1.0) * std::pow(a, N - 2.0) * (std::pow(a, N) - 1.0) -
                       2.0 * N * std::pow(a, 2.0 * (N - 1.0))) /
                      (q * (std::pow(a, N) - 1.0) * (std::pow(a, N) - 1.0) * (std::pow(a, N) - 1.0));
    return A1 * B + AA * B1 - N / q * (C1 * D + C * D1);
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
  const auto Fpp = [&](double x) { return (2.0 * G.d1(x) + (x - forward) * G.d2(x)) / G0; };
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
