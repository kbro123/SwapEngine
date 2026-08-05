#pragma once
// No-arbitrage vol-cube interpolation (Phase A §1.2). A swaption vol cube is arbitrage-free iff, at every
// point, BOTH conditions hold:
//   * BUTTERFLY (strike): the implied density d^2C/dK^2 >= 0 — checked by the SABR smile + min_price_convexity
//     detector in vol/sabr_calibration.hpp (§1.1);
//   * CALENDAR (expiry): the total variance w(T) = sigma^2(T) * T is non-decreasing in expiry — so a longer
//     option can never be worth less time value than a shorter one at the same strike.
// This header supplies the calendar side + the arb-preserving interpolator. Interpolating the ATM vol LINEARLY
// IN TOTAL VARIANCE (not in vol) keeps the term structure calendar-arb-free whenever the nodes are, which is
// the standard construction; the strike dimension stays on the (butterfly-checked) SABR smile and the tenor
// dimension interpolates the per-node smiles. A NoArbVolCurve wraps (expiries, atm_vols) as one such curve.
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace swaps::vol {

inline double total_variance(double vol, double expiry) { return vol * vol * expiry; }

// Calendar no-arbitrage: the total variance is non-decreasing across the (sorted, ascending) expiries.
inline bool calendar_arbitrage_free(const std::vector<double>& expiries, const std::vector<double>& atm_vols,
                                    double tol = 1e-10) {
  if (expiries.size() != atm_vols.size()) throw std::invalid_argument("calendar: length mismatch");
  for (std::size_t i = 1; i < expiries.size(); ++i)
    if (total_variance(atm_vols[i], expiries[i]) < total_variance(atm_vols[i - 1], expiries[i - 1]) - tol)
      return false;
  return true;
}

// Interpolate the ATM vol at `expiry` from nodes (expiries, atm_vols), LINEARLY IN TOTAL VARIANCE — the
// calendar-arb-preserving interpolation. Below the first / above the last node, hold the endpoint total
// variance flat (so w never decreases). Requires ascending, positive expiries.
inline double interp_total_variance(const std::vector<double>& expiries, const std::vector<double>& atm_vols,
                                    double expiry) {
  const std::size_t n = expiries.size();
  if (n == 0) throw std::invalid_argument("interp_total_variance: empty nodes");
  if (n != atm_vols.size()) throw std::invalid_argument("interp_total_variance: length mismatch");
  if (expiry <= 0.0) return atm_vols.front();
  if (expiry <= expiries.front()) return std::sqrt(total_variance(atm_vols.front(), expiries.front()) / expiry);
  if (expiry >= expiries.back()) return std::sqrt(total_variance(atm_vols.back(), expiries.back()) / expiry);
  const std::size_t j = static_cast<std::size_t>(
      std::upper_bound(expiries.begin(), expiries.end(), expiry) - expiries.begin());
  const double w0 = total_variance(atm_vols[j - 1], expiries[j - 1]);
  const double w1 = total_variance(atm_vols[j], expiries[j]);
  const double f = (expiry - expiries[j - 1]) / (expiries[j] - expiries[j - 1]);
  const double w = w0 + f * (w1 - w0);
  return std::sqrt(w / expiry);
}

// An ATM vol term structure for one tenor, interpolated arb-free in total variance.
class NoArbVolCurve {
 public:
  NoArbVolCurve(std::vector<double> expiries, std::vector<double> atm_vols)
      : expiries_(std::move(expiries)), vols_(std::move(atm_vols)) {
    if (expiries_.size() != vols_.size() || expiries_.empty())
      throw std::invalid_argument("NoArbVolCurve: non-empty, equal-length nodes required");
    for (std::size_t i = 1; i < expiries_.size(); ++i)
      if (!(expiries_[i] > expiries_[i - 1]))
        throw std::invalid_argument("NoArbVolCurve: expiries must be strictly ascending");
  }
  double vol(double expiry) const { return interp_total_variance(expiries_, vols_, expiry); }
  bool arbitrage_free(double tol = 1e-10) const { return calendar_arbitrage_free(expiries_, vols_, tol); }

 private:
  std::vector<double> expiries_, vols_;
};

}  // namespace swaps::vol
