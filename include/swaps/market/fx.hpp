#pragma once
// A first-class, triangulating FX SPOT store (a market VALUE type).
//
// Historically FX spot has been threaded through the model as a scattered `double fx_spot` living on
// individual legs / instruments (FloatLeg, Instrument in calibration/problem.hpp; the fx_forward / xccy_mtm
// builders in build/instruments.hpp). Every caller had to know the right number for the right pair, and there
// was no single object that understood that EURUSD and USDEUR are the same quote, or that a EURJPY cross is
// just EURUSD * USDJPY. This header adds ONE shared, order-agnostic store that owns those relationships, so a
// caller asks `rate(from, to)` and gets the conversion factor — direct, inverted, or triangulated — instead of
// carrying a hand-picked spot around.
//
// RATE-DIRECTION CONVENTION (documented once, used everywhere):
//   A quote is (base, quote, rate) meaning  1 unit of `base` = `rate` units of `quote`.
//   e.g. FxRate{"EUR","USD",1.09}  ==  EURUSD = 1.09  ==  1 EUR = 1.09 USD.
//   Consequently `rate(from, to)` returns the number of `to` units per 1 unit of `from`
//   (so rate("EUR","USD") == 1.09, and rate("USD","EUR") == 1/1.09).
//
// Header-only, value semantics, QuantLib-free.

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

namespace swaps::market {

// A single quoted spot. (base, quote, rate) reads "1 base = rate * quote".
struct FxRate {
  std::string base;
  std::string quote;
  double rate = 0.0;
};

// A shared FX-spot store. Add each quoted pair once; the inverse is implied, and any missing cross is derived
// by triangulating through a pivot currency (USD by default). A clean value type: copyable, comparable-by-state
// via its underlying map, cheap to pass around.
class FxMatrix {
 public:
  FxMatrix() = default;
  explicit FxMatrix(std::string pivot) : pivot_(std::move(pivot)) {}

  // Store the pair `base`/`quote` at `rate` (1 base = rate * quote). The inverse direction is implied and never
  // stored separately. Re-adding a pair overwrites it.
  void add(const std::string& base, const std::string& quote, double rate) {
    pairs_[key(base, quote)] = rate;
  }
  void add(const FxRate& q) { add(q.base, q.quote, q.rate); }

  // The FX rate to convert 1 unit of `from` into units of `to`. Resolution order:
  //   1. identity            from == to           -> 1
  //   2. a directly stored pair (from, to)        -> stored rate
  //   3. the inverse of a stored pair (to, from)  -> 1 / stored rate
  //   4. triangulation through the pivot          -> rate(from, pivot) * rate(pivot, to)
  // Throws std::runtime_error when no path exists (use has() to probe first, or convert()).
  double rate(const std::string& from, const std::string& to) const {
    double r = 0.0;
    if (!try_rate(from, to, r)) {
      throw std::runtime_error("FxMatrix: no FX path from '" + from + "' to '" + to + "' (pivot '" + pivot_ +
                               "')");
    }
    return r;
  }

  // Convert `amount` of `from` into `to`. Same resolution (and same throw) as rate().
  double convert(double amount, const std::string& from, const std::string& to) const {
    return amount * rate(from, to);
  }

  // True when rate(from, to) can be resolved (identity, direct, inverse, or triangulated) without throwing.
  bool has(const std::string& from, const std::string& to) const {
    double r = 0.0;
    return try_rate(from, to, r);
  }

  const std::string& pivot() const { return pivot_; }
  bool empty() const { return pairs_.empty(); }

 private:
  static std::string key(const std::string& base, const std::string& quote) { return base + quote; }

  // A direct or inverse lookup only (no triangulation). Returns false if neither direction is stored.
  bool try_direct(const std::string& from, const std::string& to, double& out) const {
    auto it = pairs_.find(key(from, to));
    if (it != pairs_.end()) {
      out = it->second;
      return true;
    }
    auto inv = pairs_.find(key(to, from));
    if (inv != pairs_.end() && inv->second != 0.0) {
      out = 1.0 / inv->second;
      return true;
    }
    return false;
  }

  // Full resolution (identity / direct / inverse / triangulate-through-pivot). Returns false, without throwing,
  // when no path exists.
  bool try_rate(const std::string& from, const std::string& to, double& out) const {
    if (from == to) {
      out = 1.0;
      return true;
    }
    if (try_direct(from, to, out)) return true;

    // Triangulate: from -> pivot -> to. Skip the leg that IS the pivot to avoid a redundant identity hop.
    if (from == pivot_ || to == pivot_) return false;  // a plain pivot leg would already be direct
    double leg1 = 0.0, leg2 = 0.0;
    if (try_direct(from, pivot_, leg1) && try_direct(pivot_, to, leg2)) {
      out = leg1 * leg2;
      return true;
    }
    return false;
  }

  std::string pivot_ = "USD";
  std::map<std::string, double> pairs_;  // keyed by normalized "BASEQUOTE"; inverse implied.
};

}  // namespace swaps::market
