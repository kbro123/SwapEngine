#pragma once
// market::Scenario — a declarative "fork over the market" primitive, expressed at the BUILD-INPUT level.
//
// Market (market/market.hpp) is MOVE-ONLY and holds each curve as a realized curve::ModularCurve value built
// from (modules, x) via add_curve(); a built curve does NOT expose a path back to its modules + forward vector.
// So a scenario cannot copy or mutate a built Market. Instead a Scenario is a set of shocks that transform the
// INPUTS a Market is assembled from — the per-curve forward vectors (x) and the FxMatrix spot store — and the
// caller rebuilds a shocked Market from the shocked inputs. Same modules, same code path, forked inputs.
//
// This keeps the primitive minimal and dependency-light: it operates on x (Eigen::VectorXd) and FxMatrix, and
// need not include market.hpp at all. It composes with Model/Market rebuilds rather than reaching inside them.
//
// SHOCK MODEL
//   * curve rate shocks   — a PARALLEL shift, in basis points, added to every forward of a named curve.
//                           1bp = 1e-4 in forward-rate space. An unshocked curve is returned unchanged.
//                           A global "shift ALL curves" default (Scenario::parallel(bp)) applies to any curve
//                           that is not explicitly keyed; an explicit shift_curve() for a name overrides it.
//   * fx spot shocks      — a RELATIVE bump per pair: rate' = rate * (1 + rel). +0.01 == +1%. Applied on a
//                           COPY of the FxMatrix (it is a copyable value type), leaving the base untouched.
//
// Header-only, value semantics, QuantLib-free.

#include <map>
#include <string>
#include <utility>

#include <Eigen/Core>

#include "swaps/market/fx.hpp"

namespace swaps::market {

class Scenario {
 public:
  Scenario() = default;

  // ---- builders (chainable) -------------------------------------------------------------------------
  // Parallel-shift the named curve by `bp` basis points (added to every forward). Re-keying overwrites.
  Scenario& shift_curve(const std::string& name, double bp) {
    curve_bp_[name] = bp;
    return *this;
  }
  // Relatively bump the FX pair base/quote by `rel` (e.g. +0.01 for +1%): rate' = rate * (1 + rel).
  Scenario& bump_fx(const std::string& base, const std::string& quote, double rel) {
    const std::string k = base + quote;
    fx_rel_[k] = rel;
    fx_pairs_[k] = {base, quote};
    return *this;
  }

  // A named convention for "shift ALL curves by bp": a global default applied to any curve that is not
  // explicitly keyed via shift_curve(). An explicit shift_curve(name, ...) always overrides the global.
  static Scenario parallel(double bp) {
    Scenario s;
    s.has_global_ = true;
    s.global_bp_ = bp;
    return s;
  }

  // ---- application to build inputs ------------------------------------------------------------------
  // The parallel forward shift (in rate space, bp/1e4) that applies to `curve_name`:
  //   an explicit shift_curve() entry, else the global parallel() default, else 0.
  double curve_shift(const std::string& curve_name) const {
    auto it = curve_bp_.find(curve_name);
    if (it != curve_bp_.end()) return it->second / 1e4;
    if (has_global_) return global_bp_ / 1e4;
    return 0.0;
  }

  // Return `x` with this scenario's shift for `curve_name` added to every forward. An unshocked curve
  // (no explicit key and no global default) returns `x` unchanged.
  Eigen::VectorXd shocked_forwards(const std::string& curve_name, const Eigen::VectorXd& x) const {
    const double d = curve_shift(curve_name);
    if (d == 0.0) return x;
    return x.array() + d;
  }

  // Return a COPY of `base` with each bumped pair scaled by (1 + rel). Unbumped pairs (and any triangulated
  // cross that is not itself stored) are unaffected. The base matrix is left untouched.
  FxMatrix shocked_fx(const FxMatrix& base) const {
    FxMatrix out = base;  // FxMatrix is a copyable value type (holds a std::map<string,double>)
    for (const auto& kv : fx_rel_) {
      const auto& pr = fx_pairs_.at(kv.first);
      const double old = out.rate(pr.first, pr.second);
      out.add(pr.first, pr.second, old * (1.0 + kv.second));
    }
    return out;
  }

  // ---- introspection --------------------------------------------------------------------------------
  bool has_global_shift() const { return has_global_; }
  double global_bp() const { return global_bp_; }
  const std::map<std::string, double>& curve_shifts_bp() const { return curve_bp_; }

 private:
  std::map<std::string, double> curve_bp_;  // curve name -> parallel shift in basis points
  std::map<std::string, double> fx_rel_;    // "BASEQUOTE" -> relative bump (fraction; +0.01 == +1%)
  std::map<std::string, std::pair<std::string, std::string>> fx_pairs_;  // "BASEQUOTE" -> (base, quote)
  bool has_global_ = false;                 // whether a global "shift ALL curves" default is set
  double global_bp_ = 0.0;                  // the global parallel shift in basis points (when has_global_)
};

}  // namespace swaps::market
