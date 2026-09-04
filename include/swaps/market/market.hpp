#pragma once
// market::Market — the MARKET ENVIRONMENT: a named, multi-currency snapshot of everything a pricer needs as
// of one valuation date. The commercial review found this the single biggest object-model gap — the engine
// held market state only implicitly inside a calibration BundleSession, addressed curves by integer role,
// and had no FX / currency / quote aggregation. Market composes the Wave-1 value objects (Currency, FxMatrix,
// Quote) with named realized curves and the ambient FixingTable into the "market as of D" you build once and
// price many books against — Strata's ImmutableRatesProvider / ORE's TodaysMarket, in our object model.
//
// It sits ABOVE the curve/pricing layers and BELOW the calibration api: it holds realized curves as named
// curve::ModularCurve values (built from a spec + solved forwards), so it depends only on curve/ + pricing/
// + market/, never on the calibration/api layer. Build with the chained setters, then query. Move-only
// (curves own unique_ptr regions) — a snapshot is built and held, not copied.

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "swaps/build/date.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/market/currency.hpp"
#include "swaps/market/fx.hpp"
#include "swaps/market/quote.hpp"
#include "swaps/pricing/fixings.hpp"

namespace swaps::market {

namespace curve = swaps::curve;
namespace pricing = swaps::pricing;
namespace build = swaps::build;

class Market {
 public:
  Market() = default;

  // ---- builders (chainable; a snapshot is assembled then frozen by convention) ----------------------
  Market& as_of(const build::Date& d) {
    as_of_ = d;
    return *this;
  }
  Market& add_currency(const Currency& c) {
    currencies_.push_back(c);
    return *this;
  }
  Market& set_fx(FxMatrix fx) {
    fx_ = std::move(fx);
    return *this;
  }
  Market& add_quote(const std::string& name, const Quote& q) {
    quotes_.insert_or_assign(name, q);
    return *this;
  }
  // Add a realized curve by NAME (the gap: curves were int-role only). `modules` is the interpolation
  // topology (e.g. curve::flat_hermite(...)); `x` the solved forwards region-by-region.
  Market& add_curve(const std::string& name, const std::vector<curve::CurveModule>& modules,
                    const Eigen::VectorXd& x) {
    auto crv = curve::make_modular_curve<double>(modules);
    crv.set_forwards(x);
    curves_.insert_or_assign(name, CurveEntry{modules, x, std::move(crv)});
    return *this;
  }
  Market& set_fixings(pricing::FixingTable f) {
    fixings_ = std::move(f);
    return *this;
  }

  // ---- queries --------------------------------------------------------------------------------------
  const build::Date& today() const { return as_of_; }
  const FxMatrix& fx() const { return fx_; }
  const pricing::FixingTable& fixings() const { return fixings_; }

  bool has_curve(const std::string& name) const { return curves_.count(name) != 0; }
  const curve::ModularCurve<double>& curve(const std::string& name) const { return entry(name).realized; }
  double discount(const std::string& curve_name, double t) const { return curve(curve_name).discount(t); }
  std::vector<std::string> curve_names() const {
    std::vector<std::string> ns;
    for (const auto& kv : curves_) ns.push_back(kv.first);
    return ns;
  }

  // The build INPUTS of a named curve — its interpolation topology and solved forwards — so the snapshot can
  // be forked/shocked (a realized ModularCurve is move-only and opaque; these are what rebuild it).
  const std::vector<curve::CurveModule>& curve_modules(const std::string& name) const {
    return entry(name).modules;
  }
  const Eigen::VectorXd& curve_forwards(const std::string& name) const { return entry(name).forwards; }

  // Fork the snapshot: an independent copy rebuilt from the retained inputs (curves are move-only, so they
  // are rebuilt, not copied). The parent is untouched — this is the basis for a Scenario shock over a Market.
  Market clone() const {
    Market m;
    m.as_of_ = as_of_;
    m.currencies_ = currencies_;
    m.fx_ = fx_;
    m.quotes_ = quotes_;
    m.fixings_ = fixings_;
    for (const auto& kv : curves_) m.add_curve(kv.first, kv.second.modules, kv.second.forwards);
    return m;
  }

  bool has_quote(const std::string& name) const { return quotes_.count(name) != 0; }
  const Quote& quote(const std::string& name) const {
    auto it = quotes_.find(name);
    if (it == quotes_.end()) throw std::runtime_error("Market: no quote named '" + name + "'");
    return it->second;
  }

  const std::vector<Currency>& currencies() const { return currencies_; }
  Currency currency(const std::string& code) const {
    for (const auto& c : currencies_)
      if (c.code == code) return c;
    throw std::runtime_error("Market: currency '" + code + "' is not in this snapshot");
  }
  // The FX rate to convert 1 unit of `from` into `to` (via the snapshot's FxMatrix triangulation).
  double fx_rate(const std::string& from, const std::string& to) const { return fx_.rate(from, to); }

 private:
  // A named curve retains its build inputs (modules + forwards) alongside the realized curve, so the
  // snapshot can be forked/shocked — a ModularCurve is move-only and does not expose its inputs.
  struct CurveEntry {
    std::vector<curve::CurveModule> modules;
    Eigen::VectorXd forwards;
    curve::ModularCurve<double> realized;
  };
  const CurveEntry& entry(const std::string& name) const {
    auto it = curves_.find(name);
    if (it == curves_.end()) throw std::runtime_error("Market: no curve named '" + name + "'");
    return it->second;
  }

  build::Date as_of_{};
  std::vector<Currency> currencies_;
  FxMatrix fx_;
  std::map<std::string, Quote> quotes_;
  std::map<std::string, CurveEntry> curves_;  // NAMED realized curves + their inputs — the missing lookup
  pricing::FixingTable fixings_;
};

}  // namespace swaps::market
