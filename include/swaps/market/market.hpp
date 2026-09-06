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
#include "swaps/vol/sabr.hpp"

namespace swaps::market {

namespace curve = swaps::curve;
namespace pricing = swaps::pricing;
namespace build = swaps::build;
namespace vol = swaps::vol;

// One cell of a named vol surface: a (expiry, tenor) grid point carrying a curve-INDEPENDENT vol model —
// either a flat normal (Bachelier) vol, or a vol::SabrParams smile. This is exactly the surface DEFINITION
// the vega verb already consumes as `cells` (the expiry/tenor labels are resolved onto a calibrated curve at
// price time — curve-independent here), lifted into a value that can live inside the Market snapshot. It
// REUSES vol::SabrParams: the SABR model itself is not rebuilt.
struct VolCell {
  std::string expiry;       // e.g. "1Y" — resolved to curve time by the consumer (kept as a label here)
  std::string tenor;        // e.g. "5Y"
  bool has_sabr = false;    // model selector: SABR smile when true, else the flat normal vol
  double normal_vol = 0.0;  // flat Bachelier vol (has_sabr == false)
  vol::SabrParams sabr{};   // SABR triple {alpha, rho, nu} (has_sabr == true) — the existing vol/ value type
};

// A named vol surface stored in the Market: an ordered grid of VolCells. Unlike a realized curve this is a
// COPYABLE value (no move-only regions), so a Market clone() carries an INDEPENDENT copy — the Scenario-fork
// invariant (mutating the fork never touches the parent) holds for vol exactly as it does for curves. NB this
// is the market-DATA surface; the compiled/streaming api::VolSurface (BundleSession-bound, in the higher api
// layer) is a different object that this can be the vol source for.
class VolSurface {
 public:
  VolSurface() = default;
  VolSurface& add_cell(VolCell c) {
    cells_.push_back(std::move(c));
    return *this;
  }
  const std::vector<VolCell>& cells() const { return cells_; }
  std::vector<VolCell>& cells() { return cells_; }  // mutate vols in place (a slider tick); clone stays private
  int n_cells() const { return static_cast<int>(cells_.size()); }

 private:
  std::vector<VolCell> cells_;
};

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
  // Add a named vol surface by NAME (the vol analogue of add_curve): the snapshot's shared vol source, so
  // vega/scenario can name ONE surface instead of re-specifying cells inline. Re-keying overwrites.
  Market& add_vol_surface(const std::string& name, VolSurface s) {
    vol_surfaces_.insert_or_assign(name, std::move(s));
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
    m.vol_surfaces_ = vol_surfaces_;  // VolSurface is a value type: this is an independent deep copy (fork-safe)
    for (const auto& kv : curves_) m.add_curve(kv.first, kv.second.modules, kv.second.forwards);
    return m;
  }

  bool has_quote(const std::string& name) const { return quotes_.count(name) != 0; }
  const Quote& quote(const std::string& name) const {
    auto it = quotes_.find(name);
    if (it == quotes_.end()) throw std::runtime_error("Market: no quote named '" + name + "'");
    return it->second;
  }

  // Named vol-surface lookup — mirrors the curve store (has_/by-name/names; missing name throws). A const
  // overload for reading and a non-const for mutating a stored surface in place (a slider tick on a fork).
  bool has_vol_surface(const std::string& name) const { return vol_surfaces_.count(name) != 0; }
  const VolSurface& vol_surface(const std::string& name) const {
    auto it = vol_surfaces_.find(name);
    if (it == vol_surfaces_.end()) throw std::runtime_error("Market: no vol surface named '" + name + "'");
    return it->second;
  }
  VolSurface& vol_surface(const std::string& name) {
    auto it = vol_surfaces_.find(name);
    if (it == vol_surfaces_.end()) throw std::runtime_error("Market: no vol surface named '" + name + "'");
    return it->second;
  }
  std::vector<std::string> vol_surface_names() const {
    std::vector<std::string> ns;
    for (const auto& kv : vol_surfaces_) ns.push_back(kv.first);
    return ns;
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
  std::map<std::string, VolSurface> vol_surfaces_;  // NAMED vol surfaces — the vol analogue of the curve store
  pricing::FixingTable fixings_;
};

}  // namespace swaps::market
