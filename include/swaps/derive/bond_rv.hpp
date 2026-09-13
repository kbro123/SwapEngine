#pragma once
// derive/bond_rv.hpp — bond relative value over a UNIVERSE (E7 stage 3.4): the `bond_universe` and `govvie_fit`
// verbs' whole computations, moved out of api/rv.cpp so they are unit-testable and mutation-reachable.
//
//   * bond_universe — batched street analytics (portfolio::BondUniverse: one Horner / Newton sweep), cleans <->
//     yields, modified duration, convexity, accrued. Settlement defaults to the bond convention's rule.
//   * govvie_fit    — the minimum-pricing-error govvie fit (spline, Nelson-Siegel, Svensson) over a Market built
//     from the universe's clean prices; per-bond fair-value residuals, and for the spline the per-bond z-spreads
//     off the fitted curve (portfolio::CompiledBondBook over the SAME bonds the fit priced).
//
// Needs the derive --> portfolio arrow (ARCHITECTURE.md): portfolio reaches only pricing / curve / ad, so no cycle.

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

#include "swaps/build/bond.hpp"
#include "swaps/calibration/bond_fit.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/parametric.hpp"
#include "swaps/derive/asset_swap.hpp"
#include "swaps/market/market.hpp"
#include "swaps/portfolio/bond_universe.hpp"

namespace swaps::derive {

namespace portfolio = swaps::portfolio;

// ---- bond_universe ------------------------------------------------------------------------------------------
struct BondUniverseRequest {
  build::Date value_date;
  std::string convention;                           // the universe's bonds[] convention (settlement) -- REQUIRED
  std::optional<build::Date> settle;                // absent => the convention's settlement rule from value_date
  std::vector<build::BondId> bonds;                 // each carries its own yield convention
  std::optional<std::vector<double>> clean, yield;  // EXACTLY ONE, one entry per bond
};
struct BondUniverseResult {
  Eigen::VectorXd clean, yield, modified_duration, convexity, accrued;
};

inline BondUniverseResult bond_universe(const BondUniverseRequest& r) {
  const AssetSwapConvention conv = asset_swap_convention(r.convention);
  const build::Date settle = r.settle ? *r.settle : settlement_date(conv, r.value_date);
  if (r.bonds.empty()) throw std::invalid_argument("bond_universe: needs at least one bond");
  if (r.clean.has_value() == r.yield.has_value())
    throw std::invalid_argument("bond_universe: provide exactly one of 'clean' or 'yield' (per bond)");
  const std::vector<double>& quotes = r.clean ? *r.clean : *r.yield;
  const int n = static_cast<int>(r.bonds.size());
  if (static_cast<int>(quotes.size()) != n)
    throw std::invalid_argument("bond_universe: quote array length must match the bond count");

  BondUniverseResult out;
  out.accrued.resize(n);
  std::vector<pricing::YieldBond> yb;
  yb.reserve(r.bonds.size());
  for (int i = 0; i < n; ++i) {
    build::BuiltBond built = build::build_bond(r.bonds[static_cast<std::size_t>(i)], r.value_date, settle);
    out.accrued[i] = built.accrued;
    yb.push_back(std::move(built.yield));
  }
  portfolio::BondUniverse uni;
  uni.set(yb);
  const Eigen::Map<const Eigen::VectorXd> q(quotes.data(), n);
  // The universe's sweeps return references into its scratch: copy each result out before the next call.
  if (r.clean) {
    out.yield = uni.yields_from_clean(q);
    out.clean = q;
  } else {
    out.yield = q;
    out.clean = uni.clean_prices(out.yield);
  }
  out.modified_duration = uni.modified_durations(out.yield);
  out.convexity = uni.convexities(out.yield);
  return out;
}

// ---- govvie_fit ---------------------------------------------------------------------------------------------
enum class GovvieModel { Spline, NelsonSiegel, Svensson };

struct GovvieFitRequest {
  build::Date value_date;
  std::string convention;          // settlement convention -- REQUIRED
  SettlementOverride settlement;   // absent fields => the bond row's calendar / settle_lag
  std::vector<build::BondId> bonds;
  std::vector<double> clean;       // one per bond
  std::vector<double> weight;      // empty => all 1; else one per bond
  GovvieModel model = GovvieModel::Spline;
  std::vector<double> meeting, back;  // Spline topology: `back` REQUIRED
  std::optional<double> tau1, tau2;   // parametric decays; absent => cal::ParametricBondFit's own
  double x0 = 0.03;                   // seed LEVEL: every spline knot; beta0 of a parametric model (other betas 0)
};
struct GovvieFitResult {
  GovvieModel model = GovvieModel::Spline;
  cal::CalibrationResult fit;
  Eigen::VectorXd residuals;               // model - market clean, per bond (the fair-value ladder)
  std::optional<Eigen::VectorXd> z_spread; // Spline only
};

namespace bond_rv_detail {
template <template <class> class Model>
GovvieFitResult fit_parametric(const AssetSwapConvention& conv, const GovvieFitRequest& r, const market::Market& m) {
  const cal::ParametricBondFit<Model> defaults;
  auto fit = make_parametric_fit<Model>(conv, r.bonds, m, r.tau1.value_or(defaults.tau1),
                                        r.tau2.value_or(defaults.tau2), r.weight);
  Eigen::VectorXd seed = Eigen::VectorXd::Zero(fit.n_knots());
  seed[0] = r.x0;
  GovvieFitResult out;
  out.model = r.model;
  out.fit = cal::calibrate(fit, seed);
  out.residuals = fit.template residuals<double>(out.fit.x);
  return out;
}
}  // namespace bond_rv_detail

inline GovvieFitResult govvie_fit(const GovvieFitRequest& r) {
  const AssetSwapConvention conv = asset_swap_convention(r.convention, r.settlement);
  if (r.bonds.empty()) throw std::invalid_argument("govvie_fit: needs at least one bond");
  if (r.clean.size() != r.bonds.size())
    throw std::invalid_argument("govvie_fit: 'clean' length must match the bond count");
  std::unordered_set<std::string> ids;
  for (const build::BondId& b : r.bonds)
    if (!ids.insert(b.id).second)  // a Market holds one quote per id: a duplicate was silently priced at the last one
      throw std::invalid_argument("govvie_fit: duplicate bond id '" + b.id + "'");

  market::Market m;  // the snapshot the derive layer consumes: as-of + one CLEAN-price quote per bond
  m.as_of(r.value_date);
  for (std::size_t i = 0; i < r.bonds.size(); ++i) m.add_quote(r.bonds[i].id, market::Quote::mid(r.clean[i]));

  switch (r.model) {
    case GovvieModel::Spline: {
      if (r.back.empty()) throw std::invalid_argument("govvie_fit: spline model needs 'back' knot times");
      cal::GovvieBondFit fit = make_govvie_fit(conv, r.bonds, m, r.meeting, r.back, r.weight);
      GovvieFitResult out;
      out.model = r.model;
      out.fit = cal::calibrate(fit, Eigen::VectorXd::Constant(fit.n_knots(), r.x0));
      out.residuals = fit.residuals<double>(out.fit.x);
      // z-spreads off the FITTED curve against each bond's dirty market price, on the very bonds the fit priced.
      Eigen::VectorXd target_dirty(fit.market_clean.size());
      for (int b = 0; b < target_dirty.size(); ++b)
        target_dirty[b] = fit.market_clean[b] + fit.bonds[static_cast<std::size_t>(b)].accrued;
      const portfolio::CompiledBondBook book(r.meeting, r.back, fit.bonds);
      out.z_spread = book.z_spreads(out.fit.x, target_dirty);
      return out;
    }
    case GovvieModel::NelsonSiegel:
      return bond_rv_detail::fit_parametric<curve::NelsonSiegel>(conv, r, m);
    case GovvieModel::Svensson:
      return bond_rv_detail::fit_parametric<curve::Svensson>(conv, r, m);
  }
  throw std::invalid_argument("govvie_fit: unknown model");
}

}  // namespace swaps::derive
