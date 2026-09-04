#pragma once
// derive/asset_swap.hpp — the market→calibration BRIDGE for asset swaps / swap spreads. This is the layer
// that turns a per-currency asset-swap CONVENTION + a market snapshot into the calibration rows (and, for
// RV, the minimum-pricing-error govvie fit). It depends on market/ (the snapshot), build/ (bond +
// swap-spread instrument builders) and calibration/ (the Problem types) — the composition layer above them.
//
// TWO workflows over ONE substrate (this is the design tie-in the RV goal needs):
//   * HEADLINE / asset-swap SPREAD (curve calibration): the benchmark bond's yield is an OBSERVED market
//     number, precomputed here (benchmark_yield: build the bond, invert its market clean price to a street
//     YTM) and fed as the pin target of build::asset_swap_spread. The govvie curve is named by the
//     convention and pinned to that yield. -> derive_asset_swap().
//   * RV / MINIMUM PRICING ERROR (bond fair value): the SAME govvie curve, but fitted to a whole UNIVERSE of
//     bonds' market prices by least squares (calibration::GovvieBondFit), each bond's leftover residual /
//     z-spread being its richness-cheapness. -> make_govvie_fit().
// Because both reference the same named govvie curve on the same curve-space bond kernel, an asset swap
// computed against the RV-fitted curve IS the bond's asset-swap richness — bonds and swaps share one curve.
//
// The bond street-yield / price↔yield inversion is a PRECOMPUTE off the Market, never on the calibration hot
// path (the compiled solve for the swap curve never sees a bond).

#include <string>
#include <vector>

#include <Eigen/Core>

#include "swaps/build/bond.hpp"
#include "swaps/build/calendar.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/swap_spread.hpp"
#include "swaps/calibration/bond_fit.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/market/market.hpp"
#include "swaps/pricing/bond.hpp"

namespace swaps::derive {

namespace build = swaps::build;
namespace cal = swaps::calibration;
namespace curve = swaps::curve;
namespace market = swaps::market;
namespace pricing = swaps::pricing;

// How a currency turns a bond yield + a matched swap into a swap-spread quote. (Only HeadlineYield is wired
// end-to-end today; the others name the remaining desk conventions so the type is stable as they land.)
enum class SwapSpreadType {
  HeadlineYield,    // OTR benchmark bond STREET yield vs the spot-start par swap of the same TENOR (USD).
  MatchedMaturity,  // the benchmark yield vs a swap of the bond's OWN maturity.
  ParAssetSwap,     // the par-par asset-swap spread (bond repriced on the swap curve; endogenous, later).
  Invoice,          // bond-future CTD FORWARD yield to delivery vs the matched swap.
};

// One bond's static identity in a universe (dates + coupon + which bond convention prices its yield). Its
// live PRICE is a Quote in the Market, keyed by `id` — ref data and market data stay separate.
struct BondRef {
  std::string id;          // the key under which the Market carries this bond's CLEAN-price quote
  std::string sector;      // benchmark bucket, e.g. "5Y" (which tenor's on-the-run this is)
  std::string yield_conv;  // conventions.json bond convention id, e.g. "US-TREASURY" (freq + stub rule)
  build::Date issue;       // dated date (first accrual start)
  build::Date maturity;
  double coupon = 0.0;     // annual coupon rate (0.04 = 4%)
};

// A per-currency asset-swap convention. Data — the benchmark selection, the govvie curve it references (by
// NAME, so it can be a 1-knot headline factor today and a fitted RV curve later), the settlement rule, and
// the swap index the spread is quoted against.
struct AssetSwapConvention {
  std::string currency;                                  // "USD"
  SwapSpreadType type = SwapSpreadType::HeadlineYield;
  std::string govvie_curve;                              // the govvie curve this spread references, by name
  std::string swap_index;                                // the matched swap's index, e.g. "USD-SOFR"
  std::string settle_calendar = "USD";                   // conventions-DB calendar key for settlement
  int settle_lag = 1;                                    // business days to settlement (T+1 for USTs)
};

// The benchmark's STREET yield, precomputed from the Market: build the bond on its convention, read its
// market CLEAN price (Market.quote(id).mid()), and invert to yield via the engine's Newton (pricing/bond).
inline double benchmark_yield(const AssetSwapConvention& conv, const BondRef& bond,
                              const market::Market& mkt) {
  const build::Date settle = build::advance_bd(conv.settle_calendar, mkt.today(), conv.settle_lag);
  const build::BuiltBond bb = build::bond_from_convention(bond.yield_conv, mkt.today(), settle, bond.issue,
                                                          bond.maturity, bond.coupon);
  const double clean = mkt.quote(bond.id).mid();  // bond quotes are CLEAN prices, per unit notional
  return pricing::bond_yield_from_clean(bb.yield, clean);
}

// Everything a caller needs from deriving one swap-spread row set.
struct DerivedAssetSwap {
  build::AssetSwapSpread rows;  // {pin, asw} — add both to the bundle (pin's row = bond bucket, asw = basis)
  double bond_yield = 0.0;      // the precomputed benchmark yield (the pin target)
  double spread = 0.0;          // the quoted swap spread (from the Market)
};

// Derive the swap-spread rows for one benchmark. `spot_swap` is the matched par swap on the swap curve
// (built by the general problem-derivation from the convention's swap_index + tenor); `factor_curve`/`anchor`
// locate the govvie yield factor; `spread_quote_id` names the Market quote holding the quoted spread.
inline DerivedAssetSwap derive_asset_swap(const AssetSwapConvention& conv, const BondRef& bond,
                                          const cal::Instrument& spot_swap, int factor_curve, double anchor,
                                          const market::Market& mkt, const std::string& spread_quote_id) {
  DerivedAssetSwap out;
  out.bond_yield = benchmark_yield(conv, bond, mkt);
  out.spread = mkt.quote(spread_quote_id).mid();
  out.rows = build::asset_swap_spread(spot_swap, factor_curve, anchor, out.bond_yield, out.spread);
  return out;
}

// Build every bond in the universe (curve-space cashflows) and pull its market clean price from the Market —
// the shared front half of both the spline and the parametric minimum-pricing-error fits.
inline void load_universe(const AssetSwapConvention& conv, const std::vector<BondRef>& universe,
                          const market::Market& mkt, std::vector<pricing::Bond>& bonds, Eigen::VectorXd& mc) {
  bonds.clear();
  bonds.reserve(universe.size());
  mc.resize(static_cast<int>(universe.size()));
  const build::Date settle = build::advance_bd(conv.settle_calendar, mkt.today(), conv.settle_lag);
  for (std::size_t b = 0; b < universe.size(); ++b) {
    const BondRef& br = universe[b];
    const build::BuiltBond bb = build::bond_from_convention(br.yield_conv, mkt.today(), settle, br.issue,
                                                            br.maturity, br.coupon);
    bonds.push_back(bb.curve);
    mc[static_cast<int>(b)] = mkt.quote(br.id).mid();
  }
}

// Assemble a SPLINE minimum-pricing-error govvie fit over a bond UNIVERSE priced from the Market. The
// returned GovvieBondFit is driven by cal::calibrate like any Problem; after the fit,
// portfolio::CompiledBondBook::z_spreads() over the SAME (meeting, back) topology gives the per-bond RV
// ladder. `weight` (optional, per bond) lets a caller down-weight illiquid/off-the-run bonds.
inline cal::GovvieBondFit make_govvie_fit(const AssetSwapConvention& conv, const std::vector<BondRef>& universe,
                                          const market::Market& mkt, const std::vector<double>& meeting,
                                          const std::vector<double>& back,
                                          const std::vector<double>& weight = {}) {
  cal::GovvieBondFit fit;
  fit.model = cal::CurveModel::Spline;
  fit.regions = curve::flat_hermite(meeting, back);
  load_universe(conv, universe, mkt, fit.bonds, fit.market_clean);
  if (!weight.empty()) fit.weight = Eigen::Map<const Eigen::VectorXd>(weight.data(), weight.size());
  return fit;
}

// Assemble a PARAMETRIC (Nelson-Siegel / Svensson) minimum-pricing-error fit: the govvie curve is a few
// time-stable parameters, calibrated to the whole universe's market prices. This is the fair-value-curve RV
// form — the fitted parameters (level/slope/curvature…) are the state, and each bond's leftover residual /
// z-spread is its richness-cheapness. Same LM, same z-spread ladder as the spline; only the curve differs.
inline cal::GovvieBondFit make_parametric_fit(const AssetSwapConvention& conv,
                                              const std::vector<BondRef>& universe, const market::Market& mkt,
                                              cal::CurveModel model, double tau1, double tau2 = 5.0,
                                              const std::vector<double>& weight = {}) {
  cal::GovvieBondFit fit;
  fit.model = model;
  fit.tau1 = tau1;
  fit.tau2 = tau2;
  load_universe(conv, universe, mkt, fit.bonds, fit.market_clean);
  if (!weight.empty()) fit.weight = Eigen::Map<const Eigen::VectorXd>(weight.data(), weight.size());
  return fit;
}

}  // namespace swaps::derive
