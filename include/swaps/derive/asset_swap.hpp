#pragma once
// derive/asset_swap.hpp — the market→calibration BRIDGE for asset swaps / swap spreads. This is the layer
// that turns a per-currency asset-swap CONVENTION + the market's numbers (the value date, a bond's clean
// price, a quoted spread) into the calibration rows (and, for RV, the minimum-pricing-error govvie fit). It
// depends on build/ (bond + swap-spread instrument builders) and calibration/ (the Problem types) — the
// composition layer above them. (It took a market::Market snapshot until 2026-11-01; the snapshot's two
// production readers only ever stored an as-of date and clean prices, so those are the arguments now.)
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
// The bond street-yield / price↔yield inversion is a PRECOMPUTE off the market price, never on the calibration
// hot path (the compiled solve for the swap curve never sees a bond).

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Core>

#include "swaps/build/bond.hpp"
#include "swaps/build/calendar.hpp"
#include "swaps/build/date.hpp"
#include "swaps/build/instruments.hpp"  // par_swap
#include "swaps/build/ref_data.hpp"     // Index
#include "swaps/build/schedule.hpp"     // resolve, curve_time
#include "swaps/build/swap_spread.hpp"
#include "swaps/calibration/bond_fit.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/bond.hpp"

namespace swaps::derive {

namespace build = swaps::build;
namespace cal = swaps::calibration;
namespace curve = swaps::curve;
namespace pricing = swaps::pricing;

// How a currency turns a bond yield + a matched swap into a swap-spread quote. HeadlineYield and MatchedMaturity are
// wired end to end (derive::spread_swap); the others name the remaining desk conventions so the type is stable.
enum class SwapSpreadType {
  HeadlineYield,    // OTR benchmark bond STREET yield vs the spot-start par swap of the same TENOR (USD).
  MatchedMaturity,  // the benchmark yield vs a par swap from the product's spot to the bond's OWN maturity.
  ParAssetSwap,     // the par-par asset-swap spread (bond repriced on the swap curve; endogenous, later).
  Invoice,          // bond-future CTD FORWARD yield to delivery vs the matched swap.
};

// A bond's identity is build::BondId (build/bond.hpp) — the ONE canonical identity type: convention-keyed,
// when-issued-capable (first_coupon), independent of any market data. Its live clean PRICE is a separate
// argument — ref data and market data stay separate. (This layer previously carried its own BondRef restating
// a subset of the terms; folded per the CurveStructure de-dup rule.)

// A per-currency asset-swap convention. Data — the benchmark selection, the govvie curve it references (by
// NAME, so it can be a 1-knot headline factor today and a fitted RV curve later), the settlement rule, and
// the swap index the spread is quoted against.
struct AssetSwapConvention {
  std::string currency;                                  // "USD"
  SwapSpreadType type = SwapSpreadType::HeadlineYield;
  std::string govvie_curve;                              // the govvie curve this spread references, by name
  std::string swap_index;                                // the matched swap's index, e.g. "USD-SOFR"
  std::string settle_calendar;                           // conventions-DB calendar key for settlement (REQUIRED)
  int settle_lag = -1;                                   // business days to settlement (REQUIRED; bonds[].settle_lag)
};

// Settlement terms a request may override; an absent field is the bond convention row's (bonds[].calendar /
// bonds[].settle_lag).
struct SettlementOverride {
  std::optional<std::string> calendar;
  std::optional<int> lag;
};

// The asset-swap convention a BOND convention implies: its currency, and its settlement calendar / lag unless
// overridden. The ONE place a bonds[] row becomes settlement terms (E7 stage 3.4 -- the RV verbs each re-read the
// row and re-applied the overrides by hand).
inline AssetSwapConvention asset_swap_convention(std::string_view bond_convention, const SettlementOverride& o = {}) {
  if (bond_convention.empty())
    throw std::invalid_argument("bond convention: needs 'convention' (a bonds[] row id, e.g. US-TREASURY)");
  const conventions::BondConv bc = conventions::require_bond(bond_convention);
  AssetSwapConvention conv;
  conv.currency = std::string(bc.currency);
  conv.settle_calendar = o.calendar ? *o.calendar : std::string(bc.calendar);
  conv.settle_lag = o.lag ? *o.lag : bc.settle_lag;
  return conv;
}

// Settlement for a trade done `today` under `conv` -- the ONE business-day roll (it was written out four times).
inline build::Date settlement_date(const AssetSwapConvention& conv, const build::Date& today) {
  return build::advance_bd(conv.settle_calendar, today, conv.settle_lag);
}

// The benchmark's STREET yield: build the bond on its convention as of `today`, and invert its market CLEAN
// price (per unit notional) to yield via the engine's Newton (pricing/bond).
inline double benchmark_yield(const AssetSwapConvention& conv, const build::BondId& bond, const build::Date& today,
                              double clean) {
  const build::Date settle = settlement_date(conv, today);
  const build::BuiltBond bb = build::build_bond(bond, today, settle);  // WI-aware via the identity
  return pricing::bond_yield_from_clean(bb.yield, clean);
}

// Everything a caller needs from deriving one swap-spread row set.
struct DerivedAssetSwap {
  build::AssetSwapSpread rows;  // {pin, asw} — add both to the bundle (pin's row = bond bucket, asw = basis)
  double bond_yield = 0.0;      // the precomputed benchmark yield (the pin target)
  double spread = 0.0;          // the quoted swap spread
};

// Derive the swap-spread rows for one benchmark. `spot_swap` is the matched par swap on the swap curve
// (built by the general problem-derivation from the convention's swap_index + tenor); `factor_curve`/`anchor`
// locate the govvie yield factor; `clean` is the benchmark's market clean price and `spread` the quoted spread.
inline DerivedAssetSwap derive_asset_swap(const AssetSwapConvention& conv, const build::BondId& bond,
                                          const cal::Instrument& spot_swap, int factor_curve, double anchor,
                                          const build::Date& today, double clean, double spread) {
  // The convention's spread_type SELECTS the derivation. The yield-vs-matched-swap forms are wired here
  // (HeadlineYield / MatchedMaturity share this derivation — they differ in which swap the caller matched);
  // Invoice (bond-future CTD FORWARD yield) and ParAssetSwap (endogenous) are named conventions without a
  // derivation (the Invoice path's forward-yield code had no consumer and was deleted on 2026-11-01, review
  // item 1.8). Fail loudly rather than silently deriving the wrong number.
  if (conv.type != SwapSpreadType::HeadlineYield && conv.type != SwapSpreadType::MatchedMaturity)
    throw std::invalid_argument("derive_asset_swap: this spread_type's derivation is not implemented yet");
  DerivedAssetSwap out;
  out.bond_yield = benchmark_yield(conv, bond, today, clean);
  out.spread = spread;
  out.rows = build::asset_swap_spread(spot_swap, factor_curve, anchor, out.bond_yield, out.spread);
  return out;
}

// Build every bond in the universe (curve-space cashflows) as of `today` beside its market clean price (`clean`
// is aligned with `universe`) — the shared front half of both the spline and the parametric
// minimum-pricing-error fits.
inline void load_universe(const AssetSwapConvention& conv, const std::vector<build::BondId>& universe,
                          const build::Date& today, const std::vector<double>& clean,
                          std::vector<pricing::Bond>& bonds, Eigen::VectorXd& mc) {
  if (clean.size() != universe.size())
    throw std::invalid_argument("load_universe: one clean price per bond (clean.size() != universe.size())");
  bonds.clear();
  bonds.reserve(universe.size());
  mc.resize(static_cast<int>(universe.size()));
  const build::Date settle = settlement_date(conv, today);
  for (std::size_t b = 0; b < universe.size(); ++b) {
    bonds.push_back(build::build_bond(universe[b], today, settle).curve);  // WI-aware via the identity
    mc[static_cast<int>(b)] = clean[b];
  }
}

// Assemble a SPLINE minimum-pricing-error govvie fit over a bond UNIVERSE at its market clean prices. The
// returned GovvieBondFit is driven by cal::calibrate like any Problem; after the fit,
// portfolio::CompiledBondBook::z_spreads() over the SAME (meeting, back) topology gives the per-bond RV
// ladder. `weight` (optional, per bond) lets a caller down-weight illiquid/off-the-run bonds.
inline cal::GovvieBondFit make_govvie_fit(const AssetSwapConvention& conv, const std::vector<build::BondId>& universe,
                                          const build::Date& today, const std::vector<double>& clean,
                                          const std::vector<double>& meeting, const std::vector<double>& back,
                                          const std::vector<double>& weight = {}) {
  cal::GovvieBondFit fit;
  fit.regions = curve::flat_hermite(meeting, back);
  load_universe(conv, universe, today, clean, fit.bonds, fit.market_clean);
  if (!weight.empty()) fit.weight = Eigen::Map<const Eigen::VectorXd>(weight.data(), weight.size());
  return fit;
}

// Assemble a PARAMETRIC minimum-pricing-error fit: the govvie curve is a few time-stable parameters,
// calibrated to the whole universe's market prices. This is the fair-value-curve RV form — the fitted
// parameters (level/slope/curvature…) are the state, and each bond's leftover residual / z-spread is its
// richness-cheapness. The MODEL is the template parameter (curve::NelsonSiegel, curve::Svensson, or any
// Scalar-templated family with n_params/set_params/discount) — same LM, same universe loading as the
// spline; only the curve type differs:  auto fit = make_parametric_fit<curve::NelsonSiegel>(...).
template <template <class> class Model>
inline cal::ParametricBondFit<Model> make_parametric_fit(const AssetSwapConvention& conv,
                                                         const std::vector<build::BondId>& universe,
                                                         const build::Date& today, const std::vector<double>& clean,
                                                         double tau1, double tau2 = 5.0,
                                                         const std::vector<double>& weight = {}) {
  cal::ParametricBondFit<Model> fit;
  fit.tau1 = tau1;
  fit.tau2 = tau2;
  load_universe(conv, universe, today, clean, fit.bonds, fit.market_clean);
  if (!weight.empty()) fit.weight = Eigen::Map<const Eigen::VectorXd>(weight.data(), weight.size());
  return fit;
}

// =================================================================================================
// swap_spread — the `swap_spread` verb's whole computation (E7 stage 3.4; matched maturity 2026-09-14).
// =================================================================================================
// The benchmark bond + its clean price and the quoted spread, and the par swap the spread is matched to: the index's
// conventions-DB par product, single-curve on `swap_curve`, starting at THAT PRODUCT's spot (its spot_lag business days
// on its calendar -- per currency, never an assumed T+2 or a US calendar).
//   * HeadlineYield   -- the swap of `tenor` (spot + tenor on the product calendar, adjusted), rolled forward from spot.
//                        `tenor` is REQUIRED.
//   * MatchedMaturity -- the swap TERMINATING ON THE BOND'S MATURITY: rolled BACKWARD from the UNADJUSTED maturity (a
//                        short FRONT stub, roll day = the maturity's day of month, no end-of-month roll), the
//                        termination rolled by the product's bdc -- QuantLib's DateGeneration::Backward and
//                        build::asset_swap_float_leg's rule. `tenor` must be ABSENT. Month/year-frequency products only.
// spread = swap par rate - benchmark yield for both (build/swap_spread.hpp). The govvie factor anchor defaults to the
// swap's termination in curve time. The bond's and the index's currencies need not match (owner decision 2026-09-14:
// a bond against another currency's swap curve is a wanted use; its cross-currency form is designed separately).
struct SwapSpreadRequest {
  build::Date value_date;
  build::BondId bond;  // the benchmark; bond.yield_conv is also the settlement convention
  SettlementOverride settlement;
  SwapSpreadType type = SwapSpreadType::HeadlineYield;
  double clean = 0.0;   // benchmark clean price -- REQUIRED, > 0
  double spread = 0.0;  // the quoted swap spread
  std::string index;    // the matched swap's index -- REQUIRED, e.g. USD-SOFR
  std::string tenor;    // HeadlineYield: REQUIRED, e.g. 5Y. MatchedMaturity: must be empty
  int swap_curve = -1, factor_curve = -1;  // bundle roles -- REQUIRED
  std::optional<double> anchor;             // absent => curve_time(value_date, the swap's termination)
};
struct SwapSpreadResult {
  DerivedAssetSwap derived;
  double anchor = 0.0;
  build::Date swap_maturity;  // the matched swap's termination (its last accrual end), business-day adjusted
};

// The par swap a swap-spread quote is matched to, and its termination.
struct SpreadSwap {
  cal::Instrument swap;
  build::Date maturity;
};

// The ONE place a spread type picks its swap. Bond-free apart from the bond's maturity, so it is the currency-generic
// seam the per-currency tests drive.
inline SpreadSwap spread_swap(const SwapSpreadRequest& r, const build::SwapConv& sconv) {
  if (r.type == SwapSpreadType::HeadlineYield) {
    if (r.tenor.empty()) throw std::invalid_argument("swap_spread: missing swap 'tenor' (e.g. 5Y)");
    const build::Date mat = build::resolve(r.tenor, r.value_date, sconv.calendar, sconv.bdc, sconv.spot_lag);
    return {build::par_swap(r.value_date, sconv, mat, r.swap_curve, r.swap_curve, 0.0), mat};
  }
  if (r.type == SwapSpreadType::MatchedMaturity) {
    if (!r.tenor.empty())
      throw std::invalid_argument("swap_spread: 'tenor' names the headline swap; a matched_maturity spread matches the "
                                  "bond's own maturity (drop 'tenor')");
    // A backward roll from a maturity is a whole-month roll; a zero_coupon product has one period and is rule-invariant.
    if (!sconv.zero_coupon &&
        (build::tok_step(sconv.fixed_freq_tok).days || build::tok_step(sconv.float_freq_tok).days))
      throw std::invalid_argument("swap_spread: matched_maturity rolls the swap back from the bond's maturity by whole "
                                  "months; product '" + sconv.product_id + "' rolls by days -- no matched swap");
    const build::Date spot = build::spot_date(r.value_date, sconv.calendar, sconv.spot_lag);
    const build::Date maturity = build::adjust(sconv.calendar, r.bond.maturity, sconv.bdc);
    if (!(maturity > spot))
      throw std::invalid_argument("swap_spread: the bond matures on or before the swap's spot date (product '" +
                                  sconv.product_id + "') -- there is no matched swap");
    build::ScheduleRule from_maturity;
    from_maturity.side = build::StubSide::Front;
    // The UNADJUSTED maturity: the schedule rolls the termination itself and takes the roll day from the date given.
    return {build::par_swap(r.value_date, sconv, r.bond.maturity, r.swap_curve, r.swap_curve, 0.0,
                            /*float_spread=*/0.0, /*notionals=*/{}, /*fixed_freq=*/"", from_maturity),
            maturity};
  }
  throw std::invalid_argument("swap_spread: spread_type is not this verb's (headline | matched_maturity)");
}

inline SwapSpreadResult swap_spread(const SwapSpreadRequest& r) {
  if (!(r.clean > 0.0)) throw std::invalid_argument("swap_spread: needs a positive benchmark 'clean' price");
  if (r.index.empty()) throw std::invalid_argument("swap_spread: missing swap 'index'");
  if (r.swap_curve < 0 || r.factor_curve < 0)
    throw std::invalid_argument("swap_spread: needs 'swap_curve' and 'factor_curve' (bundle curve roles)");
  AssetSwapConvention conv = asset_swap_convention(r.bond.yield_conv, r.settlement);
  conv.type = r.type;
  conv.swap_index = r.index;
  const build::SwapConv sconv = build::Index(r.index).swap_conv();
  const SpreadSwap m = spread_swap(r, sconv);
  SwapSpreadResult out;
  out.swap_maturity = m.maturity;
  out.anchor = r.anchor.value_or(build::curve_time(r.value_date, m.maturity));
  out.derived = derive_asset_swap(conv, r.bond, m.swap, r.factor_curve, out.anchor, r.value_date, r.clean, r.spread);
  return out;
}

}  // namespace swaps::derive
