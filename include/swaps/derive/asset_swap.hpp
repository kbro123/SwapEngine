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
#include "swaps/market/market.hpp"
#include "swaps/pricing/bond.hpp"
#include "swaps/pricing/bond_future.hpp"

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

// A bond's identity is build::BondId (build/bond.hpp) — the ONE canonical identity type: convention-keyed,
// when-issued-capable (first_coupon), independent of any market snapshot. Its live PRICE is a Quote in the
// Market, keyed by BondId::id — ref data and market data stay separate. (This layer previously carried its
// own BondRef restating a subset of the terms; folded per the CurveStructure de-dup rule.)

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

// The benchmark's STREET yield, precomputed from the Market: build the bond on its convention, read its
// market CLEAN price (Market.quote(id).mid()), and invert to yield via the engine's Newton (pricing/bond).
inline double benchmark_yield(const AssetSwapConvention& conv, const build::BondId& bond,
                              const market::Market& mkt) {
  const build::Date settle = settlement_date(conv, mkt.today());
  const build::BuiltBond bb = build::build_bond(bond, mkt.today(), settle);  // WI-aware via the identity
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
inline DerivedAssetSwap derive_asset_swap(const AssetSwapConvention& conv, const build::BondId& bond,
                                          const cal::Instrument& spot_swap, int factor_curve, double anchor,
                                          const market::Market& mkt, const std::string& spread_quote_id) {
  // The convention's spread_type SELECTS the derivation. The yield-vs-matched-swap forms are wired here
  // (HeadlineYield / MatchedMaturity share this derivation — they differ in which swap the caller matched);
  // Invoice (bond-future CTD FORWARD yield) has its own entry point derive_invoice_asset_swap() below, which
  // needs the futures contract's basket + delivery data this signature does not carry; ParAssetSwap
  // (endogenous) is still unwired. Fail loudly rather than silently deriving the wrong number.
  if (conv.type == SwapSpreadType::Invoice)
    throw std::invalid_argument("derive_asset_swap: use derive_invoice_asset_swap() for the Invoice spread");
  if (conv.type != SwapSpreadType::HeadlineYield && conv.type != SwapSpreadType::MatchedMaturity)
    throw std::invalid_argument("derive_asset_swap: this spread_type's derivation is not implemented yet");
  DerivedAssetSwap out;
  out.bond_yield = benchmark_yield(conv, bond, mkt);
  out.spread = mkt.quote(spread_quote_id).mid();
  out.rows = build::asset_swap_spread(spot_swap, factor_curve, anchor, out.bond_yield, out.spread);
  return out;
}

// =================================================================================================
// INVOICE spread — the bond-future CTD FORWARD yield feeds the same asset-swap basis machinery.
// =================================================================================================
// An INVOICE (bond-future) spread quotes the cheapest-to-deliver's FORWARD yield to the future's delivery
// date against the matched swap — the exact analogue of the headline benchmark yield, but carried forward.
// The CTD is chosen off the deliverable basket by max implied repo (pricing/bond_future.hpp); its forward
// yield is its spot clean price carried to delivery at `repo` (ACT/360, interim coupons reinvested), then
// inverted through the bond kernel as of the DELIVERY settlement. That forward yield is the pin target of
// the SAME build::asset_swap_spread rows, so the Invoice basis buckets identically to the headline ASW.

// The CTD's forward yield to `delivery`: carry its spot dirty price forward at `repo`, subtract accrued at
// delivery, invert the resulting forward clean through the yield kernel (bond rebuilt as of delivery). The
// futures price does NOT enter — the forward yield is a pure cash-carry number; the futures/CF only enter
// the basket's CTD selection and the invoice/basis. Seasoned CTD (regular coupons); interim coupons in
// (settle, delivery] are enumerated off the coupon grid and reinvested at repo.
inline double ctd_forward_yield(const AssetSwapConvention& conv, const build::BondId& ctd,
                                const build::Date& delivery, double repo, const market::Market& mkt) {
  const build::Date settle = settlement_date(conv, mkt.today());
  const build::BuiltBond bb_now = build::build_bond(ctd, mkt.today(), settle);
  const double clean_now = mkt.quote(ctd.id).mid();        // CTD live CLEAN price, per unit notional
  const double dirty_now = clean_now + bb_now.accrued;
  const double days = double(delivery - settle);
  const int freq = int(bb_now.yield.conv.freq + 0.5);
  const double cpn_per_period = ctd.coupon / double(freq);

  // Carry the dirty price to delivery at repo (the CURRENCY's repo day-count basis, currencies[].repo_day_count),
  // reinvesting any interim coupon paid in (settle, delivery].
  const double repo_basis = build::day_count_basis(std::string(conventions::require_currency(conv.currency).repo_day_count));
  double fwd_dirty = dirty_now * (1.0 + repo * days / repo_basis);
  build::Date ref_start;
  for (const build::Date& cd : build::coupon_dates_backward(ctd.issue, ctd.maturity, freq, ref_start))
    if (cd > settle && cd <= delivery)
      fwd_dirty -= cpn_per_period * (1.0 + repo * double(delivery - cd) / repo_basis);

  // Invert the forward CLEAN price through the CTD rebuilt as of delivery (accrued + flows at delivery).
  const build::BuiltBond bb_del = build::build_bond(ctd, delivery, delivery);
  return pricing::bond_yield_from_clean(bb_del.yield, fwd_dirty - bb_del.accrued);
}

// Derive the Invoice-spread {pin, asw} rows: identical to derive_asset_swap but the pinned yield is the CTD
// FORWARD yield (not a spot benchmark yield). `ctd` is the cheapest-to-deliver (typically chosen off the
// basket via pricing::select_ctd on the conversion-factor/basis analytics); `delivery` is the futures'
// delivery date; `repo` is the funding rate used to carry the CTD forward.
inline DerivedAssetSwap derive_invoice_asset_swap(const AssetSwapConvention& conv, const build::BondId& ctd,
                                                  const build::Date& delivery, double repo,
                                                  const cal::Instrument& spot_swap, int factor_curve,
                                                  double anchor, const market::Market& mkt,
                                                  const std::string& spread_quote_id) {
  if (conv.type != SwapSpreadType::Invoice)
    throw std::invalid_argument("derive_invoice_asset_swap: convention.type must be Invoice");
  DerivedAssetSwap out;
  out.bond_yield = ctd_forward_yield(conv, ctd, delivery, repo, mkt);
  out.spread = mkt.quote(spread_quote_id).mid();
  out.rows = build::asset_swap_spread(spot_swap, factor_curve, anchor, out.bond_yield, out.spread);
  return out;
}

// Build every bond in the universe (curve-space cashflows) and pull its market clean price from the Market —
// the shared front half of both the spline and the parametric minimum-pricing-error fits.
inline void load_universe(const AssetSwapConvention& conv, const std::vector<build::BondId>& universe,
                          const market::Market& mkt, std::vector<pricing::Bond>& bonds, Eigen::VectorXd& mc) {
  bonds.clear();
  bonds.reserve(universe.size());
  mc.resize(static_cast<int>(universe.size()));
  const build::Date settle = settlement_date(conv, mkt.today());
  for (std::size_t b = 0; b < universe.size(); ++b) {
    const build::BondId& br = universe[b];
    bonds.push_back(build::build_bond(br, mkt.today(), settle).curve);  // WI-aware via the identity
    mc[static_cast<int>(b)] = mkt.quote(br.id).mid();
  }
}

// Assemble a SPLINE minimum-pricing-error govvie fit over a bond UNIVERSE priced from the Market. The
// returned GovvieBondFit is driven by cal::calibrate like any Problem; after the fit,
// portfolio::CompiledBondBook::z_spreads() over the SAME (meeting, back) topology gives the per-bond RV
// ladder. `weight` (optional, per bond) lets a caller down-weight illiquid/off-the-run bonds.
inline cal::GovvieBondFit make_govvie_fit(const AssetSwapConvention& conv, const std::vector<build::BondId>& universe,
                                          const market::Market& mkt, const std::vector<double>& meeting,
                                          const std::vector<double>& back,
                                          const std::vector<double>& weight = {}) {
  cal::GovvieBondFit fit;
  fit.regions = curve::flat_hermite(meeting, back);
  load_universe(conv, universe, mkt, fit.bonds, fit.market_clean);
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
                                                         const market::Market& mkt, double tau1,
                                                         double tau2 = 5.0,
                                                         const std::vector<double>& weight = {}) {
  cal::ParametricBondFit<Model> fit;
  fit.tau1 = tau1;
  fit.tau2 = tau2;
  load_universe(conv, universe, mkt, fit.bonds, fit.market_clean);
  if (!weight.empty()) fit.weight = Eigen::Map<const Eigen::VectorXd>(weight.data(), weight.size());
  return fit;
}

// =================================================================================================
// swap_spread — the `swap_spread` verb's whole computation (E7 stage 3.4).
// =================================================================================================
// The benchmark bond + its clean price and the quoted spread; the matched spot-start par swap built from the
// index's conventions DB row to `tenor` from SPOT on the index calendar; the govvie factor anchor (default: that
// swap's maturity in curve time). KNOWN GAP, kept bit-for-bit pending an owner decision: MatchedMaturity also
// matches the TENOR swap (TASKS-ENGINE E7 3.4 finding (1)) -- it returns headline numbers under its own label.
inline constexpr const char* kSwapSpreadQuoteId = "spread";  // the Market key the quoted spread is stored under

struct SwapSpreadRequest {
  build::Date value_date;
  build::BondId bond;  // the benchmark; bond.yield_conv is also the settlement convention
  SettlementOverride settlement;
  SwapSpreadType type = SwapSpreadType::HeadlineYield;
  double clean = 0.0;   // benchmark clean price -- REQUIRED, > 0
  double spread = 0.0;  // the quoted swap spread
  std::string index;    // the matched swap's index -- REQUIRED, e.g. USD-SOFR
  std::string tenor;    // REQUIRED, e.g. 5Y
  int swap_curve = -1, factor_curve = -1;  // bundle roles -- REQUIRED
  std::optional<double> anchor;             // absent => curve_time(value_date, swap maturity)
};
struct SwapSpreadResult {
  DerivedAssetSwap derived;
  double anchor = 0.0;
};

inline SwapSpreadResult swap_spread(const SwapSpreadRequest& r) {
  if (!(r.clean > 0.0)) throw std::invalid_argument("swap_spread: needs a positive benchmark 'clean' price");
  if (r.index.empty()) throw std::invalid_argument("swap_spread: missing swap 'index'");
  if (r.tenor.empty()) throw std::invalid_argument("swap_spread: missing swap 'tenor' (e.g. 5Y)");
  if (r.swap_curve < 0 || r.factor_curve < 0)
    throw std::invalid_argument("swap_spread: needs 'swap_curve' and 'factor_curve' (bundle curve roles)");
  if (r.bond.id == kSwapSpreadQuoteId)
    throw std::invalid_argument("swap_spread: a bond id may not be the reserved quote name 'spread'");
  AssetSwapConvention conv = asset_swap_convention(r.bond.yield_conv, r.settlement);
  conv.type = r.type;
  conv.swap_index = r.index;
  const build::SwapConv sconv = build::Index(r.index).par_convention().resolve();
  const build::Date mat = build::resolve(r.tenor, r.value_date, sconv.calendar, sconv.bdc, sconv.spot_lag);
  const cal::Instrument spot_swap = build::par_swap(r.value_date, sconv, mat, r.swap_curve, r.swap_curve, 0.0);
  SwapSpreadResult out;
  out.anchor = r.anchor.value_or(build::curve_time(r.value_date, mat));
  market::Market m;
  m.as_of(r.value_date);
  m.add_quote(r.bond.id, market::Quote::mid(r.clean));
  m.add_quote(kSwapSpreadQuoteId, market::Quote::mid(r.spread));
  out.derived = derive_asset_swap(conv, r.bond, spot_swap, r.factor_curve, out.anchor, m, kSwapSpreadQuoteId);
  return out;
}

}  // namespace swaps::derive
