#ifndef SWAPS_BUILD_SWAP_SPREAD_HPP
#define SWAPS_BUILD_SWAP_SPREAD_HPP
// swaps::build — the asset-swap / swap-spread BASIS instrument.
//
// A "headline swap spread" (USD 5y: the on-the-run 5y UST street yield vs the SPOT-START 5y SOFR par swap
// — note the swap is the 5y TENOR, a different maturity than the benchmark bond) is quoted as
//     spread = swap_rate − bond_yield.
// For the calibrated curve ALONE you could bake the target of an outright par swap as (bond_yield + spread)
// and be done. But then RISK is wrong. Risk here is bucketed PER CALIBRATION ROW (calibration/risk.hpp:
// d(NPV)/dq = d(NPV)/dx · J⁻¹, one bucket per residual), so a baked target has no spread row and the
// asset-swap basis has NO bucket — it smears into the outright swap deltas. To make the ASW risk a
// FIRST-CLASS BASIS, the bond must be a LIVE FACTOR the residual depends on, and the spread must be its own
// row. Then J⁻¹ reprojects portfolio risk onto {swap outright, bond outright, ASW basis} as distinct axes —
// exactly how tenor / xccy basis already bucket, because they too are their own rows over their own factor.
//
// THE PATTERN — built entirely from EXISTING generic kinds, so there is NO new QuoteKind and NO change to
// the compiled/AAD hot path (the outright swaps stay W-cacheable; only this basis row rides the AAD path,
// like FxForward / Portfolio already do):
//   * a 1-knot Flat "govvie yield factor" curve added to the bundle (govvie_factor_regions()). Its single
//     state variable IS the benchmark bond's yield, in rate space.
//   * a PIN row — QuoteKind::Rate on the factor curve, market = the bond (or forward) yield precomputed
//     from the Market — fixes the factor. Its risk bucket is the BOND OUTRIGHT delta.
//   * the ASW row — QuoteKind::Portfolio { +1·par_swap(swap curve), −1·Rate(factor) }, market = the quoted
//     spread. Model quote = swap_par_rate − bond_yield; residual = that − spread. Its bucket is the ASW
//     BASIS. Because the ASW references the swap curve, it PINS the swap curve at that pillar in place of an
//     outright — so at constant spread a move in the bond yield flows straight into the swap curve, which
//     is precisely what makes the asset swap a basis.
//
// The factor's Rate observation is SHARED between the pin and the ASW's factor component (asw_factor_obs()),
// so the ASW residual is EXACTLY swap_par_rate − pinned_yield − spread regardless of the rate transform's
// compounding convention: both read the identical rate(obs) off the identical flat curve.
//
// Which asset-swap CONVENTION picks the yield (headline OTR street yield, matched-maturity, invoice forward
// yield) and which benchmark bond is a per-currency data concern resolved when the target is computed from
// the Market; this builder is convention-agnostic — hand it a yield and a spread.

#include <utility>
#include <vector>

#include "swaps/build/instruments.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/curve/curve_module.hpp"

namespace swaps::build {

namespace cal = swaps::calibration;

// The bundle curve spec regions for a govvie yield FACTOR: a single Flat knot, so the curve is one free
// state variable that flat-extrapolates everywhere — the benchmark bond's yield, in rate space. Add it to a
// bundle as `p.curves.push_back({.base = -1, .regions = build::govvie_factor_regions(anchor)})`.
inline std::vector<curve::CurveModule> govvie_factor_regions(double anchor_t) {
  return {curve::CurveModule{{anchor_t}, curve::Scheme::Flat}};
}

// The shared Rate observation that reads the factor curve's yield. A single non-compounded window; on a flat
// curve rate(obs) is a smooth monotone map of the one factor forward, so pinning it to a yield is well-posed.
// The pin and the ASW's factor component MUST use this same obs so the spread residual is exact.
inline px::RateObservation asw_factor_obs(double anchor_t) {
  return plain_rate_obs(0.0, anchor_t);
}

// A headline / asset-swap spread expressed as two calibration rows over a shared bond-yield factor.
struct AssetSwapSpread {
  cal::Instrument pin;  // QuoteKind::Rate on the factor curve — pins the factor to `bond_yield`.
  cal::Instrument asw;  // QuoteKind::Portfolio {+1·spot_swap, −1·Rate(factor)} — the spread row.
};

// Build the pair. `spot_swap` is the ParRate swap on the swap curve (e.g. build::par_swap of the spread's
// TENOR); `factor_curve` is the bundle index of the govvie factor curve; `anchor_t` matches the factor
// curve's single knot; `bond_yield` is the benchmark yield (precomputed from the Market); `spread` is the
// quoted swap spread. Add BOTH rows to the bundle (pin first is conventional but the residual order is the
// caller's; risk buckets follow that order).
inline AssetSwapSpread asset_swap_spread(const cal::Instrument& spot_swap, int factor_curve, double anchor_t,
                                         double bond_yield, double spread) {
  const px::RateObservation obs = asw_factor_obs(anchor_t);

  // The factor component of the spread: reads the factor curve's yield via the SHARED obs. Only its model
  // quote enters the Portfolio sum, so its own `market` is irrelevant (left 0).
  cal::Instrument factor_leg = rate_instrument(factor_curve, obs, /*market=*/0.0);

  AssetSwapSpread out;
  out.pin = rate_instrument(factor_curve, obs, bond_yield);  // rate(obs) == bond_yield pins the factor
  out.asw.quote = cal::QuoteKind::Portfolio;
  out.asw.combination = {{+1.0, spot_swap}, {-1.0, std::move(factor_leg)}};
  out.asw.market = spread;
  return out;
}

}  // namespace swaps::build

#endif  // SWAPS_BUILD_SWAP_SPREAD_HPP
