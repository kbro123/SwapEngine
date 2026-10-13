// g10_market.hpp -- the G10 rates desk as an ENGINE-ONLY fixture, with and without year-end turns.
//
// bench/g10_desk_ql_bench.cpp builds the same 8-curve desk beside QuantLib and is held to QuantLib's answer, so its
// instrument set is limited to what QuantLib's rate helpers can also bootstrap: par swaps only. A real desk carries
// a YEAR-END TURN on each major curve, which QuantLib's bootstrap has no concept of. This fixture builds the desk
// twice from the SAME DB conventions and pillars, QuantLib-free:
//   * plain    -- 8 curves x 9 par swaps (72 rows, 72 knots): the shape the QuantLib-referenced metrics time
//   * g4turns  -- the same, plus on each G4 curve (USD SOFR, EUR ESTR, GBP SONIA, JPY TONA) one year-end turn over
//                 [2026-12-31, 2027-01-04] with its bracketing December-to-January compounded rate observation and
//                 its TurnJump pin (docs/turns-calibration.md: a turn is observable only if instruments bracket it)
//                 -- 4 turns, +8 rows, +4 state (the jumps), the EURIBOR 3M curve discounting on the turned ESTR
// so the pair isolates what turns cost on the cold solve, the streamed tick and the Jacobian with nothing else
// varying. Maturities resolve through the engine's own calendars (build::resolve), as the shape ladder's do.
//
// BOTH RUNGS ARE TIMED UNDER LIGHT SMOOTHING (owner's challenge, 2026-10-13). The plain desk is SQUARE (one knot per
// par swap), and on a square problem the streamer lands exactly on frozen Newton steps however far the market moves,
// so its 25 bp refresh measured 26 us against 877 us for the turned desk -- not what the turns cost but what leaving
// the square regime costs, and no desk streams square: smoothing rows are rows, so every production bundle is
// over-determined (CLAUDE.md section 7a, "nobody streams unsmoothed"). Under Light on both (region data, 0.5) the
// same pair reads 1.28 -> 1.44 ms on the refresh, 13.8 -> 14.4 us on the tick: the honest turn cost. `light=false`
// builds the raw pair, which the bench's premise check uses to prove both desks calibrate back to their generating
// state exactly (a smoothed fit cannot).
#pragma once

#include <string>
#include <vector>

#include "shape_ladder.hpp"
#include "swaps/calibration/regularize.hpp"

namespace swaps::shapes {

namespace g10 {
enum Role { USD_SOFR = 0, EUR_ESTR, EUR_3M, GBP_SONIA, JPY_TONA, CHF_SARON, CAD_CORRA, AUD_AONIA, NROLES };
enum Ccy { USD = 0, EUR = 1, GBP = 2, JPY = 3, CHF = 4, CAD = 5, AUD = 6 };
struct Leg {
  Role role;
  Ccy ccy;
  const char* ccy_code;
  const char* index;  // conventions-DB INDEX id (swap_conv derives the par product)
  Role discount;
  bool g4;            // carries the year-end turn in the turned variant
};
inline const std::vector<Leg>& legs() {
  static const std::vector<Leg> L = {
      {USD_SOFR, USD, "USD", "USD-SOFR", USD_SOFR, true},   {EUR_ESTR, EUR, "EUR", "EUR-ESTR", EUR_ESTR, true},
      {EUR_3M, EUR, "EUR", "EUR-EURIBOR-3M", EUR_ESTR, false}, {GBP_SONIA, GBP, "GBP", "GBP-SONIA", GBP_SONIA, true},
      {JPY_TONA, JPY, "JPY", "JPY-TONA", JPY_TONA, true},   {CHF_SARON, CHF, "CHF", "CHF-SARON", CHF_SARON, false},
      {CAD_CORRA, CAD, "CAD", "CAD-CORRA", CAD_CORRA, false}, {AUD_AONIA, AUD, "AUD", "AUD-AONIA", AUD_AONIA, false},
  };
  return L;
}
inline std::vector<std::string> tenors() { return {"1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"}; }
}  // namespace g10

// The desk; `g4_turns` adds the four year-end turns. Every curve is outright (the EURIBOR curve forecasts 3M and
// discounts on ESTR by ROLE, not as a spread), as in g10_desk_ql_bench.
inline Shape g10_desk(bool g4_turns, bool light = true) {
  cal::BundleProblem p;
  p.curves.resize(g10::NROLES);
  const b::Date turn_s = b::Date::from_iso("2026-12-31"), turn_e = b::Date::from_iso("2027-01-04");
  const b::Date obs_s = b::Date::from_iso("2026-12-15"), obs_e = b::Date::from_iso("2027-01-15");
  for (const g10::Leg& lg : g10::legs()) {
    const b::SwapConv conv = b::swap_conv(lg.ccy_code, lg.index);
    const std::vector<double> knots = detail::add_par_swaps(p, conv, lg.role, lg.discount, g10::tenors());
    px::CurveStructure s = detail::spec(knots);
    s.currency = lg.ccy;
    if (g4_turns && lg.g4) {
      s.turns = {px::Turn{b::curve_time(detail::vd(), turn_s), b::curve_time(detail::vd(), turn_e)}};
      // the bracket: a December-to-January compounded observation on the index's own day count and calendar
      p.instruments.push_back(b::rate_instrument(
          lg.role, b::observation(detail::vd(), obs_s, obs_e, "compounded", 0.0, b::index_day_count(lg.index), b::index_calendar(lg.index)), 0.03));
      p.instruments.push_back(b::turn_jump(lg.role, 0, 0.0));  // the pin (its market is set at x_true: a ~20 bp turn)
    }
    p.curves[static_cast<std::size_t>(lg.role)] = std::move(s);
  }
  if (light) cal::stamp_region_smoothing(p, 0.5);  // every curve's Hermite region: the Light weight, as a desk streams
  return detail::make(g4_turns ? "g10_g4turns" : "g10_noturns",
                      g4_turns ? "G10 desk + a year-end turn on each G4 curve (bracket observation + TurnJump pin)"
                               : "G10 desk: 8 curves over 7 currencies, 9 par swaps each, no turns",
                      std::move(p));
}

}  // namespace swaps::shapes
