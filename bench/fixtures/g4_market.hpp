// g4_market.hpp -- the G4 rates desk as an ENGINE-ONLY fixture: the hardest bundle the engine is asked to calibrate.
//
// Four currencies (USD, EUR, GBP, JPY), fourteen curves, every curve family a desk carries, calibrated JOINTLY in one
// BundleProblem, QuantLib-free (markets are the model quotes at a known x_true, as the shape ladder's are):
//
//   RFR OIS curves (4, outright)   USD SOFR, EUR ESTR, GBP SONIA, JPY TONA. Each is the SHIPPED two-region shape: a
//                                  piecewise-FLAT front with one knot per central-bank meeting date (conventions DB
//                                  cb_schedules: FOMC / ECB / MPC / BoJ MPM, every meeting after the value date and
//                                  before the 1Y pillar) pinned by meeting-to-meeting compounded observations (the
//                                  FOMC-dated OIS a desk trades: [spot, m1], [m1, m2], ... each reads exactly one Flat
//                                  knot), then a SMOOTH back over the 1Y..30Y par-swap pillars whose scheme is the
//                                  fixture's parameter -- Hermite (the linear control), MonotoneCubic (Hyman: value-
//                                  dependent, piecewise-linear) or MonotoneConvex (Hagan-West: value-dependent and
//                                  NOT piecewise-linear, the AAD block). Each carries the year-end turn with its
//                                  bracket observation and TurnJump pin when `turns` is set (as g10_market.hpp does).
//   Tenor / index basis (3)        USD Fed funds as a spread over SOFR (6 monthly averaged futures + the FF/SOFR basis
//                                  2Y..30Y, the `averaged` rung); EUR 6M EURIBOR as a spread over ESTR pinned by 6M
//                                  IRS; EUR 3M EURIBOR as a spread over the 6M curve pinned by the 3s6s basis (the
//                                  liquid EUR tenor-basis chain). GBP and JPY carry no term index in the DB
//                                  (post-LIBOR), so no tenor basis is invented for them.
//   CCP discount basis (4)         One curve per currency for the OTHER clearing house: a spread over the RFR curve
//                                  pinned by the same par OIS discounted on it (CME-cleared SOFR vs LCH, Eurex vs LCH
//                                  for ESTR, CME vs LCH for SONIA, JSCC vs LCH for TONA). The CCP basis is a swap rate
//                                  difference, so its instrument IS the OIS with a different discount role -- a role
//                                  choice, not a new convention, hence no DB product.
//   Cross-currency (3, outright)   EUR-in-USD, GBP-in-USD, JPY-in-USD collateral curves: FX forwards 1W..1Y then MtM
//                                  xccy basis 2Y..30Y on the DB pairs EURUSD / GBPUSD / USDJPY (the JPY spot is quoted
//                                  here in USD per JPY, the engine's foreign-in-USD form).
//
// `vd_everywhere` puts the chosen back scheme on EVERY curve (spreads and collateral curves too); the default keeps the
// value-dependent scheme on the four RFR long ends only, which is what the owner asked to profile: flat policy steps in
// front, smooth long forwards behind. Smoothing (the K48 tie-breaker) is stamped on every non-Flat region as a desk
// streams it; the raw (unsmoothed) build is the premise check's: it must calibrate back to x_true exactly.
#pragma once

#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

#include "shape_ladder.hpp"
#include "swaps/calibration/regularize.hpp"
#include "swaps/conventions_db.hpp"

namespace swaps::shapes {

namespace g4 {
// Role order respects the bundle's rule that a spread curve's base is defined before it (CLAUDE.md 7b).
enum Role {
  USD_SOFR = 0, USD_FF, USD_CME,            // SOFR (outright), FF (spread/SOFR), CME-cleared discount (spread/SOFR)
  EUR_ESTR, EUR_6M, EUR_3M, EUR_EUREX,      // ESTR (outright), 6M (spread/ESTR), 3M (spread/6M), Eurex discount (spread/ESTR)
  GBP_SONIA, GBP_CME,                       // SONIA (outright), CME discount (spread/SONIA)
  JPY_TONA, JPY_JSCC,                       // TONA (outright), JSCC discount (spread/TONA)
  EUR_USD, GBP_USD, JPY_USD,                // the collateral curves (outright): FX forwards + MtM xccy basis
  NROLES
};
enum Ccy { USD = 0, EUR = 1, GBP = 2, JPY = 3 };
struct Rfr {
  Role role; Ccy ccy; const char* ccy_code; const char* index; Role ccp; const char* ccp_note;
};
inline const std::vector<Rfr>& rfrs() {
  static const std::vector<Rfr> R = {
      {USD_SOFR, USD, "USD", "USD-SOFR", USD_CME, "CME-cleared SOFR discounting (vs LCH)"},
      {EUR_ESTR, EUR, "EUR", "EUR-ESTR", EUR_EUREX, "Eurex-cleared ESTR discounting (vs LCH)"},
      {GBP_SONIA, GBP, "GBP", "GBP-SONIA", GBP_CME, "CME-cleared SONIA discounting (vs LCH)"},
      {JPY_TONA, JPY, "JPY", "JPY-TONA", JPY_JSCC, "JSCC-cleared TONA discounting (vs LCH)"},
  };
  return R;
}
struct Xccy { Role role; Role foreign; const char* pair; double usd_per_foreign; };
inline const std::vector<Xccy>& xccys() {
  static const std::vector<Xccy> X = {
      {EUR_USD, EUR_ESTR, "EURUSD", 1.08}, {GBP_USD, GBP_SONIA, "GBPUSD", 1.27}, {JPY_USD, JPY_TONA, "USDJPY", 1.0 / 150.0},
  };
  return X;
}
struct Opts {
  swaps::curve::Scheme back = swaps::curve::Scheme::MonotoneConvex;  // the RFR long ends' scheme
  bool vd_everywhere = false;  // the same scheme on every curve, not only the RFR long ends
  bool turns = true;           // a year-end turn on each RFR curve (bracket observation + TurnJump pin)
  bool smoothing = true;       // the K48 tie-breaker stamped on every non-Flat region (a desk streams smoothed)
};
inline const char* scheme_word(swaps::curve::Scheme s) {
  switch (s) {
    case swaps::curve::Scheme::Hermite: return "hermite";
    case swaps::curve::Scheme::MonotoneCubic: return "monocubic";
    case swaps::curve::Scheme::MonotoneConvex: return "monoconvex";
    default: throw std::invalid_argument("g4_desk: the back scheme must be Hermite, MonotoneCubic or MonotoneConvex");
  }
}
inline std::string name_of(const Opts& o) {
  std::string n = std::string("g4_") + scheme_word(o.back);
  if (o.vd_everywhere) n += "_all";
  if (!o.turns) n += "_noturns";
  return n;
}
inline b::Date date_of_serial(long s) { return b::Date{std::chrono::sys_days{std::chrono::days{s}}}; }
// The central-bank meeting dates of `ccy` strictly after `from` and strictly before `before` (the DB's schedule).
inline std::vector<b::Date> meetings(const char* ccy, const b::Date& from, const b::Date& before) {
  std::vector<b::Date> out;
  for (long s : swaps::conventions::require_cb_meetings(ccy)) {
    const b::Date d = date_of_serial(s);
    if (d.serial() > from.serial() && d.serial() < before.serial()) out.push_back(d);
  }
  if (out.empty()) throw std::runtime_error(std::string("g4_desk: no ") + ccy + " meetings between the value date and the 1Y pillar");
  return out;
}
// THE COLD SEED CARRIES A SLOPE (measured 2026-10-15). A CCP discount curve is pinned by par swaps whose forecast leg
// is the RFR curve: when the forecast forwards are exactly constant every float coupon is f·tau_i, so the par rate is f
// whatever the discounting -- the CCP block's Jacobian columns vanish at a flat seed (to the second order of the
// accrual, ~1e-6 against 1e-4..3e-3 at x_true). From the ladder's
// flat seed the joint LM wandered 211 iterations into 16000% spreads (rank deficiency 13 at its answer), and the staged
// solve did no better; from a 2 bp/knot slope on the RFR curves it converges in 7 iterations to x_true (8e-11). A real
// cold seed (yesterday's curve, a rough term structure) always has one. At x_true the CCP columns are 1e-4..3e-3 against
// 2.4 for the RFR columns: the worst-conditioned block in the desk -- a 20 bp discount spread moves a par rate ~0.5 bp,
// the market's CME/LCH, Eurex/LCH, JSCC/LCH fact.
inline void slope_the_rfr_seed(Shape& s) {
  const auto& p = s.prob;
  for (int c = 0; c < p.n_curves(); ++c)
    if (p.curves[c].base < 0)
      for (int i = 0; i < p.curves[c].n_interp_knots(); ++i) s.x0[p.offset(c) + i] = 0.030 + 0.0002 * i;
}
}  // namespace g4

// The desk (see the header comment). Every curve's rows are contiguous in role order; a turned RFR curve adds its
// bracket observation and pin right after its par swaps.
inline Shape g4_desk(const g4::Opts& o = {}) {
  namespace cv = swaps::curve;
  cal::BundleProblem p;
  p.curves.resize(g4::NROLES);
  const b::Date vd = detail::vd();
  const b::Date turn_s = b::Date::from_iso("2026-12-31"), turn_e = b::Date::from_iso("2027-01-04");
  const b::Date obs_s = b::Date::from_iso("2026-12-15"), obs_e = b::Date::from_iso("2027-01-15");
  const cv::Scheme other = o.vd_everywhere ? o.back : cv::Scheme::Hermite;  // every non-RFR curve's scheme
  const auto one_region = [&](const std::vector<double>& knots, int base) {
    px::CurveStructure s;
    s.base = base;
    s.regions = {cv::CurveModule{knots, other}};
    return s;
  };
  const auto P = detail::pillars();
  const std::vector<std::string> ccp_tenors(P.begin() + 1, P.end());  // 2Y..30Y: the CCP basis is quoted from 2Y

  // ---- the four RFR curves: meeting-flat front + smooth back, the turn, and their CCP discount basis ----
  for (const g4::Rfr& r : g4::rfrs()) {
    const b::SwapConv conv = b::swap_conv(r.ccy_code, r.index);
    const std::string dc = b::index_day_count(r.index), cl = b::index_calendar(r.index);
    const b::Date spot = b::spot_date(vd, conv.calendar, conv.spot_lag);
    const b::Date one_y = b::resolve("1Y", vd, conv.calendar, conv.bdc, conv.spot_lag);
    // front: one compounded observation per meeting interval, [spot, m1], [m1, m2], ... -> knot k = meeting k
    std::vector<double> front;
    b::Date prev = spot;
    for (const b::Date& m : g4::meetings(r.ccy_code, spot, one_y)) {
      p.instruments.push_back(b::rate_instrument(r.role, b::observation(vd, prev, m, "compounded", 0.0, dc, cl), 0.03));
      front.push_back(b::curve_time(vd, m));
      prev = m;
    }
    const std::vector<double> back = detail::add_par_swaps(p, conv, r.role, r.role);  // 1Y..30Y par OIS
    px::CurveStructure s;
    s.base = -1;
    s.currency = r.ccy;
    s.regions = {cv::CurveModule{front, cv::Scheme::Flat}, cv::CurveModule{back, o.back}};
    if (o.turns) {
      s.turns = {px::Turn{b::curve_time(vd, turn_s), b::curve_time(vd, turn_e)}};
      p.instruments.push_back(b::rate_instrument(r.role, b::observation(vd, obs_s, obs_e, "compounded", 0.0, dc, cl), 0.03));
      p.instruments.push_back(b::turn_jump(r.role, 0, 0.0));
    }
    p.curves[static_cast<std::size_t>(r.role)] = std::move(s);
    // the other clearing house: the same par OIS, forecast on the RFR, discounted on the CCP curve (a spread over it)
    const std::vector<double> kc = detail::add_par_swaps(p, conv, r.role, r.ccp, ccp_tenors);
    px::CurveStructure sc = one_region(kc, r.role);
    sc.currency = r.ccy;
    p.curves[static_cast<std::size_t>(r.ccp)] = std::move(sc);
  }
  // ---- USD tenor basis: Fed funds over SOFR (averaged futures front, FF/SOFR basis back) ----
  {
    std::vector<double> k = detail::add_averaged_futures(p, "USD-FEDFUNDS", g4::USD_FF, 6);
    const auto kb = detail::add_basis_swaps(p, b::swap_conv("USD", "USD-FEDFUNDS"), g4::USD_FF, g4::USD_SOFR, g4::USD_SOFR, ccp_tenors);
    k.insert(k.end(), kb.begin(), kb.end());
    px::CurveStructure s = one_region(k, g4::USD_SOFR);
    s.currency = g4::USD;
    p.curves[g4::USD_FF] = std::move(s);
  }
  // ---- EUR tenor basis: 6M IRS pins EUR6M over ESTR; the 3s6s basis pins EUR3M over EUR6M ----
  {
    const auto k6 = detail::add_par_swaps(p, b::swap_conv("EUR", "EUR-EURIBOR-6M"), g4::EUR_6M, g4::EUR_ESTR);
    px::CurveStructure s6 = one_region(k6, g4::EUR_ESTR);
    s6.currency = g4::EUR;
    p.curves[g4::EUR_6M] = std::move(s6);
    const b::BasisConv bc = b::basis_conv("EUR-3S6S-BASIS");
    std::vector<double> k3;
    for (const auto& t : P) {
      const b::Date mat = b::resolve(t, vd, bc.spread.calendar, bc.spread.bdc, bc.spread.spot_lag);
      p.instruments.push_back(b::basis_swap(vd, bc, mat, g4::EUR_3M, g4::EUR_6M, g4::EUR_ESTR, 0.0));
      k3.push_back(p.instruments.back().fixed.coupons.back().pay);
    }
    px::CurveStructure s3 = one_region(k3, g4::EUR_6M);
    s3.currency = g4::EUR;
    p.curves[g4::EUR_3M] = std::move(s3);
  }
  // ---- the three collateral curves: FX forwards then MtM xccy basis, all funded on SOFR ----
  for (const g4::Xccy& x : g4::xccys()) {
    const b::XccyConv xc = b::xccy_conv(x.pair);
    std::vector<double> k;
    for (const char* t : {"1W", "1M", "3M", "6M", "1Y"}) {
      const double T = b::curve_time(vd, b::resolve(t, vd, xc.calendar, xc.bdc, xc.spot_lag));
      p.instruments.push_back(b::fx_forward(x.role, g4::USD_SOFR, x.usd_per_foreign, T, x.usd_per_foreign));
      k.push_back(T);
    }
    for (const char* t : {"2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"}) {
      const b::Date mat = b::resolve(t, vd, xc.calendar, xc.bdc, xc.spot_lag);
      p.instruments.push_back(b::xccy_mtm_basis(vd, xc, mat, x.role, x.foreign, g4::USD_SOFR, x.usd_per_foreign, 0.0));
      k.push_back(p.instruments.back().fixed.coupons.back().pay);
    }
    px::CurveStructure s = one_region(k, -1);
    s.currency = g4::USD;  // a foreign-in-USD collateral curve discounts USD cashflows
    p.curves[static_cast<std::size_t>(x.role)] = std::move(s);
  }
  if (o.smoothing) cal::stamp_region_smoothing(p, 1.0);  // the switch; Flat regions take no rows
  const bool linear = o.back == cv::Scheme::Hermite;
  return detail::make(g4::name_of(o),
                      std::string("G4 desk: 14 curves -- 4 RFR (meeting-flat front + ") + g4::scheme_word(o.back) +
                          " back" + (o.turns ? ", year-end turn" : "") + "), FF + EUR 3M/6M tenor basis, 4 CCP discount "
                          "curves, 3 xccy collateral curves" + (o.vd_everywhere ? "; every curve on the back scheme" : ""),
                      std::move(p), linear, &g4::slope_the_rfr_seed);
}

}  // namespace swaps::shapes
