#pragma once
// THE SHAPE LADDER — every instrument shape the compiled / hybrid hot path accepts, as named bundles built from
// the conventions DB with REAL dates (value date 2026-07-08), simplest to most complex. One fixture library, two
// consumers: tests/hotpath_shapes_test.cpp (T3 cross-path parity + T4 allocation invariants PER SHAPE) and
// bench/shape_ladder_bench.cpp (per-shape perf-gate metrics). A kernel change is measured on every rung, so it
// cannot be right on annual OIS and wrong (or slow) on averaged futures, bands, turns, portfolios, ZC or xccy.
// QuantLib-free: markets are the model quotes at a known x_true (a stationary point), never hand-typed.
#include <cmath>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "swaps/build/calendar.hpp"
#include "swaps/build/conventions.hpp"
#include "swaps/build/instruments.hpp"
#include "swaps/build/observations.hpp"
#include "swaps/build/schedule.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/pricing/curve_spec.hpp"

namespace swaps::shapes {

namespace b = swaps::build;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;

struct Shape {
  std::string name, note;
  cal::BundleProblem prob;
  Eigen::VectorXd x_true;   // the state the markets were generated at (residual == 0 there)
  Eigen::VectorXd x0;       // a flat cold seed
  Eigen::VectorXd q0;       // the markets (prob.market())
  Eigen::VectorXd q_small;  // a ~0.1 bp tick (FX rows: relative)
  Eigen::VectorXd q_cross;  // banded shapes: a 1.5 bp tick that crosses the band edge on odd rows; else == q_small
  Eigen::VectorXd q_big;    // a ~25 bp move (forces a Jacobian refresh)
  // banded shapes: the odd banded rows' market alternating 0.05 bp ABOVE / BELOW their upper band edge (the
  // active-set stress: the model quote sits on a kink every tick); else both == q0. has_bands says whether
  // the pair means anything (the EdgeOscTick benchmark / metric exists only for banded rungs).
  Eigen::VectorXd q_edge_hi, q_edge_lo;
  bool has_bands = false;
  bool expect_compiled = true;  // every row is W-cacheable => the tick must be allocation-free (T4)
  // Can this bundle ride the frozen-Newton STREAMER (BundleSession::start_streaming)? False only for
  // mixed_scheme, and NOT because the maths forbids it: BundleSession::needs_recalibrate() is still the
  // whole-bundle veto `has_nonlinear_` -- the same over-approximation the ROUTER carried until 2026-09-10,
  // one layer up. The hybrid engine underneath already partitions a mixed bundle (front rows on the W-cache,
  // long rows on the AAD block, refreshed on staleness) exactly as it does for FX/MtM, which streams. This
  // rung is what makes that gap visible; closing it is a router-style change to the session, with its own
  // parity tests, not something to slip in with a fixture. Until then the streaming tests and the tick
  // metrics skip this rung and its Jacobian/parity coverage still runs.
  bool streams = true;
};

namespace detail {

inline const b::Date& vd() { static const b::Date d = b::Date::from_iso("2026-07-08"); return d; }
inline std::vector<std::string> pillars() { return {"1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y"}; }
inline px::CurveStructure spec(const std::vector<double>& knots, int base = -1) {
  px::CurveStructure s;
  s.base = base;
  s.regions = swaps::curve::flat_hermite({}, knots);
  return s;
}
// The SAME knots split into a LINEAR-MAP front (Hermite, W-cacheable) and a VALUE-DEPENDENT back
// (MonotoneCubic, Hyman-filtered). Every other rung is linear end to end, so until 2026-09-10 nothing in
// the ladder exercised the router's per-row partition at all: a scheme is not an instrument shape, and the
// ladder only varied instruments. `split` is the first knot that belongs to the non-linear region, so the
// curve's LINEAR HORIZON is knots[split - 1] and a row is compiled iff every time it reads sits at or below it.
inline px::CurveStructure spec_mixed(const std::vector<double>& knots, std::size_t split, int base = -1) {
  px::CurveStructure s;
  s.base = base;
  const std::vector<double> front(knots.begin(), knots.begin() + static_cast<std::ptrdiff_t>(split));
  const std::vector<double> back(knots.begin() + static_cast<std::ptrdiff_t>(split), knots.end());
  s.regions = {swaps::curve::CurveModule{front, swaps::curve::Scheme::Hermite},
               swaps::curve::CurveModule{back, swaps::curve::Scheme::MonotoneCubic}};
  return s;
}
// Par swaps of `conv` at the pillar tenors, forecasting fc / discounting disc; returns the knot times (the last
// fixed pay time of each instrument -- one knot per pillar => a square block).
inline std::vector<double> add_par_swaps(cal::BundleProblem& p, const b::SwapConv& conv, int fc, int disc,
                                         const std::vector<std::string>& tenors = pillars(), double q = 0.03) {
  std::vector<double> knots;
  for (const auto& t : tenors) {
    const b::Date mat = b::resolve(t, vd(), conv.calendar, conv.bdc, conv.spot_lag);
    p.instruments.push_back(b::par_swap(vd(), conv, mat, fc, disc, q));
    knots.push_back(p.instruments.back().fixed.coupons.back().pay);
  }
  return knots;
}
inline std::vector<double> add_basis_swaps(cal::BundleProblem& p, const b::SwapConv& conv, int fc, int bench, int disc,
                                           const std::vector<std::string>& tenors) {
  std::vector<double> knots;
  for (const auto& t : tenors) {
    const b::Date mat = b::resolve(t, vd(), conv.calendar, conv.bdc, conv.spot_lag);
    p.instruments.push_back(b::basis_swap(vd(), conv, mat, fc, bench, disc, 0.0));
    knots.push_back(p.instruments.back().fixed.coupons.back().pay);
  }
  return knots;
}
// Monthly AVERAGED futures-style Rate rows on `fc` (daily weighted sub-periods -- the shape that leaves the plain
// fast path). Returns the window end times (the knots).
inline std::vector<double> add_averaged_futures(cal::BundleProblem& p, const std::string& index, int fc, int n_months) {
  std::vector<double> knots;
  const std::string dc = b::index_day_count(index), cl = b::index_calendar(index);
  for (int k = 0; k < n_months; ++k) {
    const b::Date s = k == 0 ? b::spot_date(vd(), cl, 0) : b::resolve(std::to_string(k) + "M", vd(), cl, "Following", 0);
    const b::Date e = b::resolve(std::to_string(k + 1) + "M", vd(), cl, "Following", 0);
    p.instruments.push_back(b::rate_instrument(fc, b::observation(vd(), s, e, "averaged", 0.0, dc, cl), 0.03));
    knots.push_back(b::curve_time(vd(), e));
  }
  return knots;
}
// Par swaps whose FLOAT coupons are daily ARITHMETIC AVERAGES of the index (Fed funds OIS in reality; the DB row
// still says "compounded" and float_leg() ignores leg compounding — E3 ledger addendum). This is the shape E3-A3
// found leaving the plain fast path INSIDE a leg (pv()'s general branch), distinct from averaged futures.
inline std::vector<double> add_averaged_leg_swaps(cal::BundleProblem& p, const b::SwapConv& conv, const std::string& index,
                                                  int fc, int disc, const std::vector<std::string>& tenors) {
  std::vector<double> knots;
  const std::string cl = b::index_calendar(index);
  for (const auto& t : tenors) {
    const b::Date mat = b::resolve(t, vd(), conv.calendar, conv.bdc, conv.spot_lag);
    cal::Instrument in = b::par_swap(vd(), conv, mat, fc, disc, 0.03);
    const auto per = b::swap_periods_to(vd(), conv.calendar, mat, conv.float_freq_tok, conv.bdc, conv.spot_lag);
    for (std::size_t i = 0; i < in.fwd.coupons.size() && i < per.size(); ++i)
      in.fwd.coupons[i].obs = b::observation(vd(), per[i].first, per[i].second, "averaged", 0.0, conv.float_dc, cl);
    p.instruments.push_back(std::move(in));
    knots.push_back(p.instruments.back().fixed.coupons.back().pay);
  }
  return knots;
}
inline std::vector<double> add_moment_leg_swaps(cal::BundleProblem& p, const b::SwapConv& conv, const std::string& index,
                                                int fc, int disc, const std::vector<std::string>& tenors) {
  std::vector<double> knots;
  const std::string cl = b::index_calendar(index);
  for (const auto& t : tenors) {
    const b::Date mat = b::resolve(t, vd(), conv.calendar, conv.bdc, conv.spot_lag);
    cal::Instrument in = b::par_swap(vd(), conv, mat, fc, disc, 0.03);
    const auto per = b::swap_periods_to(vd(), conv.calendar, mat, conv.float_freq_tok, conv.bdc, conv.spot_lag);
    for (std::size_t i = 0; i < in.fwd.coupons.size() && i < per.size(); ++i)
      in.fwd.coupons[i].obs = b::moment_observation(vd(), per[i].first, per[i].second, conv.float_dc, cl);
    p.instruments.push_back(std::move(in));
    knots.push_back(p.instruments.back().fixed.coupons.back().pay);
  }
  return knots;
}
inline void fill_x(Shape& s) {
  const auto& p = s.prob;
  s.x_true.resize(p.n_knots()); s.x0.resize(p.n_knots());
  for (int c = 0; c < p.n_curves(); ++c) {
    const int off = p.offset(c), ni = p.curves[c].n_interp_knots(), nk = p.curves[c].n_knots();
    for (int i = 0; i < ni; ++i) {
      s.x_true[off + i] = p.curves[c].base < 0 ? 0.030 + 0.0004 * i : 0.0020 + 0.00005 * i;
      s.x0[off + i] = p.curves[c].base < 0 ? 0.030 : 0.0020;
    }
    for (int i = ni; i < nk; ++i) { s.x_true[off + i] = 0.0020; s.x0[off + i] = 0.0; }  // turn jumps
  }
}
// Markets := model quotes at x_true. FX rows take the outright (their residual is a log form); every other row
// takes market += residual(market=0-ish), which is exact for the linear-in-market residual kinds.
inline void set_markets(Shape& s) {
  auto& p = s.prob;
  const auto C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return s.x_true[p.offset(c) + i]; });
  const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  for (auto& ins : p.instruments)
    if (ins.quote == cal::QuoteKind::FxForward) ins.market = cal::instrument_model_quote<double>(ins, curve_of);
  const Eigen::VectorXd r = p.residuals<double>(s.x_true);
  for (int i = 0; i < p.n_residuals(); ++i)
    if (p.instruments[i].quote != cal::QuoteKind::FxForward) p.instruments[i].market += r[i];
}
inline void finish(Shape& s) {
  auto& p = s.prob;
  const int m = p.n_residuals();
  s.q0.resize(m); s.q_small.resize(m); s.q_cross.resize(m); s.q_big.resize(m);
  s.q_edge_hi.resize(m); s.q_edge_lo.resize(m);
  s.has_bands = false;
  for (int i = 0; i < m; ++i) {
    const auto& ins = p.instruments[i];
    const bool fx = ins.quote == cal::QuoteKind::FxForward;
    const double bump = std::sin(0.7 * i + 0.3);
    s.q0[i] = ins.market;
    s.q_small[i] = fx ? ins.market * (1.0 + 1e-5 * bump) : ins.market + 1e-5 * bump;
    // A ~25 bp PARALLEL rate move leaves an FX forward where it is (covered interest parity: both curves move
    // together); scaling the forwards by 0.25 % on top contradicted the rate rows and drove the xccy basis
    // curve to an absurd state whose next tick diverged (found 2026-09-10 -- the gate had been timing a
    // FAILING fx_xccy refresh tick at 17 us). Basis rows move a tenth of the rate move.
    s.q_big[i] = fx ? ins.market : (ins.quote == cal::QuoteKind::XccyMtmBasis ? ins.market + 2.5e-4 * bump : ins.market + 25e-4 * bump);
    const bool banded = ins.band_upper > ins.band_lower;
    s.q_cross[i] = (banded && (i % 2)) ? ins.market + 1.5e-4 : s.q_small[i];
    s.q_edge_hi[i] = (banded && (i % 2)) ? ins.band_upper + 0.05e-4 : ins.market;
    s.q_edge_lo[i] = (banded && (i % 2)) ? ins.band_upper - 0.05e-4 : ins.market;
    if (banded) s.has_bands = true;
  }
}
inline void band_all(cal::BundleProblem& p, double half_width, double decay) {
  for (auto& ins : p.instruments) {
    if (ins.quote == cal::QuoteKind::FxForward) continue;
    ins.band_lower = ins.market - half_width; ins.band_upper = ins.market + half_width; ins.band_decay = decay;
  }
}
inline cal::Instrument butterfly(const cal::BundleProblem& p, int lo, int belly, int hi) {
  cal::Instrument f;
  f.quote = cal::QuoteKind::Portfolio;
  f.combination = {{-1.0, p.instruments[lo]}, {2.0, p.instruments[belly]}, {-1.0, p.instruments[hi]}};
  for (auto& c : f.combination) c.instrument.market = 0.0;
  return f;
}
inline Shape make(std::string name, std::string note, cal::BundleProblem prob, bool compiled = true,
                  void (*after_markets)(Shape&) = nullptr, bool streams = true) {
  Shape s; s.name = std::move(name); s.note = std::move(note); s.prob = std::move(prob); s.expect_compiled = compiled;
  s.streams = streams;
  fill_x(s); set_markets(s);
  if (after_markets) after_markets(s);
  finish(s);
  return s;
}

}  // namespace detail

// ---- the rungs -----------------------------------------------------------------------------------------------
inline Shape ois_nolag() {  // GBP SONIA: self-discounted, NO pay lag (the telescoping shape)
  cal::BundleProblem p; const auto conv = b::swap_conv("GBP", "GBP-SONIA");
  p.curves = {detail::spec(detail::add_par_swaps(p, conv, 0, 0))};
  return detail::make("ois_nolag", "GBP SONIA OIS, 1 curve, pay lag 0", std::move(p));
}
inline Shape ois_lag() {  // USD SOFR: self-discounted, 2-day pay lag
  cal::BundleProblem p; const auto conv = b::swap_conv("USD", "USD-SOFR");
  p.curves = {detail::spec(detail::add_par_swaps(p, conv, 0, 0))};
  return detail::make("ois_lag", "USD SOFR OIS, 1 curve, pay lag 2", std::move(p));
}
inline Shape ibor_multicurve() {  // EUR: ESTR discount curve + 6M EURIBOR forecast curve as a spread over it
  cal::BundleProblem p;
  const auto k0 = detail::add_par_swaps(p, b::swap_conv("EUR", "EUR-ESTR"), 0, 0);
  const auto k1 = detail::add_par_swaps(p, b::swap_conv("EUR", "EUR-EURIBOR-6M"), 1, 0);
  p.curves = {detail::spec(k0), detail::spec(k1, 0)};
  return detail::make("ibor_multicurve", "EUR ESTR OIS + 6M EURIBOR IRS on ESTR discounting (spread curve)", std::move(p));
}
inline Shape basis_spread() {  // USD SOFR + Fed funds as a ParSpread basis chain
  cal::BundleProblem p;
  const auto k0 = detail::add_par_swaps(p, b::swap_conv("USD", "USD-SOFR"), 0, 0);
  const auto k1 = detail::add_basis_swaps(p, b::swap_conv("USD", "USD-FEDFUNDS"), 1, 0, 0, detail::pillars());
  p.curves = {detail::spec(k0), detail::spec(k1, 0)};
  return detail::make("basis_spread", "USD SOFR OIS + FF/SOFR basis (ParSpread rows, spread curve)", std::move(p));
}
inline Shape averaged() {  // + monthly AVERAGED Fed funds futures (daily weighted sub-periods) on the FF curve
  cal::BundleProblem p;
  const auto k0 = detail::add_par_swaps(p, b::swap_conv("USD", "USD-SOFR"), 0, 0);
  auto k1 = detail::add_averaged_futures(p, "USD-FEDFUNDS", 1, 6);
  const auto P = detail::pillars();
  std::vector<std::string> longer(P.begin() + 1, P.end());  // 2Y..30Y basis
  const auto k1b = detail::add_basis_swaps(p, b::swap_conv("USD", "USD-FEDFUNDS"), 1, 0, 0, longer);
  k1.insert(k1.end(), k1b.begin(), k1b.end());
  p.curves = {detail::spec(k0), detail::spec(k1, 0)};
  return detail::make("averaged", "basis_spread + 6 monthly averaged FF futures (weighted sub-periods)", std::move(p));
}
inline Shape averaged_leg() {  // SOFR + FF OIS whose float legs are daily arithmetic averages (weighted sub-periods in a LEG)
  cal::BundleProblem p;
  const auto k0 = detail::add_par_swaps(p, b::swap_conv("USD", "USD-SOFR"), 0, 0);
  const auto k1 = detail::add_averaged_leg_swaps(p, b::swap_conv("USD", "USD-FEDFUNDS"), "USD-FEDFUNDS", 1, 0, detail::pillars());
  p.curves = {detail::spec(k0), detail::spec(k1, 0)};
  return detail::make("averaged_leg", "USD SOFR OIS + FF OIS with daily-averaged float coupons (ParRate rows, weighted sub-periods)", std::move(p));
}
inline Shape averaged_leg_moment() {  // the same FF OIS legs on the MOMENT path (one bracket + xᵀQx per coupon)
  cal::BundleProblem p;
  const auto k0 = detail::add_par_swaps(p, b::swap_conv("USD", "USD-SOFR"), 0, 0);
  const auto k1 = detail::add_moment_leg_swaps(p, b::swap_conv("USD", "USD-FEDFUNDS"), "USD-FEDFUNDS", 1, 0, detail::pillars());
  p.curves = {detail::spec(k0), detail::spec(k1, 0)};
  return detail::make("averaged_leg_moment", "USD SOFR OIS + FF OIS with daily-averaged float coupons on the MOMENT path", std::move(p));
}
// EVERY LINEAR-MAP SCHEME IN ONE CURVE (2026-09-10). The ladder's promise is that a kernel change is measured
// on every shape it accepts, but until now "shape" meant only the INSTRUMENT: every rung interpolated with
// flat_hermite, so Flat, Linear, NaturalCubic, BSpline and Tension appeared nowhere in it. This rung walks one
// SOFR curve through all six W-cacheable schemes in region order -- which also exercises the claim that regions
// compose in ANY order -- and mixed_scheme() below adds the seventh, MonotoneCubic. Together the two cover
// curve::Scheme exhaustively, and ShapeLadder.CoversEveryQuoteKindAndScheme fails if a new scheme is added
// without a rung.
inline Shape all_schemes() {
  cal::BundleProblem p;
  // 18 pillars, not the usual 12: a cubic B-spline region needs >= 3 knots, so every region gets 3.
  const std::vector<std::string> tenors = {"1Y",  "2Y",  "3Y",  "4Y",  "5Y",  "6Y",  "7Y",  "8Y",  "9Y",
                                           "10Y", "11Y", "12Y", "15Y", "20Y", "25Y", "30Y", "35Y", "40Y"};
  const auto k = detail::add_par_swaps(p, b::swap_conv("USD", "USD-SOFR"), 0, 0, tenors);
  const swaps::curve::Scheme order[] = {swaps::curve::Scheme::Flat,        swaps::curve::Scheme::Linear,
                                        swaps::curve::Scheme::NaturalCubic, swaps::curve::Scheme::Hermite,
                                        swaps::curve::Scheme::BSpline,     swaps::curve::Scheme::Tension};
  px::CurveStructure s0;
  s0.base = -1;
  const std::size_t per = k.size() / 6;  // 18 knots -> 3 per region (the B-spline minimum)
  for (std::size_t r = 0; r < 6; ++r) {
    const std::size_t lo = r * per, hi = (r == 5) ? k.size() : (r + 1) * per;
    s0.regions.push_back(swaps::curve::CurveModule{
        std::vector<double>(k.begin() + static_cast<std::ptrdiff_t>(lo), k.begin() + static_cast<std::ptrdiff_t>(hi)),
        order[r]});
  }
  p.curves = {s0};
  return detail::make("all_schemes",
                      "USD SOFR OIS over six regions, one per LINEAR-MAP scheme: Flat, Linear, NaturalCubic, "
                      "Hermite, BSpline, Tension (all W-cacheable)",
                      std::move(p));
}

// THE MIXED-SCHEME RUNG (2026-09-10). The `averaged` bundle -- SOFR OIS + FF averaged futures and basis on a
// spread curve -- with the SOFR curve's long end interpolated by MonotoneCubic instead of Hermite. This is the
// desk shape the router was rewritten for: FF futures and short swaps sit under the linear horizon and keep the
// W-cache, while the long swaps that read into the value-dependent region go to the AAD block. Before the
// rewrite ONE such region sent every row in the bundle to AAD; the ladder could not see that, because every
// rung was linear. expect_compiled is false: with a partition the tick is not allocation-free.
inline Shape mixed_scheme() {
  cal::BundleProblem p;
  const auto k0 = detail::add_par_swaps(p, b::swap_conv("USD", "USD-SOFR"), 0, 0);
  auto k1 = detail::add_averaged_futures(p, "USD-FEDFUNDS", 1, 6);
  const auto P = detail::pillars();
  const std::vector<std::string> longer(P.begin() + 1, P.end());  // 2Y..30Y basis
  const auto k1b = detail::add_basis_swaps(p, b::swap_conv("USD", "USD-FEDFUNDS"), 1, 0, 0, longer);
  k1.insert(k1.end(), k1b.begin(), k1b.end());
  // pillars() is 1Y..30Y; split at index 6 => Hermite through 10Y, MonotoneCubic over 12Y..30Y.
  p.curves = {detail::spec_mixed(k0, 6), detail::spec(k1, 0)};
  return detail::make("mixed_scheme",
                      "averaged + the SOFR long end on MonotoneCubic: the router's per-row partition "
                      "(front rows W-cached, long rows on the AAD block)",
                      std::move(p), false, nullptr, /*streams=*/false);
}
inline Shape banded() {  // SOFR OIS with a +-1 bp Huber band on every row; q_cross crosses the upper edge on odd rows
  cal::BundleProblem p; const auto conv = b::swap_conv("USD", "USD-SOFR");
  p.curves = {detail::spec(detail::add_par_swaps(p, conv, 0, 0))};
  return detail::make("banded", "USD SOFR OIS, +-1bp bands decay 0.5 on every row", std::move(p), true,
                      [](Shape& s) { detail::band_all(s.prob, 1e-4, 0.5); });
}
inline Shape turns() {  // SOFR OIS + a year-end TURN (banded pin) + a bracketing compounded future
  cal::BundleProblem p; const auto conv = b::swap_conv("USD", "USD-SOFR");
  auto s0 = detail::spec(detail::add_par_swaps(p, conv, 0, 0));
  s0.turns = {px::Turn{b::curve_time(detail::vd(), b::Date::from_iso("2026-12-31")), b::curve_time(detail::vd(), b::Date::from_iso("2027-01-04"))}};
  p.curves = {s0};
  const std::string dc = b::index_day_count("USD-SOFR"), cl = b::index_calendar("USD-SOFR");
  p.instruments.push_back(b::rate_instrument(0, b::observation(detail::vd(), b::Date::from_iso("2026-12-15"), b::Date::from_iso("2027-01-15"), "compounded", 0.0, dc, cl), 0.03));
  p.instruments.push_back(b::turn_jump(0, 0, 0.0));
  return detail::make("turns", "USD SOFR OIS + year-end turn (banded pin) + bracketing future", std::move(p), true,
                      [](Shape& s) { auto& t = s.prob.instruments.back(); t.band_lower = t.market - 5e-4; t.band_upper = t.market + 5e-4; t.band_decay = 0.3; });
}
inline Shape portfolio() {  // SOFR OIS + two butterflies as Portfolio rows
  cal::BundleProblem p; const auto conv = b::swap_conv("USD", "USD-SOFR");
  p.curves = {detail::spec(detail::add_par_swaps(p, conv, 0, 0))};
  p.instruments.push_back(detail::butterfly(p, 1, 4, 6));    // 2y / 5y / 10y
  p.instruments.push_back(detail::butterfly(p, 4, 6, 11));   // 5y / 10y / 30y
  return detail::make("portfolio", "USD SOFR OIS + 2 butterflies (Portfolio rows)", std::move(p));
}
inline Shape zero_coupon() {  // BRL DI x Pre: ZeroCouponRate rows (the nonlinear quote transform)
  cal::BundleProblem p; const auto conv = b::swap_conv("BRL", "BRL-CDI");
  std::vector<std::string> ten{"1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y"};
  p.curves = {detail::spec(detail::add_par_swaps(p, conv, 0, 0, ten, 0.12))};
  return detail::make("zero_coupon", "BRL DI x Pre zero-coupon swaps (BUS/252, annually compounded quotes)", std::move(p));
}
inline Shape fx_xccy() {  // USD SOFR + EUR ESTR + EUR-in-USD collateral curve: FX forwards + MtM xccy basis
  cal::BundleProblem p;
  const auto k0 = detail::add_par_swaps(p, b::swap_conv("USD", "USD-SOFR"), 0, 0);
  const auto k1 = detail::add_par_swaps(p, b::swap_conv("EUR", "EUR-ESTR"), 1, 1);
  std::vector<double> k2;
  const auto xc = b::xccy_conv("EURUSD");
  for (const char* t : {"1W", "1M", "3M", "6M", "1Y"}) {
    const double T = b::curve_time(detail::vd(), b::resolve(t, detail::vd(), xc.calendar, xc.bdc, xc.spot_lag));
    p.instruments.push_back(b::fx_forward(2, 0, 1.08, T, 1.08));
    k2.push_back(T);
  }
  for (const char* t : {"2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"}) {
    const b::Date mat = b::resolve(t, detail::vd(), xc.calendar, xc.bdc, xc.spot_lag);
    p.instruments.push_back(b::xccy_mtm_basis(detail::vd(), xc, mat, 2, 1, 0, 1.08, 0.0));
    k2.push_back(p.instruments.back().fixed.coupons.back().pay);
  }
  p.curves = {detail::spec(k0), detail::spec(k1), detail::spec(k2)};
  return detail::make("fx_xccy", "USD SOFR + EUR ESTR + EUR-in-USD: 5 FX forwards + 8 MtM xccy basis rows", std::move(p), true);  // compiled since 2026-09-09
}
// everything at once: 5 curves, every quote kind, bands, a turn, butterflies. `mixed` puts the SOFR curve's
// long end on MonotoneCubic, which is the ONLY difference between the desk and desk_mixed rungs -- so the pair
// isolates what the router's per-row partition costs and buys on the full-coverage bundle, with nothing else
// varying between them.
inline Shape desk_impl(bool mixed) {
  cal::BundleProblem p;
  const auto k_sofr = detail::add_par_swaps(p, b::swap_conv("USD", "USD-SOFR"), 0, 0);
  auto s0 = mixed ? detail::spec_mixed(k_sofr, 6) : detail::spec(k_sofr);
  const int n_sofr = p.n_residuals();
  s0.turns = {px::Turn{b::curve_time(detail::vd(), b::Date::from_iso("2026-12-31")), b::curve_time(detail::vd(), b::Date::from_iso("2027-01-04"))}};
  auto k1 = detail::add_averaged_futures(p, "USD-FEDFUNDS", 1, 6);
  const auto P = detail::pillars();
  std::vector<std::string> longer(P.begin() + 1, P.end());
  const auto k1b = detail::add_basis_swaps(p, b::swap_conv("USD", "USD-FEDFUNDS"), 1, 0, 0, longer);
  k1.insert(k1.end(), k1b.begin(), k1b.end());
  const auto k2 = detail::add_par_swaps(p, b::swap_conv("EUR", "EUR-ESTR"), 2, 2);
  std::vector<double> k3;
  const auto xc = b::xccy_conv("EURUSD");
  for (const char* t : {"1W", "1M", "3M", "6M", "1Y"}) {
    const double T = b::curve_time(detail::vd(), b::resolve(t, detail::vd(), xc.calendar, xc.bdc, xc.spot_lag));
    p.instruments.push_back(b::fx_forward(3, 0, 1.08, T, 1.08)); k3.push_back(T);
  }
  for (const char* t : {"2Y", "3Y", "5Y", "7Y", "10Y", "15Y", "20Y", "30Y"}) {
    const b::Date mat = b::resolve(t, detail::vd(), xc.calendar, xc.bdc, xc.spot_lag);
    p.instruments.push_back(b::xccy_mtm_basis(detail::vd(), xc, mat, 3, 2, 0, 1.08, 0.0)); k3.push_back(p.instruments.back().fixed.coupons.back().pay);
  }
  const auto k4 = detail::add_par_swaps(p, b::swap_conv("EUR", "EUR-EURIBOR-6M"), 4, 2);
  const std::string dc = b::index_day_count("USD-SOFR"), cl = b::index_calendar("USD-SOFR");
  p.instruments.push_back(b::rate_instrument(0, b::observation(detail::vd(), b::Date::from_iso("2026-12-15"), b::Date::from_iso("2027-01-15"), "compounded", 0.0, dc, cl), 0.03));
  p.instruments.push_back(b::turn_jump(0, 0, 0.0));
  p.instruments.push_back(detail::butterfly(p, 1, 4, 6));
  p.instruments.push_back(detail::butterfly(p, 4, 6, 11));
  p.curves = {s0, detail::spec(k1, 0), detail::spec(k2), detail::spec(k3), detail::spec(k4, 2)};
  return detail::make(mixed ? "desk_mixed" : "desk",
                      mixed ? "desk with the SOFR long end on MonotoneCubic: the full-coverage bundle across "
                              "the router's partition (front rows W-cached, long rows on the AAD block)"
                            : "5 curves: SOFR (banded, turn, butterflies) + FF basis & averaged futures + ESTR + EUR-in-USD (FX/xccy) + EURIBOR",
                      std::move(p), !mixed,
                      [](Shape& s) {
                        auto& ins = s.prob.instruments;
                        for (int i = 0; i < 12; ++i) { ins[i].band_lower = ins[i].market - 1e-4; ins[i].band_upper = ins[i].market + 1e-4; ins[i].band_decay = 0.5; }
                        auto& t = ins[ins.size() - 3]; t.band_lower = t.market - 5e-4; t.band_upper = t.market + 5e-4; t.band_decay = 0.3;
                      },
                      /*streams=*/!mixed);
  (void)n_sofr;
}
inline Shape desk() { return desk_impl(false); }
inline Shape desk_mixed() { return desk_impl(true); }

inline std::vector<Shape> ladder() {
  return {ois_nolag(), ois_lag(), ibor_multicurve(), basis_spread(), averaged(), averaged_leg(), averaged_leg_moment(), all_schemes(), mixed_scheme(), banded(), turns(), portfolio(), zero_coupon(), fx_xccy(), desk(), desk_mixed()};
}

}  // namespace swaps::shapes
