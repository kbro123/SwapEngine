// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs)
// Gate for the book AS ROWS of the one compiled engine (calibration::BookRows, review item 1/4, 2026-10-06; until
// then the W-cache twin portfolio::CompiledMultiCurveBook with its own partials, predicates and fallback).
// QuantLib-FREE: a small 3-curve bundle is hand-built (mirroring portfolio_reprice_test.cpp) -- an
// outright domestic discount curve, a SPREAD forecast curve on it (exercising W_all's spread-base
// ancestry rows), and a foreign outright curve carrying one TURN (exercising the δ overlap column) --
// then a mixed ~20-position book is repriced off arbitrary knot states.
//
// The correctness anchor is PARITY: BookRows::npv(x) must equal the templated
// MultiCurveBook::value<double> over build_bundle_curves at the SAME x to 1e-10 (relative), at several
// distinct states AND after moving x back and forth -- the W-cache must track x, never serve stale DFs.
// The book mixes:
//   * multi-curve swaps with fwd != disc, incl. positions where fwd != disc != fixed curve;
//   * a float-only leg (empty fixed coupons), a spread-carrying leg, an FX-scaled leg, payers/receivers;
//   * one Kind::Xccy position (one row: bench leg + MtM leg + the two exchanges on the fixed leg);
//   * one COMPOUNDED-observation swap (the router sends it to the AAD block: Instrument::noncacheable) and one
//     MOMENT-path swap (compiles).
// Plus: position_instrument == MultiCurveBook::position_value on every shape, and the engine's directional
// derivative (the book's PV01; on a calibration bundle, J·dir on every quote kind) == jacobian·dir.

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <random>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/book_rows.hpp"
#include "swaps/portfolio/portfolio.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace pf = swaps::portfolio;

namespace {

// 3-curve bundle topology (shared knots): 1 front meeting + 5 back knots => 6 interp knots per curve.
const std::vector<double> kMeeting{0.5};
const std::vector<double> kBack{1.0, 2.0, 3.0, 5.0, 10.0};
constexpr int kNk = 6;
const std::vector<double> kSwapT{1.0, 2.0, 3.0, 5.0, 10.0};

// One OIS-style float coupon over [a,b]: a single telescoped sub-period (tau_pay == tau_index).
px::FloatCoupon ois_coupon(double a, double b) {
  px::FloatCoupon c;
  c.obs.sub_start = {a};
  c.obs.sub_end = {b};
  c.obs.tau_index = b - a;
  c.pay = b;
  c.tau_pay = b - a;
  return c;
}
px::FixedCoupon fixed_coupon(double a, double b) {
  px::FixedCoupon x;
  x.pay = b;
  x.tau = b - a;
  return x;
}
std::vector<px::FloatCoupon> annual_float(double T) {
  std::vector<px::FloatCoupon> v;
  double prev = 0.0;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) { v.push_back(ois_coupon(prev, t)); prev = t; }
  return v;
}
std::vector<px::FixedCoupon> annual_fixed(double T) {
  std::vector<px::FixedCoupon> v;
  double prev = 0.0;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) { v.push_back(fixed_coupon(prev, t)); prev = t; }
  return v;
}

// curve 0 = domestic discount (outright); curve 1 = forecast SPREAD on curve 0 (W ancestry rows);
// curve 2 = foreign discount (outright, ccy 1) with one TURN (a δ overlap column in W).
std::vector<px::CurveStructure> make_curves() {
  std::vector<px::CurveStructure> v;
  v.push_back({.base = -1, .regions = swaps::curve::flat_hermite(kMeeting, kBack)});
  v.push_back({.base = 0, .regions = swaps::curve::flat_hermite(kMeeting, kBack)});
  px::CurveStructure fx{.base = -1, .currency = 1, .regions = swaps::curve::flat_hermite(kMeeting, kBack)};
  fx.turns.push_back({0.90, 0.91});  // a ~4-day turn window inside year 1
  v.push_back(fx);
  return v;
}
// Stacked state: 6 + 6 + (6 interp + 1 turn δ) = 19 knots.
constexpr int kNx = 3 * kNk + 1;

Eigen::VectorXd base_state() {
  Eigen::VectorXd x(kNx);
  for (int i = 0; i < kNk; ++i) {
    x[i] = 0.030 + 0.0010 * i;            // curve 0 ~3%
    x[kNk + i] = 0.0030 + 0.0002 * i;     // curve 1 = 0 + ~30bp spread
    x[2 * kNk + i] = 0.020 + 0.0008 * i;  // curve 2 ~2% (foreign)
  }
  x[3 * kNk] = 0.0025;  // curve 2 turn jump δ
  return x;
}

// A multi-curve vanilla swap (payer-of-fixed): float forecasts `fc`, discounts `dc`; fixed leg on `xc`.
pf::MultiCurveBook::Position swap_position(double T, int fc, int dc, int xc, double rate, double notional) {
  pf::MultiCurveBook::Position p;
  p.kind = pf::MultiCurveBook::Kind::Swap;
  p.notional = notional;
  p.float_coupons = annual_float(T);
  p.fwd_curve = fc;
  p.disc_curve = dc;
  p.fixed_coupons = annual_fixed(T);
  p.fixed_curve = xc;
  p.fixed_rate = rate;
  return p;
}

// The mixed ~20-position book. Returns the number of positions the router must send to the AAD tier.
pf::MultiCurveBook mixed_book(int* n_fallback = nullptr) {
  pf::MultiCurveBook book;
  // 12 plain multi-curve swaps across tenors/roles, alternating payer/receiver.
  for (int i = 0; i < 12; ++i) {
    const double T = kSwapT[i % kSwapT.size()];
    const int fc = i % 3, dc = (i % 2 == 0) ? 0 : 2;
    book.positions.push_back(
        swap_position(T, fc, dc, dc, 0.02 + 0.001 * (i % 7), (i % 2 ? 1.0 : -1.0) * 1e6 * (1 + i)));
  }
  // fwd != disc != fixed -- all three roles on DIFFERENT curves.
  book.positions.push_back(swap_position(5.0, /*fc=*/1, /*dc=*/0, /*xc=*/2, 0.031, 2e6));
  book.positions.push_back(swap_position(10.0, /*fc=*/2, /*dc=*/1, /*xc=*/0, 0.024, -3e6));
  // Float-only leg (empty fixed coupons -> the templated float-only branch).
  {
    auto p = swap_position(3.0, 1, 0, 0, 0.0, 4e6);
    p.fixed_coupons.clear();
    book.positions.push_back(p);
  }
  // Spread-carrying float leg (konst path) + FX-scaled legs (k != 1 path).
  {
    auto p = swap_position(5.0, 1, 0, 0, 0.028, 1.5e6);
    for (auto& c : p.float_coupons) c.spread = 0.0030;
    book.positions.push_back(p);
    auto q = swap_position(3.0, 2, 2, 2, 0.021, 2.5e6);
    for (auto& c : q.float_coupons) c.scale = 1.10;   // foreign leg converted at FX spot
    for (auto& c : q.fixed_coupons) c.scale = 1.10;
    book.positions.push_back(q);
  }
  // A partially-REALIZED first coupon (constant contribution, still W-cacheable).
  {
    auto p = swap_position(2.0, 0, 0, 0, 0.029, -2e6);
    p.float_coupons[0].obs.realized = 0.004;
    book.positions.push_back(p);
  }
  // ---- Xccy (COMPILED since 2026-09-10: two rows, parity-checked below) ----
  // Xccy: receive a foreign resetting funding leg (curve 2, 50bp basis), pay a domestic leg (curve 0).
  {
    pf::MultiCurveBook::Position xp;
    xp.kind = pf::MultiCurveBook::Kind::Xccy;
    xp.notional = 3e6;
    xp.float_coupons = annual_float(5.0);
    xp.fwd_curve = 0;
    xp.disc_curve = 0;
    xp.mtm_coupons = annual_float(5.0);
    for (auto& c : xp.mtm_coupons) c.spread = 0.005;
    xp.mtm_fwd_curve = 2;
    xp.mtm_disc_curve = 2;
    xp.mtm_reset_num = 2;
    xp.mtm_reset_den = 0;
    xp.fx_spot = 1.10;
    book.positions.push_back(xp);
  }
  // COMPOUNDED (product-form) observation: must route to the fallback, never the arithmetic batch.
  {
    auto p = swap_position(2.0, 0, 0, 0, 0.030, 1e6);
    for (auto& c : p.float_coupons) {
      c.obs.compounded = true;  // single sub-period, but the PRODUCT form (1+r·dt) is demanded
      c.obs.realized_factor = 1.001;
    }
    book.positions.push_back(p);
  }
  // MOMENT-path (fixing_step > 0) averaged coupon: COMPILES since 2026-09-09 (BundleFloatBatch::set_state builds
  // the ½·step·xᵀQx correction from precomputed forward rows) -- it is parity-checked below, not a fallback.
  {
    auto p = swap_position(1.0, 2, 2, 2, 0.020, 1e6);
    p.float_coupons[0].obs.fixing_step = 1.0 / 252.0;
    book.positions.push_back(p);
  }
  if (n_fallback) *n_fallback = 1;  // the compounded position only
  return book;
}

// The templated reference: MultiCurveBook::value<double> over build_bundle_curves at x -- exactly what
// BundleSession::price_portfolio prices today.
double templated_npv(const std::vector<px::CurveStructure>& specs, const pf::MultiCurveBook& book,
                     const Eigen::VectorXd& x) {
  std::vector<int> off(specs.size(), 0);
  for (std::size_t c = 1; c < specs.size(); ++c) off[c] = off[c - 1] + specs[c - 1].n_knots();
  const auto C =
      cal::build_bundle_curves<double>(specs, [&](int c, int i) { return x[off[c] + i]; });
  const auto cof = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  return book.value<double>(cof);
}

void expect_parity(const cal::BookRows& cmb, const std::vector<px::CurveStructure>& specs,
                   const pf::MultiCurveBook& book, const Eigen::VectorXd& x, const char* label) {
  const double want = templated_npv(specs, book, x);
  const double got = cmb.npv(x);
  EXPECT_LE(std::abs(got - want), 1e-10 * (std::abs(want) + 1.0))
      << label << ": compiled npv " << got << " vs templated " << want;
}

}  // namespace

// The hybrid split lands exactly the compounded position on the fallback (Xccy and moment positions compile).
// The router lands exactly the compounded position on the AAD tier (Xccy and moment positions compile).
TEST(BookRows, SplitsCompiledAndFallbackPositions) {
  const auto specs = make_curves();
  int n_fb = 0;
  const pf::MultiCurveBook book = mixed_book(&n_fb);
  const cal::BookRows cmb(specs, book);
  EXPECT_EQ(cmb.n_positions(), static_cast<int>(book.positions.size()));
  EXPECT_EQ(cmb.n_fallback(), n_fb);
  EXPECT_EQ(cmb.n_compiled(), static_cast<int>(book.positions.size()) - n_fb);
  EXPECT_GT(cmb.n_times(), 0);
  EXPECT_EQ(cmb.n_knots(), kNx);
  EXPECT_EQ(cmb.n_rows(), cmb.n_positions()) << "one row per position (an xccy position is ONE row now)";
  for (int r = 0; r < cmb.n_rows(); ++r)
    EXPECT_EQ(cmb.engine().compiled_row(r) < 0, book.positions[static_cast<std::size_t>(r)].float_coupons[0].obs.compounded)
        << "row " << r << ": exactly the compounded position is off the W-cache";
}

// THE parity gate: compiled npv == templated value at several distinct states, to 1e-10 relative.
TEST(BookRows, NpvMatchesTemplatedKernelAtSeveralStates) {
  const auto specs = make_curves();
  const pf::MultiCurveBook book = mixed_book();
  const cal::BookRows cmb(specs, book);

  const Eigen::VectorXd x0 = base_state();
  expect_parity(cmb, specs, book, x0, "base state");

  Eigen::VectorXd x1 = x0;
  x1.array() += 10e-4;  // +10bp parallel (moves the turn δ too)
  expect_parity(cmb, specs, book, x1, "parallel +10bp");

  Eigen::VectorXd x2 = x0;
  for (int i = 0; i < x2.size(); ++i) x2[i] += 5e-4 * std::sin(0.9 * i + 0.4);  // an uneven twist
  expect_parity(cmb, specs, book, x2, "sin twist");

  Eigen::VectorXd x3 = x0;
  x3[2 * kNk + 2] -= 25e-4;  // a single foreign knot: only curve-2 (and the xccy fallback) moves
  expect_parity(cmb, specs, book, x3, "single foreign-knot bump");
}

// The W-cache must track x, not cache stale DFs: reprice at x0, then x1, then x0 again -- the third
// call must reproduce the FIRST bitwise (same kernel, same state), and every call holds parity.
TEST(BookRows, RepriceTracksXNotStaleDFs) {
  const auto specs = make_curves();
  const pf::MultiCurveBook book = mixed_book();
  const cal::BookRows cmb(specs, book);

  const Eigen::VectorXd x0 = base_state();
  Eigen::VectorXd x1 = x0;
  x1.array() += 20e-4;

  const double v0a = cmb.npv(x0);
  expect_parity(cmb, specs, book, x0, "x0 first pass");
  const double v1 = cmb.npv(x1);
  expect_parity(cmb, specs, book, x1, "x1 after x0");
  EXPECT_NE(v0a, v1) << "a 20bp move must change the book NPV";
  const double v0b = cmb.npv(x0);
  EXPECT_EQ(v0a, v0b) << "returning to the same x must reproduce the same NPV bitwise";
}

// The compiled xccy rows track the foreign curve: bumping ONLY the foreign curve moves the compiled book's
// total by exactly what the templated book moves (the xccy position + foreign-curve swaps).
TEST(BookRows, XccyRowsTrackTheForeignCurve) {
  const auto specs = make_curves();
  const pf::MultiCurveBook book = mixed_book();
  const cal::BookRows cmb(specs, book);

  const Eigen::VectorXd x0 = base_state();
  Eigen::VectorXd xb = x0;
  for (int i = 0; i < kNk; ++i) xb[2 * kNk + i] += 1e-3;  // +10bp on the foreign curve only

  const double d_compiled = cmb.npv(xb) - cmb.npv(x0);
  const double d_templated = templated_npv(specs, book, xb) - templated_npv(specs, book, x0);
  EXPECT_GT(std::abs(d_compiled), 1.0) << "the book holds foreign-curve risk; the bump must move it";
  EXPECT_LE(std::abs(d_compiled - d_templated), 1e-10 * (std::abs(d_templated) + 1.0));
}

// An all-compilable book (no fallback) and an all-fallback book are both valid degenerate splits.
TEST(BookRows, DegenerateSplitsPriceCorrectly) {
  const auto specs = make_curves();
  const Eigen::VectorXd x0 = base_state();

  pf::MultiCurveBook swaps_only;
  for (int i = 0; i < 6; ++i)
    swaps_only.positions.push_back(
        swap_position(kSwapT[i % kSwapT.size()], i % 3, 0, 0, 0.025 + 0.001 * i, 1e6 * (i + 1)));
  const cal::BookRows all_compiled(specs, swaps_only);
  EXPECT_EQ(all_compiled.n_fallback(), 0);
  expect_parity(all_compiled, specs, swaps_only, x0, "all-compiled book");

  pf::MultiCurveBook xccy_only;
  xccy_only.positions.push_back(mixed_book().positions[18]);  // the Xccy position: ONE compiled row
  ASSERT_EQ(xccy_only.positions[0].kind, pf::MultiCurveBook::Kind::Xccy);
  const cal::BookRows xccy_compiled(specs, xccy_only);
  EXPECT_EQ(xccy_compiled.n_fallback(), 0);
  EXPECT_EQ(xccy_compiled.n_compiled(), 1);
  EXPECT_EQ(xccy_compiled.n_rows(), 1);
  expect_parity(xccy_compiled, specs, xccy_only, x0, "xccy-only compiled book");
  pf::MultiCurveBook compounded_only;
  compounded_only.positions.push_back(mixed_book().positions[19]);  // the compounded position: all fallback
  const cal::BookRows all_fallback(specs, compounded_only);
  EXPECT_EQ(all_fallback.n_compiled(), 0);
  expect_parity(all_fallback, specs, compounded_only, x0, "all-fallback book");
  // A SEASONED xccy position (fixed FX reset on its first foreign coupon) rides the AAD tier.
  pf::MultiCurveBook seasoned;
  seasoned.positions.push_back(mixed_book().positions[18]);
  seasoned.positions[0].mtm_coupons.front().reset_fx = 1.05;
  const cal::BookRows seasoned_book(specs, seasoned);
  EXPECT_EQ(seasoned_book.n_fallback(), 1);
  expect_parity(seasoned_book, specs, seasoned, x0, "seasoned xccy on the fallback");
}

// =================================================================================================
// The book as ROWS (2026-10-06): the conversion, the directional derivative, per-position analytics.
// =================================================================================================
namespace {

// Every position SHAPE the twin used to route three ways: plain, roles on three curves, float-only, spread, FX-scaled,
// realized, stepped, principal (on the fixed leg's curve and on another), xccy, xccy with explicit flows, xccy with a
// spot time, seasoned xccy, compounded, moment.
std::vector<pf::MultiCurveBook::Position> every_shape() {
  std::vector<pf::MultiCurveBook::Position> v;
  for (const auto& p : mixed_book().positions) v.push_back(p);
  {
    auto p = swap_position(5.0, 1, 0, 0, 0.027, 1e7);
    for (std::size_t i = 0; i < p.fixed_coupons.size(); ++i) p.fixed_rates.push_back(0.020 + 0.003 * double(i));
    v.push_back(p);  // stepped
    auto q = swap_position(3.0, 1, 0, 0, 0.025, 5e6);
    q.principal_flows = {{0.0, 1.0}, {3.0, -1.0}};
    v.push_back(q);  // principal on the fixed leg's curve
    auto r = q;
    r.fixed_curve = 2;
    v.push_back(r);  // principal on ANOTHER curve than the fixed leg: rides the bench leg
    auto x = mixed_book().positions[18];
    x.principal_flows = {{0.0, -1.0}, {5.0, 1.0}};
    v.push_back(x);  // xccy with explicit flows
    auto y = mixed_book().positions[18];
    y.fx_spot_time = 2.0 / 365.0;
    v.push_back(y);  // xccy with a spot time (AAD tier)
  }
  return v;
}

template <class CurveOf>
double row_value(const cal::Instrument& ins, const CurveOf& C) { return cal::instrument_model_quote<double>(ins, C); }

}  // namespace

// position_instrument == MultiCurveBook::position_value, per unit notional, on every shape (the one conversion).
TEST(BookRows, PositionInstrumentMatchesTheTemplatedPositionValueOnEveryShape) {
  const auto specs = make_curves();
  const Eigen::VectorXd x = base_state();
  std::vector<int> off(specs.size(), 0);
  for (std::size_t c = 1; c < specs.size(); ++c) off[c] = off[c - 1] + specs[c - 1].n_knots();
  const auto C = cal::build_bundle_curves<double>(specs, [&](int c, int i) { return x[off[c] + i]; });
  const auto cof = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
  int k = 0;
  for (const auto& p : every_shape()) {
    const cal::Instrument ins = cal::position_instrument(p);
    cal::validate_instrument(ins, "shape");
    const double want = pf::MultiCurveBook::position_value<double>(p, cof) / p.notional;
    const double got = row_value(ins, cof);
    EXPECT_LE(std::abs(got - want), 1e-13 * (std::abs(want) + 1.0)) << "shape " << k << ": row " << got << " vs position " << want;
    ++k;
  }
  EXPECT_GE(k, 24);
}

// The whole book of shapes prices as rows to the templated value, and the ROUTING is the instrument's own answer.
TEST(BookRows, EveryShapeBookPricesAsRowsAndRoutesByTheInstrumentsOwnAnswer) {
  const auto specs = make_curves();
  pf::MultiCurveBook book;
  book.positions = every_shape();
  const cal::BookRows rows(specs, book);
  const Eigen::VectorXd x = base_state();
  expect_parity(rows, specs, book, x, "every shape");
  for (int r = 0; r < rows.n_rows(); ++r)
    EXPECT_EQ(rows.engine().compiled_row(r) < 0, cal::position_instrument(book.positions[static_cast<std::size_t>(r)]).noncacheable()) << "row " << r;
  EXPECT_EQ(rows.n_fallback(), 2) << "the compounded swap, the spot-time xccy and nothing else";
  // Per-position NPVs sum to the book, and the directional (per-position PV01) sums to pv01.
  EXPECT_NEAR(rows.npvs(x).sum(), rows.npv(x), 1e-6 * std::abs(rows.npv(x)));
  EXPECT_NEAR(rows.pv01s(x).sum(), rows.pv01(x), 1e-9 * std::abs(rows.pv01(x)) + 1e-12);
}

// A hand-built calibration bundle carrying EVERY quote kind, source and row map: par (banded), spread, future (Rate),
// zero-coupon, standalone FX (log map), a Portfolio of an FX forward and a par swap, an MtM basis, a turn pin (banded),
// an Npv row -- plus one compounded par row so the AAD block holds a row too.
namespace {
cal::BundleProblem every_kind_bundle() {
  cal::BundleProblem p;
  p.curves = make_curves();  // curve 2: ccy 1, one turn
  auto par = [&](double T, int fc, int dc, double market) {
    cal::Instrument i;
    i.quote = cal::QuoteKind::ParRate;
    i.fwd = {annual_float(T), fc, dc};
    i.fixed = {annual_fixed(T), dc};
    i.market = market;
    return i;
  };
  cal::Instrument banded = par(2.0, 0, 0, 0.031);
  banded.band_lower = 0.0305; banded.band_upper = 0.0315; banded.band_decay = 0.2;
  p.instruments.push_back(banded);
  p.instruments.push_back(par(5.0, 1, 0, 0.034));
  {
    cal::Instrument s;
    s.quote = cal::QuoteKind::ParSpread;
    s.fwd = {annual_float(3.0), 1, 0};
    s.bench = {annual_float(3.0), 0, 0};
    s.fixed = {annual_fixed(3.0), 0};
    s.market = 0.0028;
    p.instruments.push_back(s);
  }
  {
    cal::Instrument f;
    f.quote = cal::QuoteKind::Rate;
    f.obs.sub_start = {0.75};
    f.obs.sub_end = {1.0};
    f.obs.tau_index = 0.25;
    f.forecast = 0;
    f.convexity = 0.0001;
    f.market = 0.032;
    p.instruments.push_back(f);
  }
  {
    cal::Instrument z;
    z.quote = cal::QuoteKind::ZeroCouponRate;
    z.fwd = {annual_float(2.0), 1, 0};
    z.fixed = {{fixed_coupon(0.0, 2.0)}, 0};
    z.market = 0.033;
    p.instruments.push_back(z);
  }
  auto fxf = [&](double T) {
    cal::Instrument f;
    f.quote = cal::QuoteKind::FxForward;
    f.fx_num = 2; f.fx_den = 0; f.fx_spot = 1.10; f.fx_time = T;
    f.market = 1.10 * std::exp(-0.01 * T);
    return f;
  };
  p.instruments.push_back(fxf(1.0));
  {
    cal::Instrument pf_;
    pf_.quote = cal::QuoteKind::Portfolio;
    pf_.combination = {{1.0, fxf(2.0)}, {-0.5, par(3.0, 2, 2, 0.021)}};
    pf_.market = 1.0;
    p.instruments.push_back(pf_);
  }
  {
    cal::Instrument m;
    m.quote = cal::QuoteKind::XccyMtmBasis;
    m.fwd = {annual_float(3.0), 2, 2};
    m.bench = {annual_float(3.0), 1, 2};
    m.fixed = {annual_fixed(3.0), 2};
    m.mtm = {annual_float(3.0), 0, 0};
    m.mtm.reset_num = 2; m.mtm.reset_den = 0; m.mtm.fx_spot = 1.10;
    m.market = 0.002;
    p.instruments.push_back(m);
  }
  {
    cal::Instrument t;
    t.quote = cal::QuoteKind::TurnJump;
    t.turn_curve = 2; t.turn_index = 0;
    t.market = 0.002; t.band_lower = 0.001; t.band_upper = 0.004; t.band_decay = 0.1;
    p.instruments.push_back(t);
  }
  p.instruments.push_back(cal::book_problem(p.curves, pf::MultiCurveBook{{mixed_book().positions[18]}}).instruments[0]);
  {
    cal::Instrument c = par(2.0, 0, 0, 0.030);
    for (auto& cp : c.fwd.coupons) { cp.obs.compounded = true; cp.obs.realized_factor = 1.0005; }
    p.instruments.push_back(c);  // AAD block
  }
  return p;
}
}  // namespace

// directional_into == jacobian·dir on every quote kind, every source and both row maps, on both tiers, for several
// directions -- the parallel direction, a unit vector on the turn delta, and two random ones.
TEST(BookRows, DirectionalDerivativeEqualsTheJacobianTimesTheDirectionOnEveryQuoteKindAndTier) {
  const cal::BundleProblem p = every_kind_bundle();
  const cal::HybridBundleResidual eng(p);
  ASSERT_GT(eng.n_compiled_rows(), 0);
  ASSERT_LT(eng.n_compiled_rows(), p.n_residuals()) << "the compounded row must sit on the AAD block";
  const Eigen::VectorXd x = base_state();
  const Eigen::MatrixXd J = eng.jacobian(x);
  std::vector<Eigen::VectorXd> dirs{px::parallel_direction(p.curves), Eigen::VectorXd::Unit(kNx, kNx - 1)};
  std::mt19937_64 g(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  for (int k = 0; k < 2; ++k) { Eigen::VectorXd d(kNx); for (int i = 0; i < kNx; ++i) d[i] = u(g); dirs.push_back(d); }
  Eigen::VectorXd out;
  for (std::size_t k = 0; k < dirs.size(); ++k) {
    eng.directional_into(x, dirs[k], out);
    const Eigen::VectorXd want = J * dirs[k];
    ASSERT_EQ(out.size(), want.size());
    for (int r = 0; r < out.size(); ++r)
      EXPECT_LE(std::abs(out[r] - want[r]), 1e-12 * (std::abs(want[r]) + 1.0)) << "dir " << k << " row " << r << " (" << (eng.compiled_row(r) >= 0 ? "compiled" : "aad") << ")";
  }
  // and after a quote/band move (the slope of a mapped row changes): still J·dir at the new market.
  cal::HybridBundleResidual moved(p);
  moved.set_quote(0, 0.0320, 0.0305, 0.0315, 0.2);  // the banded par row's target walks outside its band
  const Eigen::MatrixXd J2 = moved.jacobian(x);
  moved.directional_into(x, dirs[2], out);
  const Eigen::VectorXd want2 = J2 * dirs[2];
  for (int r = 0; r < out.size(); ++r) EXPECT_LE(std::abs(out[r] - want2[r]), 1e-12 * (std::abs(want2[r]) + 1.0)) << "moved row " << r;
}

// Per-position PV01 and key-rate risk are the engine's directional and Jacobian: each equals a central difference of
// that position's NPV (compiled and AAD rows alike).
TEST(BookRows, PerPositionPv01AndKeyRateRiskMatchFiniteDifferences) {
  const auto specs = make_curves();
  pf::MultiCurveBook book;
  book.positions = every_shape();
  const cal::BookRows rows(specs, book);
  const Eigen::VectorXd x = base_state();
  const Eigen::VectorXd u = px::parallel_direction(specs);
  const double h = 1e-6;
  const Eigen::VectorXd pv01 = rows.pv01s(x);
  const Eigen::VectorXd up = rows.npvs(x + h * u), dn = rows.npvs(x - h * u);
  for (int r = 0; r < rows.n_rows(); ++r) {
    const double fd = 1e-4 * (up[r] - dn[r]) / (2.0 * h);
    EXPECT_LE(std::abs(pv01[r] - fd), 1e-6 * (std::abs(fd) + 1.0)) << "position " << r;
  }
  const Eigen::MatrixXd K = rows.key_rate(x);
  ASSERT_EQ(K.rows(), rows.n_rows());
  ASSERT_EQ(K.cols(), kNx);
  const Eigen::VectorXd via_K = 1e-4 * (K * u);  // the directional IS the Jacobian along u, on both tiers
  for (int r = 0; r < rows.n_rows(); ++r)
    EXPECT_LE(std::abs(pv01[r] - via_K[r]), 1e-10 * (std::abs(via_K[r]) + 1.0)) << "position " << r << " directional vs K·u";
  for (int j : {0, 4, kNk + 2, 2 * kNk + 1, kNx - 1}) {
    Eigen::VectorXd xp = x, xm = x;
    xp[j] += h; xm[j] -= h;
    const Eigen::VectorXd vp = rows.npvs(xp);  // npvs returns a ref into the engine's scratch: copy before the next call
    const Eigen::VectorXd fd = (vp - rows.npvs(xm)) / (2.0 * h);
    for (int r = 0; r < rows.n_rows(); ++r)
      EXPECT_LE(std::abs(K(r, j) - fd[r]), 1e-6 * (std::abs(fd[r]) + 1.0)) << "position " << r << " knot " << j;
  }
}

// An empty book is zero everywhere and builds without an engine error.
TEST(BookRows, EmptyBookIsZero) {
  const cal::BookRows rows(make_curves(), pf::MultiCurveBook{});
  EXPECT_EQ(rows.n_rows(), 0);
  EXPECT_EQ(rows.npv(base_state()), 0.0);
  EXPECT_EQ(rows.pv01(base_state()), 0.0);
}
