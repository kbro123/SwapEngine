// Gate for the COMPILED multi-curve book reprice kernel (portfolio::CompiledMultiCurveBook, audit U2).
// QuantLib-FREE: a small 3-curve bundle is hand-built (mirroring portfolio_reprice_test.cpp) -- an
// outright domestic discount curve, a SPREAD forecast curve on it (exercising W_all's spread-base
// ancestry rows), and a foreign outright curve carrying one TURN (exercising the δ overlap column) --
// then a mixed ~20-position book is repriced off arbitrary knot states.
//
// The correctness anchor is PARITY: CompiledMultiCurveBook::npv(x) must equal the templated
// MultiCurveBook::value<double> over build_bundle_curves at the SAME x to 1e-10 (relative), at several
// distinct states AND after moving x back and forth -- the W-cache must track x, never serve stale DFs.
// The book mixes:
//   * multi-curve swaps with fwd != disc, incl. positions where fwd != disc != fixed curve;
//   * a float-only leg (empty fixed coupons), a spread-carrying leg, an FX-scaled leg, payers/receivers;
//   * one Kind::Xccy position (NOT compiled -- priced through the templated fallback, hybrid split);
//   * one COMPOUNDED-observation swap and one MOMENT-path swap (both must route to the fallback: the
//     arithmetic W-cache batch cannot represent the product / the ∫f² correction).

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/portfolio/compiled_multi.hpp"
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

// The mixed ~20-position book. Returns the number of positions that must land on the FALLBACK path.
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
  // ---- fallback positions ----
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
  // MOMENT-path (fixing_step > 0) averaged coupon: the batch has no ∫f² correction -> fallback.
  {
    auto p = swap_position(1.0, 2, 2, 2, 0.020, 1e6);
    p.float_coupons[0].obs.fixing_step = 1.0 / 252.0;
    book.positions.push_back(p);
  }
  if (n_fallback) *n_fallback = 3;
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

void expect_parity(const pf::CompiledMultiCurveBook& cmb, const std::vector<px::CurveStructure>& specs,
                   const pf::MultiCurveBook& book, const Eigen::VectorXd& x, const char* label) {
  const double want = templated_npv(specs, book, x);
  const double got = cmb.npv(x);
  EXPECT_LE(std::abs(got - want), 1e-10 * (std::abs(want) + 1.0))
      << label << ": compiled npv " << got << " vs templated " << want;
}

}  // namespace

// The hybrid split lands exactly the Xccy + compounded + moment positions on the fallback.
TEST(CompiledMultiCurveBook, SplitsCompiledAndFallbackPositions) {
  const auto specs = make_curves();
  int n_fb = 0;
  const pf::MultiCurveBook book = mixed_book(&n_fb);
  const pf::CompiledMultiCurveBook cmb(specs, book);
  EXPECT_EQ(cmb.n_positions(), static_cast<int>(book.positions.size()));
  EXPECT_EQ(cmb.n_fallback(), n_fb);
  EXPECT_EQ(cmb.n_compiled(), static_cast<int>(book.positions.size()) - n_fb);
  EXPECT_GT(cmb.n_times(), 0);
  EXPECT_EQ(cmb.n_knots(), kNx);
}

// THE parity gate: compiled npv == templated value at several distinct states, to 1e-10 relative.
TEST(CompiledMultiCurveBook, NpvMatchesTemplatedKernelAtSeveralStates) {
  const auto specs = make_curves();
  const pf::MultiCurveBook book = mixed_book();
  const pf::CompiledMultiCurveBook cmb(specs, book);

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
TEST(CompiledMultiCurveBook, RepriceTracksXNotStaleDFs) {
  const auto specs = make_curves();
  const pf::MultiCurveBook book = mixed_book();
  const pf::CompiledMultiCurveBook cmb(specs, book);

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

// The xccy fallback genuinely rides along: bumping ONLY the foreign curve moves the compiled book's
// total by exactly what the templated book moves (the xccy position + foreign-curve swaps).
TEST(CompiledMultiCurveBook, XccyFallbackTracksTheForeignCurve) {
  const auto specs = make_curves();
  const pf::MultiCurveBook book = mixed_book();
  const pf::CompiledMultiCurveBook cmb(specs, book);

  const Eigen::VectorXd x0 = base_state();
  Eigen::VectorXd xb = x0;
  for (int i = 0; i < kNk; ++i) xb[2 * kNk + i] += 1e-3;  // +10bp on the foreign curve only

  const double d_compiled = cmb.npv(xb) - cmb.npv(x0);
  const double d_templated = templated_npv(specs, book, xb) - templated_npv(specs, book, x0);
  EXPECT_GT(std::abs(d_compiled), 1.0) << "the book holds foreign-curve risk; the bump must move it";
  EXPECT_LE(std::abs(d_compiled - d_templated), 1e-10 * (std::abs(d_templated) + 1.0));
}

// An all-compilable book (no fallback) and an all-fallback book are both valid degenerate splits.
TEST(CompiledMultiCurveBook, DegenerateSplitsPriceCorrectly) {
  const auto specs = make_curves();
  const Eigen::VectorXd x0 = base_state();

  pf::MultiCurveBook swaps_only;
  for (int i = 0; i < 6; ++i)
    swaps_only.positions.push_back(
        swap_position(kSwapT[i % kSwapT.size()], i % 3, 0, 0, 0.025 + 0.001 * i, 1e6 * (i + 1)));
  const pf::CompiledMultiCurveBook all_compiled(specs, swaps_only);
  EXPECT_EQ(all_compiled.n_fallback(), 0);
  expect_parity(all_compiled, specs, swaps_only, x0, "all-compiled book");

  pf::MultiCurveBook xccy_only;
  xccy_only.positions.push_back(mixed_book().positions[18]);  // the Xccy position
  ASSERT_EQ(xccy_only.positions[0].kind, pf::MultiCurveBook::Kind::Xccy);
  const pf::CompiledMultiCurveBook all_fallback(specs, xccy_only);
  EXPECT_EQ(all_fallback.n_compiled(), 0);
  expect_parity(all_fallback, specs, xccy_only, x0, "all-fallback book");
}
