// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD) | T3 cross-path parity (two engine paths, same inputs) | T4 hot-path invariant (allocation / determinism / structure)
// E5.2 KERNEL PINS (2026-09-10) -- the value pins and cross-path checks the 2026-09-08 kernel-test audit found
// missing (review-2026-09-08/kernel-tests.md, mutations M5 / M8 / fixes 3, 10, 11 and axis-3 M3):
//   T5  the Hermite (Bessel tangents), NaturalCubic and Linear interpolants pinned to HAND-COMPUTED values
//       (before this file a Hermite with its Bessel weights swapped passed all 33 tests that use Hermite);
//   T5  the tension series helpers sinhm1 / xcoshm pinned against long-double references (only coshm2 was);
//   T5  the B-spline interior breakpoints are the de Boor averages of the control-point sites (asserted
//       nowhere before);
//   T3  the MonotoneCubic forward-AAD derivative equals a central finite difference where the Hyman filter
//       CLAMPS (the AAD-tier scheme had no derivative check at the kink);
//   T4  structure_equal (the warm-vs-recompile gate since E4.A) sees every structural field the old hash
//       missed: the MtM funding leg, interior observation data, fx_spot/fx_time, pv_currency, reg_sigma,
//       the fixed leg's discount role, the bench leg's forecast role.
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

#include "tolerances.hpp"
#include "swaps/ad/dual.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/structure_fingerprint.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/regions.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace cv = swaps::curve;
namespace ad = swaps::ad;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace tol = swaps::tol;

// ---- T5: interpolant VALUES vs hand computation ------------------------------------------------------------
// Three knots {1, 2, 4} with forwards {0.030, 0.035, 0.032} (non-monotone, so the tangents matter). A LEADING
// region flat-extrapolates its first value before t=1. Hand computation (exact rational arithmetic, 2026-09-10):
//   h = {1, 2}, secants {0.005, -0.0015}
//   Hermite / Bessel (parabolic) tangents  m0 = ((2h0+h1)s0 - h0 s1)/(h0+h1) = 0.0215/3,
//                                         m1 = (h1 s0 + h0 s1)/(h0+h1)       = 0.0085/3,
//                                         m2 = ((2h1+h0)s1 - h1 s0)/(h1+h0)  = -0.0175/3
//     => f(1.5) = 0.0330416666..., f(3) = 0.0356666666..., ∫₀⁴ f = 0.13275
//   Natural cubic (M0 = M2 = 0, 2(h0+h1) M1 = 6 (s1 - s0) => M1 = -0.0065)
//     => f(1.5) = 0.03290625, f(3) = 0.035125, ∫₀⁴ f = 0.1319375
//   Linear => f(1.5) = 0.0325, f(3) = 0.0335, ∫₀⁴ f = 0.1295
namespace {
cv::ModularCurve<double> three_knot(cv::Scheme s) {
  auto c = cv::make_modular_curve<double>({cv::CurveModule{{1.0, 2.0, 4.0}, s}});
  c.set_forwards((Eigen::VectorXd(3) << 0.030, 0.035, 0.032).finished());
  return c;
}
}  // namespace

TEST(SchemeValues, HermiteBesselTangentsMatchTheHandComputation) {
  const auto c = three_knot(cv::Scheme::Hermite);
  EXPECT_NEAR(c.forward(0.5), 0.030, tol::literal);  // flat before the first knot
  EXPECT_NEAR(c.forward(1.5), 0.033041666666666664, tol::literal);
  EXPECT_NEAR(c.forward(3.0), 0.035666666666666666, tol::literal);
  EXPECT_NEAR(c.integral(4.0), 0.13275, tol::literal);
}

TEST(SchemeValues, NaturalCubicMatchesTheHandComputation) {
  const auto c = three_knot(cv::Scheme::NaturalCubic);
  EXPECT_NEAR(c.forward(1.5), 0.032906249999999998, tol::literal);
  EXPECT_NEAR(c.forward(3.0), 0.035125, tol::literal);
  EXPECT_NEAR(c.integral(4.0), 0.1319375, tol::literal);
}

TEST(SchemeValues, LinearMatchesTheHandComputation) {
  const auto c = three_knot(cv::Scheme::Linear);
  EXPECT_NEAR(c.forward(1.5), 0.0325, tol::literal);
  EXPECT_NEAR(c.forward(3.0), 0.0335, tol::literal);
  EXPECT_NEAR(c.integral(4.0), 0.1295, tol::literal);
}

// MonotoneCubic = the NATURAL-SPLINE tangents (the C2 tridiagonal in first-derivative form) passed through
// QuantLib's Hyman filter. Knots {1, 2, 4}, values {0.030, 0.0301, 0.010}: secants S = {0.0001, −0.01005};
// the tridiagonal  2m0 + m1 = 3S0,  h1 m0 + 2(h0+h1) m1 + h0 m2 = 3(h1 S0 + h0 S1),  m1 + 2m2 = 3S1  gives
// m = {0.00179167, −0.00328333, −0.01343333}. Hyman: m0 has S0's sign but exceeds 3|S0| = 0.0003 -> CLAMPED to
// 0.0003 (the end-clamp branch); m1 (interior, N = 3: M = 3·min(|S0|, |S1|, |pm|) = 0.0003, pm·m1 > 0) -> −0.0003;
// m2 within 3|S1| -> untouched. Segment 0 (h = 1): c = 3S0 − 2m0 − m1 = 0, d = m0 + m1 − 2S0 = −0.0002 =>
// f(1.5) = 0.030 + 0.00015 − 0.000025 = 0.030125 (the UNCLAMPED m0 would give 0.03031146); segment 1 (h = 2):
// c = 3S1/2 − (2m1 + m2)/2 = −0.00805833, d = (m1 + m2)/4 − 2S1/4 = 0.00159167 => f(3) = 0.0301 − 0.0003 −
// 0.00805833 + 0.00159167 = 0.02333333. Exact rational arithmetic, 2026-09-10. E5.4: the mutation harness
// showed the END clamp was pinned by nothing outside the QuantLib binary.
TEST(SchemeValues, MonotoneCubicHymanEndClampMatchesTheHandComputation) {
  auto c = cv::make_modular_curve<double>({cv::CurveModule{{1.0, 2.0, 4.0}, cv::Scheme::MonotoneCubic}});
  c.set_forwards((Eigen::VectorXd(3) << 0.030, 0.0301, 0.010).finished());
  EXPECT_NEAR(c.forward(1.5), 0.030125, tol::literal);
  EXPECT_NEAR(c.forward(3.0), 0.023333333333333334, tol::literal);
}

// ---- T4: the static "is this scheme linear" answer equals the built curve's runtime answer, per scheme ----------
// (E6.1c: curve::scheme_is_linear replaced three enum tests; the runtime truth is ModularCurve::is_linear_map.)
TEST(SchemeLinearity, StaticAnswerMatchesTheBuiltCurve) {
  for (cv::Scheme s : {cv::Scheme::Flat, cv::Scheme::Linear, cv::Scheme::NaturalCubic, cv::Scheme::Hermite,
                       cv::Scheme::MonotoneCubic, cv::Scheme::BSpline, cv::Scheme::Tension}) {
    cv::CurveModule m{{1.0, 2.0, 3.0, 5.0, 7.0, 10.0}, s};
    if (s == cv::Scheme::Tension) m.sigma = 1.0;
    const auto c = cv::make_modular_curve<double>({m});
    EXPECT_EQ(c.is_linear_map(), cv::scheme_is_linear(s)) << "scheme " << static_cast<int>(s);
  }
}

// ---- T5: the tension series helpers ------------------------------------------------------------------------
// sinhm1(x) = Σ_{k>=1} x^(2k+1)/(2k+1)!, xcoshm(x) = Σ_{k>=1} (2k) x^(2k+1)/(2k+1)!, both evaluated by a
// truncated series for |x| < 0.5. The references are long-double sums to x²⁵. The series stop at x¹¹, so at
// the 0.5 branch point the truncation error is x¹⁰·6/13! ≈ 9e-13 (sinhm1) and x¹⁰·36/13! ≈ 6e-12 (xcoshm)
// RELATIVE -- that is the helpers' documented accuracy, pinned here (coshm2 carries terms to x¹⁴: 3e-16).
TEST(TensionSeries, SinhM1AndXCoshMAreAccurateOnTheSeriesBranch) {
  for (double x = 0.02; x < 0.5; x += 0.02) {
    const long double xl = x, x2 = xl * xl;
    long double ref_s = 0.0L, ref_x = 0.0L, pw = x2 * xl, fact = 6.0L;  // x³, 3!
    for (int k = 1; k <= 12; ++k) {
      ref_s += pw / fact;
      ref_x += (2.0L * k) * pw / fact;
      pw *= x2;
      fact *= static_cast<long double>(2 * k + 2) * static_cast<long double>(2 * k + 3);
    }
    EXPECT_NEAR(cv::tension_detail::sinhm1(x) / static_cast<double>(ref_s), 1.0, 2e-12) << "x=" << x;
    EXPECT_NEAR(cv::tension_detail::xcoshm(x) / static_cast<double>(ref_x), 1.0, 1e-11) << "x=" << x;
    // and the direct branch agrees with the series branch where both are accurate
    EXPECT_NEAR(std::sinh(x) - x, static_cast<double>(ref_s), 1e-17 + 1e-12 * static_cast<double>(ref_s));
  }
}

// ---- T5: B-spline interior breakpoints are the de Boor averages of the control-point SITES ------------------
// flat_bspline({0.25}, {1,2,3,5,7,10}): the B-spline region follows the flat region, so its sites are
// u = {0.25 (the join), 1, 2, 3, 5, 7, 10} and the n-3 = 3 interior breakpoints are mean(u[j], u[j+1], u[j+2])
// for j = 1..3: {2, 10/3, 5}. A uniform grid over [0.25, 10] would put them at {2.6875, 5.125, 7.5625}.
TEST(BSplineKnots, InteriorBreakpointsAreTheDeBoorAveragesOfTheSites) {
  auto c = cv::make_modular_curve<double>(cv::flat_bspline({0.25}, {1, 2, 3, 5, 7, 10}));
  c.set_forwards(Eigen::VectorXd::Constant(7, 0.03));
  const std::vector<double> pieces = c.pieces();
  const auto has = [&](double t) {
    return std::any_of(pieces.begin(), pieces.end(), [&](double p) { return std::abs(p - t) < 1e-12; });
  };
  EXPECT_TRUE(has(2.0));
  EXPECT_TRUE(has(10.0 / 3.0));
  EXPECT_TRUE(has(5.0));
  EXPECT_FALSE(has(2.6875)) << "uniform breakpoints are the old (rejected) placement";
  EXPECT_FALSE(has(5.125));
  EXPECT_FALSE(has(7.5625));
}

// ---- T3: MonotoneCubic AAD derivative vs finite difference where the Hyman filter clamps -------------------
TEST(MonotoneCubicAad, DerivativeMatchesCentralDifferenceThroughTheHymanClamp) {
  const std::vector<double> meeting{0.25}, back{1, 2, 3, 5, 7, 10};
  // Wiggly data: the Hyman filter clamps several tangents (the natural tangents overshoot the secants).
  Eigen::VectorXd x(7);
  x << 0.030, 0.052, 0.021, 0.061, 0.028, 0.041, 0.046;
  auto cd = cv::make_modular_curve<ad::Dual>(cv::flat_monotone(meeting, back));
  cd.set_forwards(ad::seed(x));
  auto c = cv::make_modular_curve<double>(cv::flat_monotone(meeting, back));
  double worst = 0.0;
  for (double t : {0.5, 1.3, 1.5, 1.9, 2.4, 2.75, 3.6, 4.9, 6.2, 8.5, 9.8}) {
    const ad::Dual f = cd.forward(t);
    for (int k = 0; k < 7; ++k) {
      const double h = 1e-6;
      Eigen::VectorXd xp = x, xm = x;
      xp[k] += h; xm[k] -= h;
      c.set_forwards(xp); const double fp = c.forward(t);
      c.set_forwards(xm); const double fm = c.forward(t);
      const double fd = (fp - fm) / (2 * h);
      const double an = f.derivatives().size() ? f.derivatives()[k] : 0.0;
      worst = std::max(worst, std::abs(fd - an));
      EXPECT_NEAR(an, fd, 1e-7) << "t=" << t << " knot " << k;
    }
  }
  std::cout << "  [monotone] max |AAD - FD| over 77 (t, knot) pairs = " << worst << "\n";
}

// ---- T4: structure_equal sees every structural field ------------------------------------------------------
namespace {
px::FloatCoupon ois_coupon(double s, double e) {
  px::FloatCoupon c;
  c.obs.sub_start = {s, (s + e) / 2};
  c.obs.sub_end = {(s + e) / 2, e};
  c.obs.weight = {1.0, 1.0};
  c.obs.tau_index = e - s;
  c.pay = e;
  c.tau_pay = e - s;
  return c;
}
cal::FloatLeg float_leg(double T, int fc, int disc) {
  cal::FloatLeg leg;
  leg.forecast = fc;
  leg.discount = disc;
  double prev = 0.0;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) { leg.coupons.push_back(ois_coupon(prev, t)); prev = t; }
  return leg;
}
cal::FixedLeg fixed_leg(double T, int disc) {
  cal::FixedLeg leg;
  leg.discount = disc;
  for (double t = 1.0; t <= T + 1e-9; t += 1.0) { px::FixedCoupon fc; fc.pay = t; fc.tau = 1.0; leg.coupons.push_back(fc); }
  return leg;
}
cal::BundleProblem two_curve_problem() {
  cal::BundleProblem p;
  cal::BundleCurveSpec a, b;
  a.base = -1; a.currency = 0; a.regions = cv::flat_hermite({0.5}, {2.0, 5.0, 10.0});
  b.base = 0;  b.currency = 1; b.regions = cv::flat_hermite({0.5}, {2.0, 5.0, 10.0});
  p.curves = {a, b};
  cal::Instrument par; par.quote = cal::QuoteKind::ParRate; par.fwd = float_leg(5.0, 0, 0); par.fixed = fixed_leg(5.0, 0); par.market = 0.03;
  cal::Instrument basis; basis.quote = cal::QuoteKind::ParSpread; basis.fwd = float_leg(5.0, 1, 0); basis.bench = float_leg(5.0, 0, 0); basis.fixed = fixed_leg(5.0, 0);
  cal::Instrument mtm; mtm.quote = cal::QuoteKind::XccyMtmBasis; mtm.fwd = float_leg(5.0, 1, 1); mtm.bench = float_leg(5.0, 0, 0); mtm.mtm = float_leg(5.0, 0, 0);
  mtm.mtm.reset_num = 0; mtm.mtm.reset_den = 1; mtm.mtm.fx_spot = 1.1; mtm.fixed = fixed_leg(5.0, 1); mtm.pv_currency = 1;
  p.instruments = {par, basis, mtm};
  return p;
}
}  // namespace

TEST(StructureEqual, EveryStructuralFieldIsCompared) {
  const cal::BundleProblem b = two_curve_problem();
  ASSERT_TRUE(cal::structure_equal(b, b));
  { auto p = b; p.instruments[0].market += 1e-4;                       EXPECT_TRUE(cal::structure_equal(p, b)) << "a re-quote is not structural"; }
  { auto p = b; p.instruments[0].band_lower = 0.02; p.instruments[0].band_upper = 0.04; EXPECT_TRUE(cal::structure_equal(p, b)) << "a band is not structural"; }
  const auto differs = [&](cal::BundleProblem p, const char* what) {
    EXPECT_FALSE(cal::structure_equal(p, b)) << "structural edit invisible to structure_equal: " << what;
  };
  // The fields the 2026-09-08 audit found the old hash did NOT see (fp_probe.cpp):
  differs([&] { auto p = b; p.instruments[2].mtm.reset_num = 1; return p; }(), "mtm reset role");
  differs([&] { auto p = b; p.instruments[2].mtm.fx_spot = 1.3; return p; }(), "mtm fx_spot");
  differs([&] { auto p = b; p.instruments[2].mtm.coupons[1].pay += 0.01; return p; }(), "mtm coupon pay");
  differs([&] { auto p = b; p.instruments[2].mtm.coupons.pop_back(); return p; }(), "mtm coupon dropped");
  differs([&] { auto p = b; p.instruments[2].mtm.forecast = 1; return p; }(), "mtm forecast role");
  differs([&] { auto p = b; p.instruments[0].fwd.coupons[1].obs.weight[1] = 0.7; return p; }(), "an INTERIOR observation weight");
  differs([&] { auto p = b; p.instruments[0].fwd.coupons[1].obs.sub_start[1] += 0.01; return p; }(), "an interior observation start");
  differs([&] { auto p = b; p.instruments[0].fwd.coupons[1].obs.sub_end[0] += 0.01; return p; }(), "an interior observation end");
  differs([&] { auto p = b; p.instruments[1].fwd.fx_spot = 1.2; return p; }(), "a leg fx_spot");
  differs([&] { auto p = b; p.instruments[2].pv_currency = 0; return p; }(), "pv_currency");
  differs([&] { auto p = b; p.instruments[2].fx_time = 0.5; return p; }(), "fx_time");
  differs([&] { auto p = b; p.instruments[0].fixed.discount = 1; return p; }(), "the fixed leg's discount role");
  differs([&] { auto p = b; p.instruments[1].bench.forecast = 1; return p; }(), "the bench leg's forecast role");
  differs([&] { auto p = b; p.curves[1].regions[1].reg_sigma = 0.7; return p; }(), "a region's regulariser sigma");
  differs([&] { auto p = b; p.curves[1].regions[1].reg_lambda = 0.7; return p; }(), "a region's regulariser lambda");
  // and the ones the old enumeration already had
  differs([&] { auto p = b; p.curves[0].regions[1].scheme = cv::Scheme::Linear; return p; }(), "region scheme");
  differs([&] { auto p = b; p.curves[0].regions[1].knots[0] += 0.25; return p; }(), "a knot time");
  differs([&] { auto p = b; p.curves[1].base = -1; return p; }(), "curve base role");
  differs([&] { auto p = b; p.curves[1].currency = 0; return p; }(), "curve currency");
  differs([&] { auto p = b; p.instruments[1].quote = cal::QuoteKind::ParRate; return p; }(), "quote kind");
  differs([&] { auto p = b; p.instruments[0].fwd.coupons[0].spread += 0.001; return p; }(), "a float spread");
  differs([&] { auto p = b; p.instruments[0].fwd.coupons[0].scale = 0.5; return p; }(), "a notional scale");
  differs([&] { auto p = b; p.instruments[0].fixed.coupons[0].tau += 0.01; return p; }(), "a fixed-leg accrual");
  differs([&] { auto p = b; p.instruments[0].convexity += 0.001; return p; }(), "a futures convexity adj");
  differs([&] { auto p = b; px::Turn tn; tn.start = 0.9; tn.end = 1.1; p.curves[0].turns.push_back(tn); return p; }(), "an added turn");
  differs([&] { auto p = b; std::swap(p.instruments[0], p.instruments[1]); return p; }(), "reordered instruments");
}
