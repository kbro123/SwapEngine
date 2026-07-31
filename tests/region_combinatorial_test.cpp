// Order-agnostic region composition — the combinatorial proof that EVERY interpolation scheme can be a
// leading, a middle, and a trailing region, in ANY combination. There is no "front"/"back" concept: a
// ModularCurve is an ordered list of region modules stitched left-to-right by a C0 boundary handoff
// (curve_module.hpp), and this file exhausts the compositions of that contract.
//
// The single canonical scheme list is ALL_SCHEMES below; EVERY test is driven off it, so adding a new
// Scheme to the enum + this array automatically extends single/pair/triple coverage. The tests assert
// the properties that must hold for ANY valid composition:
//   * single region (7)           : builds, DF(0)=1, DF>0/finite/strictly-decreasing, integral continuous,
//                                    and forward-at-knot round-trips for the interpolating schemes.
//   * ordered pairs (7x7 = 49)     : builds, integral/log-discount is C0 across the A->B join (the FORWARD
//                                    may legitimately jump — Flat steps, BSpline clamped ends — so only the
//                                    integral is asserted continuous), DF>0/finite throughout.
//   * ordered triples (7^3 = 343)  : every scheme in the MIDDLE, preceded AND succeeded by every other;
//                                    C0 continuity + positivity at BOTH joins.
//   * calibration regression       : a LEADING B-spline curve calibrates through the AAD path (the just-
//                                    fixed de-Boor derivative-width crash), and a Hermite->BSpline curve
//                                    calibrates (guards the following-region case did not regress).
//
// QuantLib-free: this lives in the engine-only `swaps_tests` binary.

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <string>
#include <vector>

#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/curve/regions.hpp"
#include "swaps/pricing/cashflows.hpp"

using swaps::curve::CurveModule;
using swaps::curve::make_modular_curve;
using swaps::curve::Scheme;
namespace cal = swaps::calibration;
namespace cv = swaps::curve;
namespace px = swaps::pricing;

namespace {

// THE canonical list. Every combinatorial test iterates over this, so extending the enum + this array is
// all it takes to grow coverage. Order matches enum class Scheme (curve_module.hpp).
constexpr Scheme ALL_SCHEMES[] = {Scheme::Flat,   Scheme::Linear,  Scheme::NaturalCubic, Scheme::Hermite,
                                  Scheme::MonotoneCubic, Scheme::BSpline, Scheme::Tension};
constexpr int kNScheme = static_cast<int>(sizeof(ALL_SCHEMES) / sizeof(ALL_SCHEMES[0]));

const char* name(Scheme s) {
  switch (s) {
    case Scheme::Flat: return "Flat";
    case Scheme::Linear: return "Linear";
    case Scheme::NaturalCubic: return "NaturalCubic";
    case Scheme::Hermite: return "Hermite";
    case Scheme::MonotoneCubic: return "MonotoneCubic";
    case Scheme::BSpline: return "BSpline";
    case Scheme::Tension: return "Tension";
  }
  return "?";
}

// Every scheme interpolates its knot values EXCEPT BSpline, whose free values are control points that do
// not lie on the curve (regions.hpp) — so only there is forward-at-knot != the input value.
bool interpolates_knots(Scheme s) { return s != Scheme::BSpline; }

// Factory: a region of `scheme` over `knots`, supplying the per-scheme extras (Tension's σ = 1.0). Every
// other constraint (>= 3 strictly-increasing knots; time-ordering across regions) is the caller's job and
// is satisfied by region_knots() below.
CurveModule make_region(Scheme scheme, std::vector<double> knots) {
  CurveModule m;
  m.knots = std::move(knots);
  m.scheme = scheme;
  m.sigma = (scheme == Scheme::Tension) ? 1.0 : 0.0;  // 0 => region default (1.0) for the rest
  return m;
}

// Region r (0-based) gets three strictly-increasing knots sitting strictly AFTER region r-1's last knot,
// so any sequence of these regions time-orders (check_region_joins). A gap of 1.0 separates the blocks.
std::vector<double> region_knots(int r) {
  const double base = 2.0 * r;
  return {base + 0.5, base + 1.0, base + 1.5};
}

// A gentle, strictly-positive forward level — bounded in ~[0.026, 0.034], so even a global natural-cubic
// cannot overshoot to a non-positive forward (which would break strict monotonicity of the discount).
double fwd_level(double t) { return 0.030 + 0.004 * std::sin(1.3 * t); }

// The knot-forward vector for a curve: fwd_level at each knot of each region, region by region.
Eigen::VectorXd forwards_for(const std::vector<CurveModule>& spec) {
  std::vector<double> v;
  for (const auto& m : spec)
    for (double k : m.knots) v.push_back(fwd_level(k));
  Eigen::VectorXd x(static_cast<int>(v.size()));
  for (int i = 0; i < x.size(); ++i) x[i] = v[i];
  return x;
}

// Integral (log-discount) is C0 across a join at t_join: sample just left/right and assert no jump. The
// forward may legitimately step here, so we assert ONLY the integral. eps is tiny so the genuine slope
// contribution (~forward*2eps ~ 1e-10) stays well under the 1e-9 gate; a real handoff bug is O(rate).
void expect_integral_c0(const cv::ModularCurve<double>& c, double t_join, const char* where) {
  const double eps = 1e-9;
  const double li = c.integral(t_join - eps), ri = c.integral(t_join + eps);
  EXPECT_LT(std::abs(ri - li), 1e-9) << where << ": integral jumps across join at t=" << t_join
                                     << " (|Δ|=" << std::abs(ri - li) << ")";
  const double ld = c.discount(t_join - eps), rd = c.discount(t_join + eps);
  EXPECT_LT(std::abs(rd - ld), 1e-9) << where << ": discount jumps across join at t=" << t_join;
}

// DF strictly positive & finite on a fine grid over [0, t_hi].
void expect_positive_finite(const cv::ModularCurve<double>& c, double t_hi) {
  for (double t = 0.0; t <= t_hi + 1e-12; t += t_hi / 400.0) {
    const double df = c.discount(t);
    ASSERT_TRUE(std::isfinite(df)) << "DF non-finite at t=" << t;
    ASSERT_GT(df, 0.0) << "DF non-positive at t=" << t;
  }
}

// ---- Calibration harness (leading-BSpline regression) --------------------------------------------
// A calibration problem over an ARBITRARY region spec. It duck-types the calibration interface
// (residuals<Scalar>/n_knots/n_residuals), so it routes through the generic AAD residual engine — which
// rebuilds the curve with the AAD scalar every Jacobian sweep, exactly the path that crashed when the
// leading region was a B-spline (empty vs width-m derivative in de Boor). No W-cache here, so a spec
// containing MonotoneCubic (non-linear) would work too.
struct ModularProblem {
  cal::CalibrationProblem inst;      // instruments (schedules + market)
  std::vector<CurveModule> spec;     // the curve layout being calibrated
  int n_knots() const { return inst.n_residuals(); }  // square system: one free value per instrument
  int n_residuals() const { return inst.n_residuals(); }
  Eigen::VectorXd market() const {
    Eigen::VectorXd m(inst.n_residuals());
    for (int i = 0; i < m.size(); ++i) m[i] = inst.instruments[i].market;
    return m;
  }
  template <class Scalar, class Vec>
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residuals(const Vec& x) const {
    auto c = cv::make_modular_curve<Scalar>(spec);
    c.set_forwards(x);
    return inst.template price_residuals<Scalar>(c);
  }
};

// A self-discounting annual OIS as a generic ParRate instrument (mirrors bspline_test.cpp::make_ois).
cal::Instrument make_ois(double T) {
  std::vector<double> ends;
  for (double u = 1.0; u < T - 1e-9; u += 1.0) ends.push_back(u);
  ends.push_back(T);
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  double prev = 0.0;
  for (double u : ends) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev};
    c.obs.sub_end = {u};
    c.obs.tau_index = u - prev;
    c.pay = u;
    c.tau_pay = u - prev;
    ins.fwd.coupons.push_back(c);
    ins.fixed.coupons.push_back({u, u - prev});
    prev = u;
  }
  return ins;
}

}  // namespace

// =================================================================================================
// (a) SINGLE REGION — each scheme as the sole region.
// =================================================================================================
TEST(RegionCombinatorial, SingleRegion) {
  for (int i = 0; i < kNScheme; ++i) {
    const Scheme s = ALL_SCHEMES[i];
    SCOPED_TRACE(std::string("single=") + name(s));
    const std::vector<CurveModule> spec{make_region(s, region_knots(0))};
    ASSERT_NO_THROW(make_modular_curve<double>(spec));
    auto c = make_modular_curve<double>(spec);
    const Eigen::VectorXd x = forwards_for(spec);
    ASSERT_EQ(c.n_knots(), static_cast<int>(x.size()));
    ASSERT_NO_THROW(c.set_forwards(x));

    EXPECT_DOUBLE_EQ(c.discount(0.0), 1.0) << "DF(0) must be 1";
    const double t_hi = spec[0].knots.back() + 2.0;  // include the flat-extrapolation tail
    expect_positive_finite(c, t_hi);

    // A LEADING region flat-extrapolates its FIRST FREE value v1=x[0] backwards: forward is FLAT (== v1)
    // for every t in (0, t1), exactly mirroring the far-end flat extrapolation beyond the last knot. This
    // locks in the corrected short end — previously a non-Flat leading region pinned a phantom node-0 to 0
    // and RAMPED up to v1. Holds for every scheme (Flat included): the pre-first-knot forward is the first
    // value, never 0, never a ramp. (BSpline's first value is its first control point x[0], = the clamp.)
    {
      const double t1 = spec[0].knots.front(), v1 = x[0];
      for (double frac : {0.05, 0.25, 0.5, 0.75, 0.99}) {
        const double t = frac * t1;
        EXPECT_NEAR(c.forward(t), v1, 1e-10)
            << "leading " << name(s) << " must be FLAT == v1(=x[0]) on (0,t1) at t=" << t;
      }
      EXPECT_GT(v1, 0.0) << "sanity: the flat short-end value is the calibrated first knot, not a phantom 0";
    }

    // Strictly decreasing DF (positive forwards). forward(0)=v1>0 now (flat short end), so DF decreases
    // from t=0; sampling from 0.02 keeps the assert robust to the DF(0)=1 exact tie.
    double prev = 1.0;  // DF(0)
    for (double t = 0.02; t <= t_hi; t += 0.02) {
      const double df = c.discount(t);
      EXPECT_LT(df, prev) << "DF must strictly decrease at t=" << t;
      prev = df;
    }

    // Integral continuity on a fine grid (no internal jumps).
    for (double t = 0.05; t <= t_hi; t += 0.05) {
      const double eps = 1e-9;
      EXPECT_LT(std::abs(c.integral(t + eps) - c.integral(t - eps)), 1e-8)
          << "integral discontinuous at t=" << t;
    }

    // Forward-at-knot round-trip for the interpolating schemes.
    if (interpolates_knots(s)) {
      for (std::size_t k = 0; k < spec[0].knots.size(); ++k)
        EXPECT_NEAR(c.forward(spec[0].knots[k]), x[static_cast<int>(k)], 1e-10)
            << "forward must interpolate knot " << k;
    }
  }
}

// =================================================================================================
// (b) TWO REGION — all 49 ordered pairs (A,B). Core proof: each scheme can PRECEDE and SUCCEED every
// other, with a C0 integral/log-discount handoff at the join.
// =================================================================================================
TEST(RegionCombinatorial, AllOrderedPairs) {
  int built = 0;
  for (int a = 0; a < kNScheme; ++a) {
    for (int b = 0; b < kNScheme; ++b) {
      const Scheme A = ALL_SCHEMES[a], B = ALL_SCHEMES[b];
      SCOPED_TRACE(std::string("pair ") + name(A) + "->" + name(B));
      const std::vector<CurveModule> spec{make_region(A, region_knots(0)), make_region(B, region_knots(1))};
      ASSERT_NO_THROW(make_modular_curve<double>(spec)) << "pair failed to build";
      auto c = make_modular_curve<double>(spec);
      const Eigen::VectorXd x = forwards_for(spec);
      ASSERT_EQ(c.n_knots(), static_cast<int>(x.size()));
      ASSERT_NO_THROW(c.set_forwards(x));

      EXPECT_DOUBLE_EQ(c.discount(0.0), 1.0);
      const double t_join = spec[0].knots.back();  // A ends here; B covers (t_join, ...]
      expect_integral_c0(c, t_join, "pair");
      expect_positive_finite(c, spec[1].knots.back() + 2.0);
      ++built;
    }
  }
  EXPECT_EQ(built, kNScheme * kNScheme) << "every ordered pair must build & pass";
  std::cout << "  [pairs] " << built << " ordered pairs composed & C0-verified\n";
}

// =================================================================================================
// (c) THREE REGION — full 7^3 = 343 ordered triples X->M->Y. Every scheme M appears in the MIDDLE,
// preceded AND succeeded by every scheme; C0 + positivity checked at BOTH joins.
// =================================================================================================
TEST(RegionCombinatorial, AllOrderedTriples) {
  int built = 0;
  for (int a = 0; a < kNScheme; ++a) {
    for (int m = 0; m < kNScheme; ++m) {
      for (int b = 0; b < kNScheme; ++b) {
        const Scheme A = ALL_SCHEMES[a], M = ALL_SCHEMES[m], B = ALL_SCHEMES[b];
        SCOPED_TRACE(std::string("triple ") + name(A) + "->" + name(M) + "->" + name(B));
        const std::vector<CurveModule> spec{make_region(A, region_knots(0)),
                                            make_region(M, region_knots(1)),
                                            make_region(B, region_knots(2))};
        ASSERT_NO_THROW(make_modular_curve<double>(spec)) << "triple failed to build";
        auto c = make_modular_curve<double>(spec);
        const Eigen::VectorXd x = forwards_for(spec);
        ASSERT_EQ(c.n_knots(), static_cast<int>(x.size()));
        ASSERT_NO_THROW(c.set_forwards(x));

        EXPECT_DOUBLE_EQ(c.discount(0.0), 1.0);
        expect_integral_c0(c, spec[0].knots.back(), "triple join-1");
        expect_integral_c0(c, spec[1].knots.back(), "triple join-2");
        expect_positive_finite(c, spec[2].knots.back() + 2.0);
        ++built;
      }
    }
  }
  EXPECT_EQ(built, kNScheme * kNScheme * kNScheme) << "every ordered triple must build & pass";
  std::cout << "  [triples] " << built << " ordered triples composed & C0-verified\n";
}

// =================================================================================================
// (d) CALIBRATION REGRESSION — the specific bug: a LEADING B-spline region must calibrate through the
// AAD path without the de-Boor derivative-width crash, to a low residual.
// =================================================================================================
TEST(RegionCombinatorial, LeadingBSplineCalibrates) {
  ModularProblem prob;
  const std::vector<double> knots{1, 2, 3, 4, 5, 7, 10};  // 7 control points => 7 free vars
  prob.spec = {make_region(Scheme::BSpline, knots)};      // B-spline is the LEADING (and only) region
  const std::vector<double> mats{1, 2, 3, 4, 5, 7, 10};   // 7 OIS => square, consistent
  for (double T : mats) prob.inst.instruments.push_back(make_ois(T));

  // Self-consistent market from a known B-spline (control points), so x_true zeros the residual.
  Eigen::VectorXd xt(7);
  xt << 0.030, 0.033, 0.036, 0.039, 0.041, 0.043, 0.046;
  auto ct = cv::make_modular_curve<double>(prob.spec);
  ct.set_forwards(xt);
  for (auto& ins : prob.inst.instruments)
    ins.market = px::par_rate<double>(ins.fwd.coupons, ins.fixed.coupons, ct, ct);
  ASSERT_LT(prob.residuals<double>(xt).cwiseAbs().maxCoeff(), 1e-12) << "x_true must zero the residual";

  const auto res = cal::calibrate(prob, Eigen::VectorXd::Constant(7, 0.035));
  const double reprice = prob.residuals<double>(res.x).cwiseAbs().maxCoeff();
  std::cout << "  [leading-bspline-calib] reprice=" << reprice << " stat=" << res.stationarity
            << " iters=" << res.iterations << "\n";
  EXPECT_LT(reprice, 1e-9) << "leading-B-spline curve must reprice the market it was fit to";
  EXPECT_LT(res.stationarity, 1e-6) << "first-order optimality";

  // The leading short end must anchor to the FITTED first control point, not a phantom 0: the flat
  // pre-segment [0,t1] (t1 = first knot = 1.0) equals res.x[0] (the calibrated clamp start), far from 0.
  auto cc = cv::make_modular_curve<double>(prob.spec);
  cc.set_forwards(res.x);
  for (double t : {0.1, 0.4, 0.9})
    EXPECT_NEAR(cc.forward(t), res.x[0], 1e-10) << "leading B-spline short end must equal fitted x[0] at t=" << t;
  EXPECT_GT(res.x[0], 0.005) << "fitted short-end forward must be a real calibrated value, not ~0";
}

// A LEADING interpolating region (Hermite) must calibrate its FIRST knot as a free value and flat-
// extrapolate it to the short end — the pre-t1 forward equals the FITTED v1, NEVER 0 and NEVER a ramp.
// This is the crux: a leading region must not anchor its short end to a phantom zero.
TEST(RegionCombinatorial, LeadingHermiteShortEndIsCalibratedNotZero) {
  ModularProblem prob;
  const std::vector<double> knots{1, 2, 3, 4, 5, 7, 10};
  prob.spec = {make_region(Scheme::Hermite, knots)};       // Hermite is the LEADING (and only) region
  const std::vector<double> mats{1, 2, 3, 4, 5, 7, 10};     // 7 OIS => square
  for (double T : mats) prob.inst.instruments.push_back(make_ois(T));

  Eigen::VectorXd xt(7);
  xt << 0.030, 0.033, 0.036, 0.039, 0.041, 0.043, 0.046;    // forward-at-knot (Hermite interpolates)
  auto ct = cv::make_modular_curve<double>(prob.spec);
  ct.set_forwards(xt);
  for (auto& ins : prob.inst.instruments)
    ins.market = px::par_rate<double>(ins.fwd.coupons, ins.fixed.coupons, ct, ct);
  ASSERT_LT(prob.residuals<double>(xt).cwiseAbs().maxCoeff(), 1e-12) << "x_true must zero the residual";

  const auto res = cal::calibrate(prob, Eigen::VectorXd::Constant(7, 0.035));
  const double reprice = prob.residuals<double>(res.x).cwiseAbs().maxCoeff();
  std::cout << "  [leading-hermite-calib] reprice=" << reprice << " x0=" << res.x[0]
            << " stat=" << res.stationarity << "\n";
  EXPECT_LT(reprice, 1e-9) << "leading Hermite curve must reprice the market it was fit to";
  EXPECT_LT(res.stationarity, 1e-6) << "first-order optimality";

  auto cc = cv::make_modular_curve<double>(prob.spec);
  cc.set_forwards(res.x);
  EXPECT_NEAR(cc.forward(knots.front()), res.x[0], 1e-12) << "Hermite interpolates its first knot";
  for (double t : {0.1, 0.4, 0.9})  // t1 = 1.0
    EXPECT_NEAR(cc.forward(t), res.x[0], 1e-12) << "leading short end must be FLAT == fitted v1 at t=" << t;
  EXPECT_GT(res.x[0], 0.005) << "fitted short-end forward must be a real calibrated value, not ~0";
  EXPECT_NEAR(res.x[0], xt[0], 1e-6) << "the fitted first knot recovers the true first forward";
}

TEST(RegionCombinatorial, HermiteThenBSplineCalibrates) {
  // Guards the FOLLOWING-region case: the fix must not have regressed a B-spline that succeeds another
  // region (here Hermite), where the pinned control point carries the join's REAL sensitivities.
  ModularProblem prob;
  const std::vector<double> hermite_knots{0.5, 1, 2};        // 3 leading Hermite knots
  const std::vector<double> bspline_knots{3, 4, 5, 7, 10};   // 5 following B-spline control points
  prob.spec = {make_region(Scheme::Hermite, hermite_knots), make_region(Scheme::BSpline, bspline_knots)};
  const std::vector<double> mats{0.5, 1, 2, 3, 4, 5, 7, 10};  // 8 OIS = 3 + 5 free vars (square)
  for (double T : mats) prob.inst.instruments.push_back(make_ois(T));

  Eigen::VectorXd xt(8);
  xt << 0.028, 0.031, 0.034, 0.037, 0.039, 0.041, 0.043, 0.046;
  auto ct = cv::make_modular_curve<double>(prob.spec);
  ct.set_forwards(xt);
  for (auto& ins : prob.inst.instruments)
    ins.market = px::par_rate<double>(ins.fwd.coupons, ins.fixed.coupons, ct, ct);
  ASSERT_LT(prob.residuals<double>(xt).cwiseAbs().maxCoeff(), 1e-12) << "x_true must zero the residual";

  const auto res = cal::calibrate(prob, Eigen::VectorXd::Constant(8, 0.035));
  const double reprice = prob.residuals<double>(res.x).cwiseAbs().maxCoeff();
  std::cout << "  [hermite->bspline-calib] reprice=" << reprice << " stat=" << res.stationarity
            << " iters=" << res.iterations << "\n";
  EXPECT_LT(reprice, 1e-9) << "Hermite->B-spline curve must reprice the market it was fit to";
  EXPECT_LT(res.stationarity, 1e-6) << "first-order optimality";
}
