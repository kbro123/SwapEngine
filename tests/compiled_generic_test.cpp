// Design §4 gate: the compiled W-cache engine on the GENERIC coupon.
//
// The pre-existing compiled tests (Compiled.AnalyticJacobianMatchesAad,
// BundleRealistic.CompiledBundleResidualMatchesAad, BundleSpread.CompiledResidualHandlesSpreadCurves)
// only ever exercise the LEGACY shape: one unit-weight sub-period per coupon, no spread,
// tau_pay == tau_index. They would all still pass if every new term (w_k, spread, tau_pay != tau_index,
// realized-inside-a-leg) were silently dropped. This file pins the new degrees of freedom directly on
// pricing::BundleFloatBatch, against the templated kernel (values) and AAD (the analytic Jacobian).
//
// Deliberately exercised here, because each is a place the analytic chain can be subtly wrong while
// every other test stays green:
//   - w_k != 1                       (weights are R_sub's VALUES, and the scatter must read them back)
//   - spread != 0                    (enters ONLY via d pv/d DF[pay] = A·k -- easy to drop)
//   - tau_pay != tau_index (k != 1)  (30/360 pay vs ACT/360 index)
//   - a payment lag (pay != accrual end)
//   - a FULLY FIXED coupon summed with live siblings (empty sub-periods)
//   - CONTIGUOUS sub-periods (e_k == s_{k+1}) and pay == e_k when forecast == discount -- the two
//     structural aliases that make `+=` (not `=`) load-bearing in the scatter
//   - a leg forecasting a SPREAD curve while discounting its base (W_all ancestry columns)

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <iostream>
#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "swaps/pricing/compiled_book.hpp"

namespace px = swaps::pricing;
namespace cal = swaps::calibration;

namespace {

const std::vector<double> kMeeting{0.25, 0.5};
const std::vector<double> kBack{1.0, 2.0, 3.0, 5.0, 10.0};
constexpr int kNk = 7;  // per curve: 2 front + 5 back

// Curve 0 = outright base; curve 1 = base + forward spread. A leg forecasting curve 1 and discounting
// curve 0 therefore drags the base-ancestry columns of W_all into its Jacobian row.
std::vector<cal::BundleCurveSpec> specs() {
  const auto m = swaps::curve::flat_hermite(kMeeting, kBack);
  return {{.base = -1, .regions = m}, {.base = 0, .regions = m}};
}

Eigen::VectorXd stacked_knots() {
  Eigen::VectorXd x(2 * kNk);
  for (int i = 0; i < kNk; ++i) {
    x[i] = 0.040 + 0.0011 * i;         // base forwards
    x[kNk + i] = 0.0050 + 0.0004 * i;  // forward spreads
  }
  return x;
}

px::RateObservation obs(std::vector<double> s, std::vector<double> e, std::vector<double> w,
                        double realized, double tau_index) {
  px::RateObservation o;
  o.sub_start = std::move(s);
  o.sub_end = std::move(e);
  o.weight = std::move(w);
  o.realized = realized;
  o.tau_index = tau_index;
  return o;
}

// Every coupon shape the generic model admits, in one leg.
std::vector<px::FloatCoupon> rich_leg() {
  std::vector<px::FloatCoupon> leg;
  // (1) plain compounded shape: one sub-period, tau_pay == tau_index, no spread (the legacy form).
  leg.push_back({obs({0.0}, {0.5}, {}, 0.0, 0.5), 0.5, 0.5, 0.0});
  // (2) contractual spread.
  leg.push_back({obs({0.5}, {1.0}, {}, 0.0, 0.5), 1.0, 0.5, 25e-4});
  // (3) tau_pay != tau_index (30/360 pay vs ACT/360 index), a payment LAG, and a negative spread.
  leg.push_back({obs({1.0}, {1.5}, {}, 0.0, 0.5069444), 1.5137, 0.5, -10e-4});
  // (4) averaged, NON-UNIT weights, partially fixed (realized), contiguous sub-periods (e_k == s_{k+1}).
  leg.push_back({obs({1.5, 1.6, 1.7, 1.8}, {1.6, 1.7, 1.8, 2.0}, {0.8, 1.3, 1.0, 2.1}, 3.1e-3, 0.5083),
                 2.0083, 0.5, 5e-4});
  // (5) FULLY FIXED coupon (no sub-periods) summed with live siblings.
  leg.push_back({obs({}, {}, {}, 4.2e-3, 0.5), 2.5, 0.5, 0.0});
  // (6) multi-sub-period, unit weights (the plain averaging shape).
  leg.push_back({obs({2.5, 2.75}, {2.75, 3.0}, {}, 0.0, 0.5), 3.0, 0.5, 0.0});
  return leg;
}

// Self-discounting leg whose pay date IS the accrual end -- `pay` and `e_k` then register to the SAME
// global DF index, so d pv/d DF[pay] and d pv/d DF[e] land in the same cell and MUST accumulate.
std::vector<px::FloatCoupon> aliasing_leg() {
  std::vector<px::FloatCoupon> leg;
  for (double t = 0.0; t < 3.0 - 1e-9; t += 1.0)
    leg.push_back({obs({t}, {t + 1.0}, {}, 0.0, 1.0), t + 1.0, 1.0, 0.0});
  return leg;
}

struct Book {
  px::CompiledCurveSet cs;
  px::BundleFloatBatch fl;
  Eigen::VectorXd DF;
};

// Register both legs (leg A forecasts the SPREAD curve, discounts the base; leg B is self-discounting).
Book build(const Eigen::VectorXd& x, const std::vector<px::FloatCoupon>& a,
           const std::vector<px::FloatCoupon>& b) {
  Book bk;
  std::vector<px::CurveStructure> st;
  for (const auto& s : specs()) st.push_back(s);
  bk.cs.init(st);
  bk.fl.add(bk.cs, /*fc=*/1, /*dc=*/0, a);
  bk.fl.add(bk.cs, /*fc=*/0, /*dc=*/0, b);
  bk.cs.finalize();
  bk.fl.finalize();
  bk.DF = bk.cs.df(x);
  return bk;
}

}  // namespace

TEST(CompiledGeneric, FloatBatchMatchesTemplatedKernel) {
  const Eigen::VectorXd x = stacked_knots();
  const auto a = rich_leg(), b = aliasing_leg();
  Book bk = build(x, a, b);
  const Eigen::VectorXd pv = bk.fl.pv(bk.DF);

  const auto C = cal::build_bundle_curves<double>(specs(), [&](int c, int i) { return x[c * kNk + i]; });
  const double ref_a = px::float_leg_pv<double>(a, *C[1], *C[0]);  // forecast spread curve, discount base
  const double ref_b = px::float_leg_pv<double>(b, *C[0], *C[0]);  // self-discounting

  std::cout << "  [compiled-generic] rich leg  compiled=" << pv[0] << " kernel=" << ref_a
            << " |d|=" << std::abs(pv[0] - ref_a) << "\n"
            << "  [compiled-generic] alias leg compiled=" << pv[1] << " kernel=" << ref_b
            << " |d|=" << std::abs(pv[1] - ref_b) << "\n";
  EXPECT_LT(std::abs(pv[0] - ref_a), 1e-15) << "compiled float batch must match the templated kernel";
  EXPECT_LT(std::abs(pv[1] - ref_b), 1e-15);
}

TEST(CompiledGeneric, AnalyticJacobianMatchesAadOnEveryCouponShape) {
  const Eigen::VectorXd x = stacked_knots();
  const auto a = rich_leg(), b = aliasing_leg();
  Book bk = build(x, a, b);

  // Analytic: J = -(dpv/dDF · diag(DF)) · W_all, exactly the factorization the residual uses.
  Eigen::MatrixXd G = Eigen::MatrixXd::Zero(2, bk.cs.n_times());
  bk.fl.d_pv_from_num(bk.fl.num(bk.DF), bk.DF, G, /*row0=*/0, /*sign=*/1.0);
  const Eigen::MatrixXd J = -((G * bk.DF.asDiagonal()) * bk.cs.W());

  // AAD: one differentiated pass of the templated kernel over the stacked x.
  const auto xd = swaps::ad::seed(x);
  const auto Cd = cal::build_bundle_curves<swaps::ad::Dual>(specs(),
                                                            [&](int c, int i) { return xd[c * kNk + i]; });
  Eigen::MatrixXd Jaad(2, x.size());
  Jaad.row(0) = px::float_leg_pv<swaps::ad::Dual>(a, *Cd[1], *Cd[0]).derivatives().transpose();
  Jaad.row(1) = px::float_leg_pv<swaps::ad::Dual>(b, *Cd[0], *Cd[0]).derivatives().transpose();

  const double worst = (J - Jaad).cwiseAbs().maxCoeff();
  std::cout << "  [compiled-generic] float |analytic J - AAD| = " << worst << "\n";
  EXPECT_LT(worst, 1e-12) << "analytic Jacobian must match AAD on every generic coupon shape";
}

// A sign/spread error can hide inside a leg PV but not in the SIGNED difference of two legs -- the
// basis quote's shape. This is the transform the bundle residual actually uses for `bases`.
TEST(CompiledGeneric, AnalyticJacobianMatchesAadWithOppositeSign) {
  const Eigen::VectorXd x = stacked_knots();
  const auto a = rich_leg(), b = aliasing_leg();
  Book bk = build(x, a, b);

  Eigen::MatrixXd G = Eigen::MatrixXd::Zero(2, bk.cs.n_times());
  bk.fl.d_pv_from_num(bk.fl.num(bk.DF), bk.DF, G, /*row0=*/0, /*sign=*/-1.0);
  const Eigen::MatrixXd J = -((G * bk.DF.asDiagonal()) * bk.cs.W());

  const auto xd = swaps::ad::seed(x);
  const auto Cd = cal::build_bundle_curves<swaps::ad::Dual>(specs(),
                                                            [&](int c, int i) { return xd[c * kNk + i]; });
  Eigen::MatrixXd Jaad(2, x.size());
  Jaad.row(0) = -px::float_leg_pv<swaps::ad::Dual>(a, *Cd[1], *Cd[0]).derivatives().transpose();
  Jaad.row(1) = -px::float_leg_pv<swaps::ad::Dual>(b, *Cd[0], *Cd[0]).derivatives().transpose();

  const double worst = (J - Jaad).cwiseAbs().maxCoeff();
  std::cout << "  [compiled-generic] signed float |analytic J - AAD| = " << worst << "\n";
  EXPECT_LT(worst, 1e-12);
}

// A future is the SAME batch stopping at rate + convexity (design §4). Mixing a weighted, partially
// fixed averaging future with a single-period (IBOR/compounded) future in ONE batch also forces the
// non-identity R_sub path.
TEST(CompiledGeneric, FuturesBatchAndJacobianMatchTheKernelAndAad) {
  const Eigen::VectorXd x = stacked_knots();
  const auto o_avg = obs({1.5, 1.6, 1.7, 1.8}, {1.6, 1.7, 1.8, 2.0}, {1.4, 0.7, 1.0, 0.9}, 2.6e-3, 0.5083);
  const auto o_one = obs({3.0}, {3.25}, {}, 0.0, 0.2528);
  const double conv_avg = 1.7e-4, conv_one = 3.4e-4;

  std::vector<px::CurveStructure> st;
  for (const auto& s : specs()) st.push_back(s);
  px::CompiledCurveSet cs;
  cs.init(st);
  px::BundleFloatBatch fu;
  fu.add_future(cs, /*fc=*/1, o_avg, conv_avg);  // averaging future on the spread curve
  fu.add_future(cs, /*fc=*/0, o_one, conv_one);  // single-period future on the base curve
  cs.finalize();
  fu.finalize();
  const Eigen::VectorXd DF = cs.df(x);

  const auto C = cal::build_bundle_curves<double>(specs(), [&](int c, int i) { return x[c * kNk + i]; });
  const Eigen::VectorXd r = fu.rate(DF);
  const double ref0 = px::future_rate<double>(o_avg, conv_avg, *C[1]);
  const double ref1 = px::future_rate<double>(o_one, conv_one, *C[0]);
  std::cout << "  [compiled-generic] futures |d| = " << std::abs(r[0] - ref0) << ", "
            << std::abs(r[1] - ref1) << "\n";
  // The batch computes DF[s]·(1/DF[e]) with a SHARED reciprocal (compiled_book.hpp inverse_of); the templated
  // kernel divides. a·(1/b) vs a/b differ by ≤ 1 ULP, so this is a few-ULP check, not bit identity.
  EXPECT_LT(std::abs(r[0] - ref0), 8e-15);
  EXPECT_LT(std::abs(r[1] - ref1), 8e-15);

  Eigen::MatrixXd G = Eigen::MatrixXd::Zero(2, cs.n_times());
  fu.d_rate(DF, G, /*row0=*/0);
  const Eigen::MatrixXd J = -((G * DF.asDiagonal()) * cs.W());

  const auto xd = swaps::ad::seed(x);
  const auto Cd = cal::build_bundle_curves<swaps::ad::Dual>(specs(),
                                                            [&](int c, int i) { return xd[c * kNk + i]; });
  Eigen::MatrixXd Jaad(2, x.size());
  Jaad.row(0) = px::future_rate<swaps::ad::Dual>(o_avg, conv_avg, *Cd[1]).derivatives().transpose();
  Jaad.row(1) = px::future_rate<swaps::ad::Dual>(o_one, conv_one, *Cd[0]).derivatives().transpose();

  const double worst = (J - Jaad).cwiseAbs().maxCoeff();
  std::cout << "  [compiled-generic] futures |analytic J - AAD| = " << worst << "\n";
  EXPECT_LT(worst, 1e-12) << "analytic futures Jacobian must match AAD";
}

// The identity fast path in pv()/rate() is a bitwise shortcut for R_sub == I, not a different model:
// re-registering the SAME legs with explicit unit weights must reproduce it to the last bit.
TEST(CompiledGeneric, IdentityFastPathIsExactlyTheGeneralPath) {
  const Eigen::VectorXd x = stacked_knots();
  const auto b = aliasing_leg();
  Book fast = build(x, rich_leg(), b);
  ASSERT_FALSE(fast.fl.sub_is_identity) << "rich leg must exercise the general R_sub path";

  // The aliasing leg alone is one unit-weight sub-period per coupon => the identity path.
  Book id = build(x, b, b);
  ASSERT_TRUE(id.fl.sub_is_identity) << "plain one-sub-period legs must take the identity fast path";

  // num() honours the same shortcut; compare it against an explicit R_sub reduction of the same data,
  // written in the batch's own reciprocal form (DF[s]·INV[e] − 1, compiled_book.hpp inverse_of).
  // (a plain loop, not an Eigen expression: clang contracts a*b-1 into an FMA in a scalar loop but not
  // inside Eigen's packet evaluation, and this test is about the R_sub shortcut, not about contraction).
  const Eigen::VectorXd n_fast = id.fl.num(id.DF);
  const Eigen::VectorXd inv = id.DF.cwiseInverse();
  Eigen::VectorXd sub(id.fl.subS.size());
  for (int i = 0; i < sub.size(); ++i) sub[i] = id.DF[id.fl.subS[i]] * inv[id.fl.subE[i]] - 1.0;
  const Eigen::VectorXd n_gen = id.fl.R_sub * sub;
  EXPECT_EQ(n_fast, n_gen) << "identity shortcut must be bit-identical to the sparse reduction";
}
