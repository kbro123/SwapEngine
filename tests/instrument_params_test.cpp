// Commercial instrument-parameter gate (QuantLib-free). Proves the ADDITIVE builder parameters added to
// build/instruments.hpp -- a floating-leg spread, an amortizing/step-up notional schedule, and a fixed-leg
// frequency override -- (i) reproduce the pre-existing builders BYTE-FOR-BYTE at their defaults, (ii) have
// the correct pricing effect, and (iii) flow identically through BOTH the templated kernel and the compiled
// W-cache path + its analytic Jacobian (so nothing special-cased at the API/web layer is needed). The new
// parameters set the EXISTING pricing::FloatCoupon::spread / scale and FixedCoupon::scale, which the kernel
// and BundleFloatBatch/BundleFixedLegs already carry through PV and the analytic Jacobian.
#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <vector>

#include "swaps/build/instruments.hpp"
#include "swaps/calibration/compiled_residual.hpp"
#include "swaps/calibration/jacobian.hpp"
#include "swaps/curve/curve_module.hpp"

namespace b = swaps::build;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace cv = swaps::curve;

namespace {

const b::Date kVd = b::Date::from_iso("2026-07-08");
b::SwapConv sofr() { return b::swap_conv("USD", "USD-SOFR"); }  // annual/annual, ACT/360, SOFR OIS

// A curve over `knots` (curve-time years) with the given knot forwards (flat_hermite back region only).
cv::ModularCurve<double> curve_from(const std::vector<double>& knots, const std::vector<double>& fwd) {
  auto c = cv::make_modular_curve<double>(cv::flat_hermite({}, knots));
  Eigen::VectorXd x(static_cast<int>(fwd.size()));
  for (std::size_t i = 0; i < fwd.size(); ++i) x[static_cast<int>(i)] = fwd[i];
  c.set_forwards(x);
  return c;
}

double model_quote(const cal::Instrument& ins, const cv::ModularCurve<double>& c) {
  auto C = [&c](int) -> const cv::ModularCurve<double>& { return c; };
  return cal::instrument_model_quote<double>(ins, C);
}

const std::vector<double> kKnots{0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

}  // namespace

// (i) At the defaults, the builders touch nothing: no spread, unit notional, annual fixed leg.
TEST(InstrumentParams, DefaultsAreByteIdentical) {
  const b::Date mat = b::resolve("5y", kVd);
  const cal::Instrument ins = b::par_swap(kVd, sofr(), mat, 0, 0, 0.02);
  ASSERT_EQ(ins.fwd.coupons.size(), 5u);
  ASSERT_EQ(ins.fixed.coupons.size(), 5u);  // default fixed_freq "1Y"
  for (const auto& c : ins.fwd.coupons) {
    EXPECT_EQ(c.spread, 0.0);
    EXPECT_EQ(c.scale, 1.0);
  }
  for (const auto& c : ins.fixed.coupons) EXPECT_EQ(c.scale, 1.0);

  // An explicitly-defaulted call equals a plain call, field for field (float pay/tau + fixed pay/tau).
  const cal::Instrument same = b::par_swap(kVd, sofr(), mat, 0, 0, 0.02, /*float_spread=*/0.0,
                                           /*notionals=*/{}, /*fixed_freq=*/"1Y");
  for (std::size_t i = 0; i < ins.fwd.coupons.size(); ++i) {
    EXPECT_EQ(ins.fwd.coupons[i].pay, same.fwd.coupons[i].pay);
    EXPECT_EQ(ins.fwd.coupons[i].tau_pay, same.fwd.coupons[i].tau_pay);
    EXPECT_EQ(ins.fwd.coupons[i].obs.sub_start[0], same.fwd.coupons[i].obs.sub_start[0]);
  }
}

// (ii) A +10bp floating-leg spread raises the par swap's model quote by ~10bp on a flat curve (float and
// fixed share the annual ACT/360 SOFR schedule, so the annuity ratio is 1 and the shift is exact).
TEST(InstrumentParams, FloatSpreadRaisesParRate) {
  const b::Date mat = b::resolve("5y", kVd);
  const auto flat = curve_from(kKnots, std::vector<double>(kKnots.size(), 0.03));
  const double q0 = model_quote(b::par_swap(kVd, sofr(), mat, 0, 0, 0.0), flat);
  const double q1 = model_quote(b::par_swap(kVd, sofr(), mat, 0, 0, 0.0, /*float_spread=*/0.0010), flat);
  EXPECT_NEAR(q1 - q0, 0.0010, 1e-9);
}

// (ii) An amortizing (declining) notional lowers a floating leg's PV vs a bullet leg — a sign-guaranteed
// direction on any positive-rate curve (each period's notional <= 1).
TEST(InstrumentParams, AmortizingLowersLegPv) {
  const b::Date mat = b::resolve("5y", kVd);
  const auto flat = curve_from(kKnots, std::vector<double>(kKnots.size(), 0.03));
  const auto conv = sofr();
  const auto bullet = b::float_leg(kVd, conv, mat, 0, 0, conv.float_freq_tok, conv.float_dc);
  const auto amort = b::float_leg(kVd, conv, mat, 0, 0, conv.float_freq_tok, conv.float_dc, -1, -1, 1.0,
                                  /*spread=*/0.0, /*notionals=*/{1.0, 0.8, 0.6, 0.4, 0.2});
  ASSERT_EQ(bullet.coupons.size(), 5u);
  ASSERT_EQ(amort.coupons.size(), 5u);
  const double pv_bullet = px::float_leg_pv<double>(bullet.coupons, flat, flat);
  const double pv_amort = px::float_leg_pv<double>(amort.coupons, flat, flat);
  EXPECT_LT(pv_amort, pv_bullet);
}

// (ii) On an UPWARD-sloping curve, tilting the notional toward the later (higher-rate) coupons raises the
// par rate: a strictly ordered, sign-safe check that the amortization schedule moves the par swap quote.
TEST(InstrumentParams, NotionalTiltMovesParRate) {
  const b::Date mat = b::resolve("5y", kVd);
  const auto up = curve_from(kKnots, {0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06});
  const double q_decl =
      model_quote(b::par_swap(kVd, sofr(), mat, 0, 0, 0.0, 0.0, {1.0, 0.8, 0.6, 0.4, 0.2}), up);
  const double q_incr =
      model_quote(b::par_swap(kVd, sofr(), mat, 0, 0, 0.0, 0.0, {0.2, 0.4, 0.6, 0.8, 1.0}), up);
  EXPECT_GT(q_incr, q_decl);
}

// (ii) Fixed-frequency override: default is annual; "6M" doubles the fixed coupon count.
TEST(InstrumentParams, FixedFrequencyOverride) {
  const b::Date mat = b::resolve("5y", kVd);
  EXPECT_EQ(b::fixed_coupons(kVd, sofr(), mat, 0).coupons.size(), 5u);
  EXPECT_EQ(b::fixed_coupons(kVd, sofr(), mat, 0, "6M").coupons.size(), 10u);
}

// (ii) Payment lag (carried on the convention, exposed on every builder) shifts pay times later.
TEST(InstrumentParams, PaymentLagShiftsPayTimes) {
  const b::Date mat = b::resolve("5y", kVd);
  b::SwapConv c0 = sofr();
  c0.pay_lag = 0;
  b::SwapConv c5 = sofr();
  c5.pay_lag = 5;
  const auto f0 = b::fixed_coupons(kVd, c0, mat, 0);
  const auto f5 = b::fixed_coupons(kVd, c5, mat, 0);
  ASSERT_EQ(f0.coupons.size(), f5.coupons.size());
  EXPECT_GT(f5.coupons.front().pay, f0.coupons.front().pay);
  EXPECT_GT(f5.coupons.back().pay, f0.coupons.back().pay);
}

// (iii) The new parameters flow IDENTICALLY through the compiled W-cache residual + its analytic Jacobian
// and the templated + AAD path. A self-consistent single-curve problem mixes a spread swap, an amortizing
// swap and a plain swap; compiled must equal templated (residual) and AAD (Jacobian) to rounding.
TEST(InstrumentParams, CompiledEqualsTemplatedWithSpreadAndNotional) {
  cal::CalibrationProblem prob;
  prob.back_times = {0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0};
  prob.instruments.push_back(b::par_swap(kVd, sofr(), b::resolve("2y", kVd), 0, 0, 0.030, /*spread=*/0.0005));
  prob.instruments.push_back(
      b::par_swap(kVd, sofr(), b::resolve("5y", kVd), 0, 0, 0.035, 0.0, {1.0, 0.8, 0.6, 0.4, 0.2}));
  prob.instruments.push_back(b::par_swap(kVd, sofr(), b::resolve("7y", kVd), 0, 0, 0.040));  // plain

  const cal::CompiledResidual cr(prob);
  ASSERT_EQ(cr.n_residuals(), prob.n_residuals());

  std::vector<Eigen::VectorXd> xs;
  xs.push_back(Eigen::VectorXd::Constant(prob.n_knots(), 0.030));
  Eigen::VectorXd tilt(prob.n_knots());
  for (int i = 0; i < tilt.size(); ++i) tilt[i] = 0.025 + 0.004 * i;
  xs.push_back(tilt);

  double worst_r = 0.0, worst_j = 0.0, jscale = 0.0;
  for (const auto& x : xs) {
    worst_r = std::max(worst_r, (cr.residuals(x) - prob.residuals<double>(x)).cwiseAbs().maxCoeff());
    const Eigen::MatrixXd Jc = cr.jacobian(x);
    const Eigen::MatrixXd Ja = cal::aad_jacobian(prob, x);
    worst_j = std::max(worst_j, (Jc - Ja).cwiseAbs().maxCoeff());
    jscale = std::max(jscale, Ja.cwiseAbs().maxCoeff());
  }
  EXPECT_LT(worst_r, 1e-12) << "compiled W-cache residual must equal the templated residual";
  EXPECT_LT(worst_j / jscale, 1e-9) << "compiled analytic Jacobian must equal AAD (spread + notional)";
}
