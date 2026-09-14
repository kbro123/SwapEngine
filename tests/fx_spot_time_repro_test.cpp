// E5 taxonomy: T6 regression (fails on the reverted bug) | T5 properties + value pins (hand / closed-form literals, identities, FD)
// O-X3 REPRODUCTION (2026-09-14). fx_spot is the SPOT-DATE quote (market, QuantLib FxSwapRateHelper, ORE); the engine read it as
// a t = 0 rate. With flat continuously-compounded curves (premise-checked) covered interest parity gives, independent of the
// engine's formula:  F(T) = S * exp((r_den - r_num)(T - t_s)),  FX_0 = S * exp(-(r_den - r_num) t_s).
// 200 bp, T+2: the t = 0 reading is +1.096e-4 relative (1.2 pips); a 1W-pinned zero is off ~44 bp.
// REPRODUCED pre-field on 8770c32 (R1-R4a, R4d, R6 failed by the spot factor; the fixing-value-date pins held).
// R4b/R4c are CONTROLS that PASS TODAY and FAIL a kernel reading the reset ratio at the FIXING date (see their comment).
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "swaps/build/instruments.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/portfolio/portfolio.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "tolerances.hpp"

namespace b = swaps::build;
namespace cal = swaps::calibration;
namespace cv = swaps::curve;
namespace pf = swaps::portfolio;
namespace px = swaps::pricing;
namespace tol = swaps::tol;

namespace {

constexpr double kS = 1.10;
constexpr double kRNum = 0.020;       // curve 0: EUR discounted under USD collateral (FX numerator, pinned by FX forwards)
constexpr double kRDen = 0.040;       // curve 1: USD SOFR (FX denominator)
constexpr double kTs = 2.0 / 365.0;   // T+2 spot date
constexpr int kNum = 0, kDen = 1;
const std::vector<double>& knots() {
  static const std::vector<double> k{9.0 / 365.0, 0.25, 0.5, 1.0, 2.0};
  return k;
}

cal::BundleProblem two_curves() {
  cal::BundleProblem p;
  p.curves.push_back({.base = -1, .currency = 1, .regions = cv::flat_hermite({}, knots())});
  p.curves.push_back({.base = -1, .currency = 0, .regions = cv::flat_hermite({}, knots())});
  return p;
}
Eigen::VectorXd flat_state(const cal::BundleProblem& p) {
  Eigen::VectorXd x(p.n_knots());
  for (int c = 0; c < 2; ++c)
    for (int i = 0; i < p.curves[static_cast<std::size_t>(c)].n_knots(); ++i) x[p.offset(c) + i] = c == kNum ? kRNum : kRDen;
  return x;
}
struct Flat {
  cal::BundleProblem p = two_curves();
  Eigen::VectorXd x = flat_state(p);
  std::vector<std::unique_ptr<cal::CurveHandle<double>>> C;
  Flat() { C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; }); }
  const cal::CurveHandle<double>& operator()(int i) const { return *C[static_cast<std::size_t>(i)]; }
};
// Every closed form is in exp(); this checks the flat curve IS exp(-r t) (1e-14: rounding of the primitive r*t).
void flat_premise(const Flat& w) {
  for (double t : {0.0, 1.0 / 365.0, kTs, 5.0 / 365.0, 9.0 / 365.0, 0.3, 1.0, 2.0}) {
    ASSERT_NEAR(w(kNum).discount(t), std::exp(-kRNum * t), 1e-14) << "premise: flat curve 0 at t=" << t;
    ASSERT_NEAR(w(kDen).discount(t), std::exp(-kRDen * t), 1e-14) << "premise: flat curve 1 at t=" << t;
  }
}
double outright(double T) { return kS * std::exp((kRDen - kRNum) * (T - kTs)); }
const double kT0Gap = std::expm1((kRDen - kRNum) * kTs);  // what the t = 0 reading adds (+1.096e-4)

// One MtM funding period on curve kDen resetting against kNum/kDen, +100 bp spread so its value spread*tau*DF(e) is not 0.
pf::MultiCurveBook::Position mtm_position(double s, double e, double reset) {
  px::FloatCoupon c;
  c.obs.sub_start = {s};
  c.obs.sub_end = {e};
  c.obs.tau_index = e - s;
  c.pay = e;
  c.tau_pay = e - s;
  c.spread = 0.01;
  c.reset_time = reset;
  c.accrual_set = true;
  c.accrual_start = s;
  c.accrual_end = e;
  pf::MultiCurveBook::Position p;
  p.kind = pf::MultiCurveBook::Kind::Xccy;
  p.notional = 1.0;
  p.mtm_coupons = {c};
  p.mtm_fwd_curve = kDen;
  p.mtm_disc_curve = kDen;
  p.mtm_reset_num = kNum;
  p.mtm_reset_den = kDen;
  p.fx_spot = kS;
  return p;
}
// N = value / (the same period value with the notional FIXED at 1 via reset_fx): identical v on both, so N is exact to ULPs.
double resetting_notional(pf::MultiCurveBook::Position p, const Flat& w) {
  const double value = pf::MultiCurveBook::position_value<double>(p, w);
  const double s = p.mtm_coupons.front().accrual_start, e = p.mtm_coupons.front().accrual_end;
  for (auto& c : p.mtm_coupons) c.reset_fx = 1.0;
  const double per_unit = pf::MultiCurveBook::position_value<double>(p, w);
  // premise (1e-11: DF(s) - DF(e) cancels inside v, ~4 ULP / 5e-3)
  EXPECT_NEAR(per_unit, 0.01 * (e - s) * std::exp(-kRDen * e), 1e-11) << "premise: period value";
  return value / per_unit;
}

}  // namespace

// ---- fail before the change -------------------------------------------------------------------------------------------
TEST(FxSpotTimeRepro, R2TheForwardForDeliveryOnTheSpotDateIsTheSpotQuote) {
  const Flat w;
  flat_premise(w);
  cal::Instrument ins = b::fx_forward(kNum, kDen, kS, /*fx_time=*/kTs, /*market=*/kS);
  ins.fx_spot_time = kTs;
  const double F = cal::instrument_model_quote<double>(ins, w);
  EXPECT_NEAR(F / kS - 1.0, 0.0, tol::literal) << "the t = 0 reading gives " << kT0Gap;
  EXPECT_NEAR(cal::instrument_residual<double>(ins, w), 0.0, tol::literal / kTs);
}

TEST(FxSpotTimeRepro, R1EveryOutrightRollsFromTheSpotDate) {
  const Flat w;
  flat_premise(w);
  for (double T : {1.0 / 365.0, kTs + 7.0 / 365.0, 0.25, 1.0, 2.0}) {  // tom (before spot), spot+1W, 3M, 1Y, 2Y
    SCOPED_TRACE(T * 365.0);
    cal::Instrument ins = b::fx_forward(kNum, kDen, kS, T, outright(T));
    ins.fx_spot_time = kTs;
    const double F = cal::instrument_model_quote<double>(ins, w);
    std::cout << "  [fx-spot] T=" << T * 365 << "d implied zero error " << std::log(F / outright(T)) / T * 1e4 << " bp\n";
    EXPECT_NEAR(F / outright(T) - 1.0, 0.0, tol::literal);
  }
}

// TN = spot vs tomorrow. One day of covered interest parity anchored at F(t_s) = S: TN points = S - F(tom)
// = S * (1 - exp(-(r_den - r_num)/365)). The t = 0 reading FLIPS ITS SIGN. (ON needs F(0): fx_time > 0 is validated, so ON is
// not an FxForward row; it is FX_0, pinned through the MtM notional below.)
TEST(FxSpotTimeRepro, R3TomNextPointsAreOneDayOfCarryBackFromTheSpotQuote) {
  const Flat w;
  flat_premise(w);
  cal::Instrument tom = b::fx_forward(kNum, kDen, kS, 1.0 / 365.0, kS);
  tom.fx_spot_time = kTs;
  EXPECT_NO_THROW(cal::validate_instrument(tom, "TN")) << "a forward delivering BEFORE spot is valid";
  const double tn = kS - cal::instrument_model_quote<double>(tom, w);
  EXPECT_NEAR(tn, -kS * std::expm1(-(kRDen - kRNum) / 365.0), tol::literal * kS) << "TN points";
}

// A spot-starting MtM swap exchanges its initial notional at the traded spot: N_0 = S when the notional is read for value on the
// period start (= the spot date).
TEST(FxSpotTimeRepro, R4aTheInitialResettingNotionalIsTheSpotQuote) {
  const Flat w;
  flat_premise(w);
  pf::MultiCurveBook::Position p = mtm_position(kTs, kTs + 0.5, /*reset=*/kTs);
  p.fx_spot_time = kTs;
  EXPECT_NEAR(resetting_notional(p, w) / kS - 1.0, 0.0, tol::literal) << "the t = 0 reading gives " << kT0Gap;
}

// A later period whose FX fixing is 2 business days before its start ACROSS A WEEKEND (4 calendar days): the fixing is a spot rate
// for value on the period start s1, so N_1 = F(s1) = S exp((r_den - r_num)(s1 - t_s)). The fixing DATE never moves the forward:
// reset_time stays unset (= the period start); when a fixing makes the coupon seasoned is the separate fixing time (owner
// decision 2026-09-14). Pre-field the t = 0 reading missed this by the spot factor.
TEST(FxSpotTimeRepro, R4dAWeekendFixingIsStillTheForwardForThePeriodStart) {
  const Flat w;
  flat_premise(w);
  const double s1 = kTs + 0.25;
  pf::MultiCurveBook::Position p = mtm_position(s1, s1 + 0.25, /*reset=*/-1.0);  // fixing s1 - 4d; forward at s1
  p.fx_spot_time = kTs;
  EXPECT_NEAR(resetting_notional(p, w) / outright(s1) - 1.0, 0.0, tol::literal);
}

// Bootstrap: SOFR pinned by simple forward rows, EUR-in-USD by spot-date FX outrights at the knots. The true (flat) curve must be
// the zero-residual solution and LM must return it on BOTH engines (use_aad = true -> hybrid/compiled FX rows; false -> templated).
TEST(FxSpotTimeRepro, R6SpotDateFxForwardsBootstrapTheTrueForeignCurve) {
  cal::BundleProblem p = two_curves();
  const Eigen::VectorXd x_true = flat_state(p);
  const int n = static_cast<int>(knots().size());
  double a = 0.0;
  for (double T : knots()) {
    p.instruments.push_back(b::rate_instrument(kDen, b::plain_rate_obs(a, T), std::expm1(kRDen * (T - a)) / (T - a)));
    a = T;
  }
  for (double T : knots()) {
    cal::Instrument f = b::fx_forward(kNum, kDen, kS, T, outright(T));
    f.fx_spot_time = kTs;
    p.instruments.push_back(f);
  }
  ASSERT_EQ(p.n_residuals(), p.n_knots()) << "premise: square";
  const Eigen::VectorXd r = p.residuals<double>(x_true);
  for (int i = 0; i < n; ++i) ASSERT_NEAR(r[i], 0.0, 1e-13) << "premise: SOFR row " << i;
  // 1e-11: flat-curve DF rounding (~1e-14 relative) divided by the shortest T (9/365)
  for (int i = 0; i < n; ++i) EXPECT_NEAR(r[n + i], 0.0, 1e-11) << "FX row " << i << " at the true curve";
  const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(p.n_knots(), 0.03);
  for (bool aad : {true, false}) {
    SCOPED_TRACE(aad ? "hybrid" : "templated");
    const cal::CalibrationResult sol = cal::calibrate(p, x0, aad);
    double worst = 0.0;
    for (int i = 0; i < n; ++i) worst = std::max(worst, std::abs(sol.x[p.offset(kNum) + i] - kRNum));
    std::cout << "  [fx-spot] bootstrap worst EUR-in-USD knot error " << worst * 1e4 << " bp\n";
    // 1e-8 = 1e-4 bp: LM stop on a square zero-residual system (FX block conditioned ~1/T at 9/365)
    EXPECT_LE(worst, 1e-8);
    EXPECT_LE((sol.x - x_true).cwiseAbs().maxCoeff(), 1e-8);
  }
}

// ---- the fixing's value date ---------------------------------------------------------------------------------------------
// CARR fixes each period's FX 2 BD before its start, for value ON the start. For a spot-starting swap the first fixing is on the
// TRADE date, yet it is a spot rate -> N_0 = S; a later period's fixing is the forward for its start. The forward is therefore read
// at the period start (reset_time unset), never at the fixing date -- reading it there with the spot roll-back double-counts the
// lag (S*(1 - 1.1e-4) for R4b, 2 days of carry for R4c). (The drafts encoded the fixing date in reset_time; the owner's design
// keeps a separate fixing time for seasoning, so these pins read the forward at the start.)
TEST(FxSpotTimeRepro, R4bATradeDateFixingStillSetsTheInitialNotionalToTheSpotQuote) {
  const Flat w;
  flat_premise(w);
  pf::MultiCurveBook::Position p = mtm_position(kTs, kTs + 0.5, /*reset=*/-1.0);  // fixing on the trade date; forward at spot
  p.fx_spot_time = kTs;
  EXPECT_NEAR(resetting_notional(p, w) / kS - 1.0, 0.0, tol::literal);
}
TEST(FxSpotTimeRepro, R4cATwoDayFixingIsTheForwardForThePeriodStart) {
  const Flat w;
  flat_premise(w);
  const double s1 = kTs + 0.25;
  pf::MultiCurveBook::Position p = mtm_position(s1, s1 + 0.25, /*reset=*/-1.0);  // fixing s1 - 2d; forward at s1
  p.fx_spot_time = kTs;
  EXPECT_NEAR(resetting_notional(p, w) / outright(s1) - 1.0, 0.0, tol::literal);
}
