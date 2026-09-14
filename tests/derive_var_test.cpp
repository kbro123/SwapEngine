// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// derive/var.hpp (E7): the library behind the `var` verb. The statistics are checked against closed forms (type-7
// quantile, tail mean, sample stdev, loss = -P&L). The reval sweep is driven by a session whose calibration lands a fixed
// 1e-3 off its seed (so revaluing at the SEED instead of the calibrated state cannot survive) and reports a
// non-converged status (so the result carries the session's own outcome, and a failed solve is reported, not
// refused); every P&L is compared bit for bit with the library pieces it is made of (bundle_state.hpp,
// fx_pairs.hpp) under the ADD rule (SC1). Header-only (swaps_tests), so tools/mutate.py reaches it; the verb end
// to end stays pinned byte for byte by tests/scenario_golden_test.cpp.
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "swaps/derive/var.hpp"

namespace cal = swaps::calibration;
namespace crv = swaps::curve;
namespace dv = swaps::derive;
namespace pf = swaps::portfolio;
namespace px = swaps::pricing;

namespace {

constexpr double kCalibrationShift = 1e-3;

class SeedSession {
 public:
  explicit SeedSession(cal::BundleProblem p) : p_(std::move(p)) {}
  const cal::CalibrationResult& calibrate(const Eigen::VectorXd& x0, const cal::RegSpec& reg) {
    x_ = (x0.array() + kCalibrationShift).matrix();
    ++calls;
    last_lambda = reg.lambda;
    res_.x = x_;
    res_.iterations = 7;
    res_.info = 5;
    res_.rms_residual = 2.5e-9;
    res_.stationarity = 1.25e-7;
    res_.rank_deficiency = 1;
    res_.converged = false;
    res_.status = "too many function evaluations";
    return res_;
  }
  const cal::BundleProblem& problem() const { return p_; }
  const Eigen::VectorXd& x() const { return x_; }
  Eigen::MatrixXd jacobian(const cal::RegSpec&) const { return Eigen::MatrixXd::Zero(p_.n_residuals(), p_.n_knots()); }
  Eigen::MatrixXd risk_operator(const cal::RegSpec&) const { return Eigen::MatrixXd::Zero(p_.n_knots(), p_.n_residuals()); }

  inline static int calls = 0;
  inline static double last_lambda = 0.0;

 private:
  cal::BundleProblem p_;
  Eigen::VectorXd x_;
  cal::CalibrationResult res_;
};

cal::BundleCurveSpec flat_curve(int currency) {
  cal::BundleCurveSpec s;
  s.currency = currency;
  crv::CurveModule m;
  m.scheme = crv::Scheme::Flat;
  m.knots = {1.0};
  s.regions = {m};
  return s;
}
px::FloatCoupon float_coupon(double a, double b) {
  px::FloatCoupon c;
  c.obs.sub_start = {a};
  c.obs.sub_end = {b};
  c.obs.tau_index = b - a;
  c.pay = b;
  c.tau_pay = b - a;
  return c;
}
// A swap forecasting on curve 1, discounted on 0, and an xccy position whose MtM leg is on curve 2.
pf::MultiCurveBook book_with_xccy() {
  pf::MultiCurveBook::Position swap;
  swap.notional = 1e6;
  swap.fixed_rate = 0.02;
  swap.fwd_curve = 1;
  swap.disc_curve = 0;
  swap.fixed_curve = 0;
  swap.float_coupons = {float_coupon(0.0, 1.0)};
  px::FixedCoupon f;
  f.pay = 1.0;
  f.tau = 1.0;
  swap.fixed_coupons = {f};

  pf::MultiCurveBook::Position xccy;
  xccy.kind = pf::MultiCurveBook::Kind::Xccy;
  xccy.notional = 3e7;
  xccy.fwd_curve = 0;
  xccy.disc_curve = 0;
  xccy.float_coupons = {float_coupon(0.0, 1.0), float_coupon(1.0, 2.0)};
  xccy.mtm_coupons = xccy.float_coupons;
  for (auto& c : xccy.mtm_coupons) c.spread = 0.005;
  xccy.mtm_fwd_curve = 2;
  xccy.mtm_disc_curve = 2;
  xccy.mtm_reset_num = 2;
  xccy.mtm_reset_den = 0;
  xccy.fx_spot = 1.10;
  return pf::MultiCurveBook{{swap, xccy}};
}
dv::ScenarioMove parallel(double bp) {
  dv::ScenarioMove m;
  m.parallel_bp = bp;
  return m;
}
dv::VarRevalRequest three_curve_reval() {
  dv::VarRevalRequest r;
  r.bundle.curves = {flat_curve(0), flat_curve(0), flat_curve(1)};
  r.bundle.currency_codes = {"USD", "EUR"};
  r.x0 = Eigen::Vector3d(0.030, 0.033, 0.020);
  r.book = book_with_xccy();
  return r;
}

}  // namespace

// ---- the statistics ---------------------------------------------------------------------------------------------

TEST(VarStatistics, Type7QuantileAndTailMeanHaveTheirClosedForms) {
  std::vector<double> uniform(1000);
  for (int i = 0; i < 1000; ++i) uniform[static_cast<std::size_t>(i)] = i;
  // q = 0.95 -> p = 0.05 -> position 0.05 * 999 = 49.95 -> 49.95; ES = mean{0..49} = 24.5 (m = 50, exact).
  const dv::VarQuantile q95 = dv::var_es_at(uniform, 0.95);
  EXPECT_NEAR(q95.var_pnl, 49.95, 1e-9) << "the order statistics are interpolated, not floored";
  EXPECT_EQ(q95.es_pnl, 24.5);
  EXPECT_EQ(q95.q, 0.95);

  const std::vector<double> seven{-5.0, -3.0, -1.0, 0.0, 2.0, 4.0, 6.0};
  // q = 0.99: round(0.01 * 7) = 0 worst points -> clamped to 1, so ES is the single worst P&L.
  EXPECT_EQ(dv::var_es_at(seven, 0.99).es_pnl, -5.0) << "m = round((1-q) N) is clamped to at least one point";
  // q below 2^-53: 1 - q == 1, the position is N - 1 exactly -> the last point (and ES the mean of all N).
  const dv::VarQuantile tiny = dv::var_es_at(seven, 1e-17);
  EXPECT_EQ(tiny.var_pnl, 6.0);
  EXPECT_NEAR(tiny.es_pnl, 3.0 / 7.0, 1e-15);

  const dv::VarQuantile single = dv::var_es_at({7.0}, 0.9);
  EXPECT_EQ(single.var_pnl, 7.0);
  EXPECT_EQ(single.es_pnl, 7.0);
}

TEST(PnlDistribution, SortsSummarisesAndReportsLossesAsTheNegatedPnl) {
  // mean 0.75; deviations 1.25, -3.75, 4.25, -1.75 -> squares sum 36.75 -> / (4 - 1) = 12.25 -> stdev 3.5 (all exact).
  const dv::PnlDistribution d = dv::pnl_distribution({2.0, -3.0, 5.0, -1.0}, {0.5, 0.75});
  EXPECT_EQ(d.n, 4);
  EXPECT_EQ(d.mean_pnl, 0.75);
  EXPECT_EQ(d.stdev_pnl, 3.5) << "the SAMPLE standard deviation (n - 1)";
  EXPECT_EQ(d.pnl_sorted, (std::vector<double>{-3.0, -1.0, 2.0, 5.0}));
  ASSERT_EQ(d.quantiles.size(), 2u);
  // q = 0.5: position 1.5 -> (-1 + 2) / 2 = 0.5; m = 2 -> ES P&L (-3 - 1) / 2 = -2.
  EXPECT_EQ(d.quantiles[0].q, 0.5);
  EXPECT_EQ(d.quantiles[0].var_pnl, 0.5);
  EXPECT_EQ(d.quantiles[0].es_pnl, -2.0);
  EXPECT_EQ(d.quantiles[0].var, -0.5) << "var is the LOSS: -var_pnl";
  EXPECT_EQ(d.quantiles[0].es, 2.0) << "es is the LOSS: -es_pnl";
  // q = 0.75: p = 0.25, position 0.75 -> -3 * 0.25 + -1 * 0.75 = -1.5; m = 1 -> -3. Request order kept.
  EXPECT_EQ(d.quantiles[1].q, 0.75);
  EXPECT_EQ(d.quantiles[1].var_pnl, -1.5);
  EXPECT_EQ(d.quantiles[1].es_pnl, -3.0);

  EXPECT_EQ(dv::pnl_distribution({4.0}, {0.95}).stdev_pnl, 0.0) << "one point has no spread (no 0 / 0)";

  EXPECT_THROW((void)dv::pnl_distribution({}, {0.95}), std::invalid_argument);
  EXPECT_THROW((void)dv::pnl_distribution({1.0}, {}), std::invalid_argument) << "an explicit empty set of quantiles";
  EXPECT_THROW((void)dv::pnl_distribution({1.0}, {0.0}), std::invalid_argument);
  EXPECT_THROW((void)dv::pnl_distribution({1.0}, {1.0}), std::invalid_argument);
  EXPECT_THROW((void)dv::pnl_distribution({1.0}, {std::numeric_limits<double>::quiet_NaN()}), std::invalid_argument);
}

// ---- the verb's computation -------------------------------------------------------------------------------------

TEST(Var, SuppliedModeReducesTheSeriesAndNeverCalibrates) {
  dv::VarRequest r;
  r.pnl = std::vector<double>{2.0, -3.0, 5.0, -1.0};
  r.quantiles = {0.5, 0.75};
  SeedSession::calls = 0;
  const dv::VarResult out = dv::var<SeedSession>(r);
  EXPECT_EQ(SeedSession::calls, 0);
  EXPECT_FALSE(out.reval.has_value());
  const dv::PnlDistribution want = dv::pnl_distribution(*r.pnl, r.quantiles);
  EXPECT_EQ(out.distribution.pnl_sorted, want.pnl_sorted);
  EXPECT_EQ(out.distribution.stdev_pnl, want.stdev_pnl);
  EXPECT_EQ(out.distribution.quantiles[1].es, want.quantiles[1].es);

  dv::VarRequest defaults;
  defaults.pnl = std::vector<double>{1.0, 2.0};
  const dv::VarResult d = dv::var<SeedSession>(defaults);
  ASSERT_EQ(d.distribution.quantiles.size(), 2u);
  EXPECT_EQ(d.distribution.quantiles[0].q, 0.95) << "the struct's own default quantiles";
  EXPECT_EQ(d.distribution.quantiles[1].q, 0.99);
}

TEST(Var, RevalPricesEveryMoveAtTheCalibratedStateThroughTheCompiledBookUnderTheAddRule) {
  dv::VarRequest r;
  r.quantiles = {0.9};
  r.reval = three_curve_reval();
  r.reval->reg.lambda = 0.25;
  dv::ScenarioMove keyed = parallel(34.0);
  keyed.shift_curve_bp = {{0, 14.5}};
  dv::ScenarioMove fx;
  fx.fx = {{"EUR", "USD", 0.02}, {"GBP", "USD", -0.01}};
  dv::ScenarioMove both = parallel(-31.75);
  both.fx = {{"EUR", "USD", -0.05}};
  r.reval->scenarios = {parallel(37.5), keyed, fx, both, dv::ScenarioMove{}};
  const cal::BundleProblem P = r.reval->bundle;
  const pf::MultiCurveBook book = r.reval->book;
  const Eigen::VectorXd x = (r.reval->x0->array() + kCalibrationShift).matrix();  // the CALIBRATED state

  SeedSession::calls = 0;
  const dv::VarResult out = dv::var<SeedSession>(r);
  EXPECT_EQ(SeedSession::calls, 1) << "one calibration for every move";
  EXPECT_EQ(SeedSession::last_lambda, 0.25);
  ASSERT_TRUE(out.reval.has_value());
  const dv::VarRevalResult& rv = *out.reval;
  EXPECT_EQ(rv.n_positions, 2);
  EXPECT_GE(rv.reval_us, 0.0);

  const auto at_fx = [&P, &book](double factor) {  // the EURUSD spot moved by hand, freshly compiled
    pf::MultiCurveBook moved = book;
    moved.positions[1].fx_spot = 1.10 * factor;
    return pf::CompiledMultiCurveBook(P.curves, moved);
  };
  EXPECT_EQ(rv.base_npv, pf::CompiledMultiCurveBook(P.curves, book).npv(x)) << "the base is the calibrated state, not the seed";
  const auto pnl_at = [&](const std::vector<double>& delta, double factor) {
    return at_fx(factor).npv(cal::shift_interp_forwards(P, x, delta)) - rv.base_npv;
  };
  ASSERT_EQ(rv.pnl.size(), 5u);
  EXPECT_EQ(rv.pnl[0], pnl_at({37.5 / 1e4, 37.5 / 1e4, 37.5 / 1e4}, 1.0));
  EXPECT_EQ(rv.pnl[1], pnl_at({0.0 + 34.0 / 1e4 + 14.5 / 1e4, 34.0 / 1e4, 34.0 / 1e4}, 1.0))
      << "an explicit curve key ADDS to the parallel (SC1)";
  EXPECT_EQ(rv.pnl[2], pnl_at({0.0, 0.0, 0.0}, 1.0 * (1.0 + 0.02)))
      << "per pair (SC2): the EURUSD position takes the EURUSD bump alone; the GBPUSD bump reaches no position";
  EXPECT_NE(rv.pnl[2], 0.0);
  EXPECT_EQ(rv.pnl[3], pnl_at({-31.75 / 1e4, -31.75 / 1e4, -31.75 / 1e4}, 1.0 * (1.0 + -0.05)));
  EXPECT_EQ(rv.pnl[4], 0.0) << "the no-op move is the base: no earlier move mutated it";

  std::vector<double> sorted = rv.pnl;
  std::sort(sorted.begin(), sorted.end());
  EXPECT_EQ(out.distribution.pnl_sorted, sorted) << "the reval P&L (request order) is what the distribution reduces";
  EXPECT_EQ(out.distribution.n, 5);

  // The base's solve is reported as the session returned it -- a failed solve is not refused (SC3 emits it).
  EXPECT_EQ(rv.calibration.iterations, 7);
  EXPECT_FALSE(rv.calibration.converged);
  EXPECT_EQ(std::string(rv.calibration.status), "too many function evaluations");
  EXPECT_EQ(rv.calibration.rank_deficiency, 1);
}

TEST(Var, WithoutX0TheFlatSeedIsCalibrated) {
  dv::VarRequest r;
  r.reval = three_curve_reval();
  r.reval->x0.reset();
  r.reval->scenarios = {parallel(10.0)};
  const cal::BundleProblem P = r.reval->bundle;
  const dv::VarResult out = dv::var<SeedSession>(r);
  const Eigen::VectorXd x = (cal::flat_x0(P).array() + kCalibrationShift).matrix();
  EXPECT_EQ(out.reval->base_npv, pf::CompiledMultiCurveBook(P.curves, r.reval->book).npv(x));
}

TEST(Var, ChecksItsInputsBeforeCalibrating) {
  const auto refused = [](dv::VarRequest r, const char* why) {
    SeedSession::calls = 0;
    EXPECT_THROW((void)dv::var<SeedSession>(std::move(r)), std::invalid_argument) << why;
    EXPECT_EQ(SeedSession::calls, 0) << why << ": refused before calibrating";
  };
  dv::VarRequest ok;
  ok.reval = three_curve_reval();
  ok.reval->scenarios = {parallel(1.0)};
  EXPECT_NO_THROW((void)dv::var<SeedSession>(ok)) << "the control";

  refused(dv::VarRequest{}, "neither a P&L series nor a reval request");
  dv::VarRequest both = ok;
  both.pnl = std::vector<double>{1.0};
  refused(both, "both modes at once");
  dv::VarRequest no_curves = ok;
  no_curves.reval->bundle = cal::BundleProblem{};
  refused(no_curves, "a bundle with no curves");
  dv::VarRequest no_moves = ok;
  no_moves.reval->scenarios.clear();
  refused(no_moves, "no market moves");
  dv::VarRequest bad_role = ok;
  dv::ScenarioMove past = parallel(1.0);
  past.shift_curve_bp = {{3, 1.0}};
  bad_role.reval->scenarios = {parallel(1.0), past};
  refused(bad_role, "a shift_curve role past the bundle, in the LAST move");
  dv::VarRequest bad_q = ok;
  bad_q.quantiles = {0.95, 1.5};
  refused(bad_q, "a quantile outside (0, 1)");
  dv::VarRequest no_q = ok;
  no_q.quantiles.clear();
  refused(no_q, "an explicit empty set of quantiles");
  dv::VarRequest no_codes = ok;
  no_codes.reval->bundle.currency_codes.clear();
  dv::ScenarioMove fx_move;
  fx_move.fx = {{"EUR", "USD", 0.01}};
  no_codes.reval->scenarios = {fx_move};
  refused(no_codes, "an FX move on an xccy book with no currency codes");
  dv::VarRequest bad_seed = ok;
  bad_seed.reval->x0 = Eigen::VectorXd::Zero(2);
  refused(bad_seed, "an x0 of the wrong length");
}
