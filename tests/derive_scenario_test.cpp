// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// derive/scenario.hpp (E7 stage 5.2): the library behind the `scenario` verb, driven by a session whose calibration
// returns its seed -- so the move rule, the fork, the one calibration and the valuation are checked against the
// library pieces they are made of (calibration/bundle_state.hpp, portfolio/fx_pairs.hpp), independent of any
// solver. Header-only (swaps_tests), so tools/mutate.py reaches it; the verb end to end stays pinned byte for byte by
// tests/scenario_golden_test.cpp.
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "swaps/derive/scenario.hpp"

namespace cal = swaps::calibration;
namespace crv = swaps::curve;
namespace dv = swaps::derive;
namespace pf = swaps::portfolio;
namespace px = swaps::pricing;

namespace {

// calibrate() keeps its seed, so x_base is whatever the library seeded with; it counts calls and records the regulariser.
class SeedSession {
 public:
  explicit SeedSession(cal::BundleProblem p) : p_(std::move(p)) {}
  const cal::CalibrationResult& calibrate(const Eigen::VectorXd& x0, const cal::RegSpec& reg) {
    x_ = x0;
    ++calls;
    last_lambda = reg.lambda;
    res_.x = x_;
    res_.converged = true;
    res_.iterations = 4;
    res_.status = "cosine too small";
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
pf::MultiCurveBook::Position one_period_swap(double fixed_rate) {
  pf::MultiCurveBook::Position p;
  p.notional = 1e6;
  p.fixed_rate = fixed_rate;
  p.fwd_curve = 0;
  p.disc_curve = 0;
  p.fixed_curve = 0;
  p.float_coupons = {float_coupon(0.0, 1.0)};
  px::FixedCoupon f;
  f.pay = 1.0;
  f.tau = 1.0;
  p.fixed_coupons = {f};
  return p;
}
pf::MultiCurveBook::Position xccy_position() {
  pf::MultiCurveBook::Position p;
  p.kind = pf::MultiCurveBook::Kind::Xccy;
  p.notional = 3e7;
  p.fwd_curve = 0;
  p.disc_curve = 0;
  p.float_coupons = {float_coupon(0.0, 1.0), float_coupon(1.0, 2.0)};
  p.mtm_coupons = p.float_coupons;
  for (auto& c : p.mtm_coupons) c.spread = 0.005;
  p.mtm_fwd_curve = 1;
  p.mtm_disc_curve = 1;
  p.mtm_reset_num = 1;
  p.mtm_reset_den = 0;
  p.fx_spot = 1.10;
  return p;
}
std::vector<cal::BundleCurveSpec> outright_curves(int n) { return std::vector<cal::BundleCurveSpec>(static_cast<std::size_t>(n), flat_curve(0)); }

cal::BundleProblem two_flat_curves() {
  cal::BundleProblem p;
  p.curves = {flat_curve(0), flat_curve(1)};
  p.currency_codes = {"USD", "EUR"};
  return p;
}

}  // namespace

TEST(ResolveScenarioMove, AnExplicitCurveKeyAddsOntoTheParallel) {
  dv::ScenarioMove m;
  m.parallel_bp = 34.0;
  m.shift_curve_bp = {{0, 14.5}};
  m.fx = {{"EUR", "USD", 0.02}};  // FX is resolved per pair by book_fx_moves (fx_pairs_test.cpp), not here
  const dv::ResolvedMove r = dv::resolve_scenario_move(m, outright_curves(3));
  EXPECT_EQ(r.curve_delta, (std::vector<double>{34.0 / 1e4 + 14.5 / 1e4, 34.0 / 1e4, 34.0 / 1e4}))
      << "a key ADDS onto the parallel (SC1, owner decision 2026-09-14), as in scenario_grid and var";
  EXPECT_NE(r.curve_delta[1], 34.0 * 1e-4) << "bp / 1e4, bit for bit: bp * 1e-4 differs at 34 bp";

  const dv::ResolvedMove none = dv::resolve_scenario_move(dv::ScenarioMove{}, outright_curves(2));
  EXPECT_EQ(none.curve_delta, (std::vector<double>{0.0, 0.0}));

  dv::ScenarioMove held;
  held.parallel_bp = 20.0;
  held.shift_curve_bp = {{1, 0.0}};
  EXPECT_EQ(dv::resolve_scenario_move(held, outright_curves(2)).curve_delta, (std::vector<double>{20.0 / 1e4, 20.0 / 1e4}))
      << "an explicit 0 adds nothing (under SC1 it no longer holds its curve still)";

  dv::ScenarioMove out_of_range;
  out_of_range.shift_curve_bp = {{2, 1.0}};
  EXPECT_THROW((void)dv::resolve_scenario_move(out_of_range, outright_curves(2)), std::invalid_argument);
  out_of_range.shift_curve_bp = {{-1, 1.0}};
  EXPECT_THROW((void)dv::resolve_scenario_move(out_of_range, outright_curves(2)), std::invalid_argument);
}

TEST(Scenarios, CalibratesOnceAndValuesEveryMoveAtItsForkOfTheBase) {
  const cal::BundleProblem p = two_flat_curves();
  const pf::MultiCurveBook book{{one_period_swap(0.02), xccy_position()}};
  const Eigen::VectorXd x0 = Eigen::Vector2d(0.03, 0.02);
  dv::ScenarioRequest r;
  r.bundle = p;
  r.x0 = x0;
  r.reg.lambda = 0.25;
  r.sample_times = {0.0, 2.0};
  r.book = book;
  dv::ScenarioMove par, fx, noop;
  par.name = "par";
  par.parallel_bp = 37.5;
  fx.name = "fx";
  fx.fx = {{"EUR", "USD", 0.02}};
  noop.name = "noop";
  r.scenarios = {par, fx, noop};

  SeedSession::calls = 0;
  const dv::ScenarioResult out = dv::scenarios<SeedSession>(r);
  EXPECT_EQ(SeedSession::calls, 1) << "one calibration for every move";
  EXPECT_EQ(SeedSession::last_lambda, 0.25);
  EXPECT_EQ(out.calibration.iterations, 4) << "the base's solve is reported as the session returned it (SC3)";
  EXPECT_EQ(std::string(out.calibration.status), "cosine too small");
  EXPECT_EQ(out.n_curves, 2);
  EXPECT_EQ(out.n_knots, 2);
  EXPECT_TRUE(out.x_base == x0);
  EXPECT_TRUE(out.has_book);
  EXPECT_EQ(out.n_positions, 2);
  EXPECT_EQ(out.base_npv, cal::book_value_at(book, p, x0));
  ASSERT_EQ(out.base_curves.size(), 2u);
  ASSERT_EQ(out.rows.size(), 3u);

  const Eigen::VectorXd shifted = cal::shift_interp_forwards(p, x0, {37.5 / 1e4, 37.5 / 1e4});
  EXPECT_EQ(out.rows[0].npv, cal::book_value_at(book, p, shifted));
  EXPECT_EQ(out.rows[0].npv_delta, out.rows[0].npv - out.base_npv);
  EXPECT_EQ(out.rows[0].curves[1].zero, cal::sample_bundle_curves(p, shifted, {0.0, 2.0})[1].zero);

  pf::MultiCurveBook eurusd_up = book;  // the xccy position is EURUSD: num curve 1 (EUR) over den curve 0 (USD)
  eurusd_up.positions[1].fx_spot = 1.10 * (1.0 * (1.0 + 0.02));
  EXPECT_EQ(out.rows[1].npv, cal::book_value_at(eurusd_up, p, x0)) << "the FX row values the book at its pair's factor";
  EXPECT_NE(out.rows[1].npv, out.base_npv);
  EXPECT_EQ(out.rows[2].npv, out.base_npv) << "no earlier move mutated the base";
  EXPECT_EQ(out.rows[2].curves[0].zero, out.base_curves[0].zero);
  EXPECT_EQ(out.moves[out.rows[1].move].name, "fx");
}

TEST(Scenarios, WithoutABookOrSampleTimesItOnlyForksAndItChecksItsInputs) {
  const cal::BundleProblem p = two_flat_curves();
  dv::ScenarioMove par;
  par.parallel_bp = 37.5;
  dv::ScenarioRequest r;
  r.bundle = p;
  r.scenarios = {par};
  const dv::ScenarioResult out = dv::scenarios<SeedSession>(r);
  EXPECT_FALSE(out.has_book);
  EXPECT_TRUE(out.base_curves.empty());
  ASSERT_EQ(out.rows.size(), 1u);
  EXPECT_TRUE(out.rows[0].curves.empty());
  EXPECT_EQ(out.rows[0].npv, 0.0);
  EXPECT_TRUE(out.x_base == cal::flat_x0(p)) << "no x0 -> the flat seed";

  dv::ScenarioRequest bad_seed = r;
  bad_seed.x0 = Eigen::VectorXd::Zero(3);
  EXPECT_THROW((void)dv::scenarios<SeedSession>(bad_seed), std::invalid_argument);
  EXPECT_THROW((void)dv::scenarios<SeedSession>(dv::ScenarioRequest{}), std::invalid_argument) << "a bundle with no curves";
}

// SC2: a move's FX is resolved per currency pair BEFORE calibrating -- a book with xccy positions under an FX bump needs
// bundle.currency_codes (the pair's orientation), and a move with no FX bump needs none.
TEST(Scenarios, AnFxMoveOnAnXccyBookIsResolvedPerPairBeforeCalibrating) {
  dv::ScenarioRequest r;
  r.bundle = two_flat_curves();
  r.bundle.currency_codes.clear();
  r.x0 = Eigen::Vector2d(0.03, 0.02);
  r.book = pf::MultiCurveBook{{xccy_position()}};
  dv::ScenarioMove fx;
  fx.fx = {{"EUR", "USD", 0.02}};
  r.scenarios = {fx};
  SeedSession::calls = 0;
  EXPECT_THROW((void)dv::scenarios<SeedSession>(r), std::invalid_argument) << "no currency codes: the orientation is unknown";
  EXPECT_EQ(SeedSession::calls, 0) << "refused before calibrating";
  r.scenarios = {dv::ScenarioMove{}};
  EXPECT_NO_THROW((void)dv::scenarios<SeedSession>(r)) << "no FX move: no codes needed";
}
