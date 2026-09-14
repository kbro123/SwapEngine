// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// calibration::pnl_report (E7 stage 6.7): the library behind the `pnl` verb, driven by a session whose calibration
// moves its seed by a fixed amount (so a calibrated state is never an explicit x0 / x1) and whose ladder is a fixed
// vector -- so the wiring (which state the decomposition reprices at, how
// dq is formed, the checks) is compared bit for bit with a direct calibration::pnl_explain call. Header-only
// (swaps_tests), so tools/mutate.py reaches it; the verb end to end is checked against QuantLib by
// tests/pnl_reference_test.cpp.
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "swaps/calibration/pnl_explain.hpp"

namespace cal = swaps::calibration;
namespace crv = swaps::curve;
namespace pf = swaps::portfolio;
namespace px = swaps::pricing;

namespace {

struct FakeRisk {
  Eigen::VectorXd ladder;
};

// "Calibration" lands a fixed 1e-3 off its seed. With a session that returned its seed, an explicit x0 / x1 and the
// calibrated state were the same vector, so dropping the decomposition overrides survived mutation (2026-09-14).
constexpr double kCalibrationShift = 1e-3;

class SeedSession {
 public:
  explicit SeedSession(cal::BundleProblem p) : p_(std::move(p)) {}
  const cal::CalibrationResult& calibrate(const Eigen::VectorXd& x0, const cal::RegSpec&) {
    x_ = (x0.array() + kCalibrationShift).matrix();
    res_.x = x_;
    res_.converged = true;
    return res_;
  }
  const cal::BundleProblem& problem() const { return p_; }
  const Eigen::VectorXd& x() const { return x_; }
  Eigen::MatrixXd jacobian(const cal::RegSpec&) const { return Eigen::MatrixXd::Zero(p_.n_residuals(), p_.n_knots()); }
  Eigen::MatrixXd risk_operator(const cal::RegSpec&) const { return Eigen::MatrixXd::Zero(p_.n_knots(), p_.n_residuals()); }
  FakeRisk price_portfolio_risk(const pf::MultiCurveBook&, const cal::RegSpec&) const {
    return {Eigen::VectorXd::LinSpaced(p_.n_residuals(), 100.0, 200.0)};
  }

 private:
  cal::BundleProblem p_;
  Eigen::VectorXd x_;
  cal::CalibrationResult res_;
};

cal::Instrument rate(double a, double b, double market) {
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::Rate;
  ins.forecast = 0;
  ins.obs.sub_start = {a};
  ins.obs.sub_end = {b};
  ins.obs.tau_index = b - a;
  ins.market = market;
  return ins;
}

// One Hermite curve (knots 1, 3), two Rate quotes: square.
cal::BundleProblem bundle(double m1, double m2) {
  cal::BundleProblem p;
  cal::BundleCurveSpec c;
  c.regions = crv::flat_hermite({}, {1.0, 3.0});
  p.curves = {c};
  p.instruments = {rate(0.0, 1.0, m1), rate(1.0, 3.0, m2)};
  return p;
}

pf::MultiCurveBook book() {
  pf::MultiCurveBook::Position s;
  s.notional = 1e6;
  s.fixed_rate = 0.03;
  s.fwd_curve = 0;
  s.disc_curve = 0;
  s.fixed_curve = 0;
  for (double t : {1.0, 2.0, 3.0}) {
    px::FloatCoupon c;
    c.obs.sub_start = {t - 1.0};
    c.obs.sub_end = {t};
    c.obs.tau_index = 1.0;
    c.pay = t;
    c.tau_pay = 1.0;
    s.float_coupons.push_back(c);
    px::FixedCoupon f;
    f.pay = t;
    f.tau = 1.0;
    s.fixed_coupons.push_back(f);
  }
  return pf::MultiCurveBook{{s}};
}

}  // namespace

TEST(PnlReport, ReprisesAtTheOverridesAndDifferencesTheMarkets) {
  cal::PnlRequest r;
  r.bundle0 = bundle(0.030, 0.032);
  r.bundle1 = bundle(0.031, 0.0345);
  r.book = book();
  r.dt_years = 0.5;
  r.x0 = Eigen::Vector2d(0.029, 0.031);
  r.x1 = Eigen::Vector2d(0.030, 0.033);
  const cal::BundleProblem P0 = r.bundle0;
  const cal::PnlReport out = cal::pnl_report<SeedSession>(r);

  const Eigen::VectorXd ladder = Eigen::VectorXd::LinSpaced(2, 100.0, 200.0);
  const Eigen::VectorXd dq = Eigen::Vector2d(0.031 - 0.030, 0.0345 - 0.032);
  EXPECT_TRUE(out.dq == dq) << "dq = q1 - q0";
  const cal::PnlExplain want = cal::pnl_explain(P0, r.book, *r.x0, *r.x1, 0.5, ladder, dq);
  EXPECT_EQ(out.explain.total, want.total);
  EXPECT_EQ(out.explain.carry, want.carry);
  EXPECT_EQ(out.explain.roll, want.roll);
  EXPECT_EQ(out.explain.market, want.market);
  EXPECT_EQ(out.explain.residual, want.residual);
  EXPECT_EQ(out.explain.npv_t1, want.npv_t1);
  EXPECT_EQ(out.dt_years, 0.5);
  EXPECT_EQ(out.n, 1);
  EXPECT_NE(out.explain.npv_t1, out.explain.npv_t0) << "the fixture moves the market";
}

TEST(PnlReport, WithoutBundle1ItIsPureCarryAndRollAndTheCalibratedStatesAreUsed) {
  cal::PnlRequest r;
  r.bundle0 = bundle(0.030, 0.032);
  r.book = book();
  r.dt_years = 1.0;
  const cal::BundleProblem P0 = r.bundle0;
  const cal::PnlReport out = cal::pnl_report<SeedSession>(r);
  EXPECT_TRUE(out.dq == Eigen::VectorXd::Zero(2));
  const Eigen::VectorXd x = (cal::flat_x0(P0).array() + kCalibrationShift).matrix();  // calibrated from the flat seed
  const cal::PnlExplain want =
      cal::pnl_explain(P0, r.book, x, x, 1.0, Eigen::VectorXd::LinSpaced(2, 100.0, 200.0), Eigen::VectorXd::Zero(2));
  EXPECT_EQ(out.explain.total, want.total);
  EXPECT_EQ(out.explain.market, 0.0);
}

TEST(PnlReport, AnExplicitX1OverridesTheHorizonStateEvenWithoutBundle1) {
  cal::PnlRequest r;
  r.bundle0 = bundle(0.030, 0.032);
  r.book = book();
  r.dt_years = 0.5;
  r.x1 = Eigen::Vector2d(0.0335, 0.0360);
  const cal::BundleProblem P0 = r.bundle0;
  const cal::PnlReport out = cal::pnl_report<SeedSession>(r);
  const Eigen::VectorXd x0 = (cal::flat_x0(P0).array() + kCalibrationShift).matrix();
  const cal::PnlExplain want = cal::pnl_explain(P0, r.book, x0, *r.x1, 0.5, Eigen::VectorXd::LinSpaced(2, 100.0, 200.0),
                                                Eigen::VectorXd::Zero(2));
  EXPECT_EQ(out.explain.npv_t1, want.npv_t1) << "the horizon state is the explicit x1, not the calibrated x0";
  EXPECT_EQ(out.explain.total, want.total);
  EXPECT_TRUE(out.dq == Eigen::VectorXd::Zero(2)) << "no bundle1: no market move";
}

TEST(PnlReport, ChecksItsInputs) {
  cal::PnlRequest ok;
  ok.bundle0 = bundle(0.030, 0.032);
  ok.book = book();

  cal::PnlRequest no_curves = ok;
  no_curves.bundle0 = cal::BundleProblem{};
  EXPECT_THROW((void)cal::pnl_report<SeedSession>(no_curves), std::invalid_argument);

  cal::PnlRequest bad_x0 = ok;
  bad_x0.x0 = Eigen::VectorXd::Zero(3);
  EXPECT_THROW((void)cal::pnl_report<SeedSession>(bad_x0), std::invalid_argument) << "was silently the flat seed";

  cal::PnlRequest bad_x1 = ok;
  bad_x1.x1 = Eigen::VectorXd::Zero(3);
  EXPECT_THROW((void)cal::pnl_report<SeedSession>(bad_x1), std::invalid_argument)
      << "without bundle1 a wrong-length x1 would reach the decomposition";

  cal::PnlRequest knots = ok;
  knots.bundle1 = ok.bundle0;
  knots.bundle1->curves[0].regions = crv::flat_hermite({}, {1.0, 2.0, 3.0});
  EXPECT_THROW((void)cal::pnl_report<SeedSession>(knots), std::invalid_argument);

  cal::PnlRequest rows = ok;
  rows.bundle1 = ok.bundle0;
  rows.bundle1->instruments.push_back(rate(0.0, 3.0, 0.03));
  EXPECT_THROW((void)cal::pnl_report<SeedSession>(rows), std::invalid_argument);
}

// PN1 (reproduced 2026-09-14): bundle1 was checked by COUNTS only (n_knots, n_residuals). pnl_explain reads x1 on
// bundle0's knot grid and forms dq = q1 - q0 row by row, so a bundle1 whose knots sit at other times, or whose rows are
// the same instruments in another order, was accepted and silently misread. Only the quotes may differ.
TEST(PnlReport, RefusesABundle1OfADifferentStructure) {
  cal::PnlRequest ok;
  ok.bundle0 = bundle(0.030, 0.032);
  ok.book = book();
  ok.dt_years = 0.5;

  cal::PnlRequest requoted = ok;
  requoted.bundle1 = bundle(0.031, 0.0345);
  EXPECT_NO_THROW((void)cal::pnl_report<SeedSession>(requoted)) << "the control: a re-quote is the use case";

  cal::PnlRequest moved_knots = ok;
  moved_knots.bundle1 = bundle(0.031, 0.0345);
  moved_knots.bundle1->curves[0].regions = crv::flat_hermite({}, {1.0, 5.0});  // same count, other times
  EXPECT_THROW((void)cal::pnl_report<SeedSession>(moved_knots), std::invalid_argument)
      << "x1 would be read on bundle0's knots {1, 3}";

  cal::PnlRequest reordered = ok;
  reordered.bundle1 = bundle(0.031, 0.0345);
  std::swap(reordered.bundle1->instruments[0], reordered.bundle1->instruments[1]);  // same rows, other order
  EXPECT_THROW((void)cal::pnl_report<SeedSession>(reordered), std::invalid_argument)
      << "dq would difference the 1y quote against the 1y-3y quote";
}
