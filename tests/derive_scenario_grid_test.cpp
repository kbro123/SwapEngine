// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// derive/scenario_grid.hpp (E7 stage 5.3): the library behind the `scenario_grid` verb, driven by a session whose
// calibration returns its seed -- so the axis rule, the checks, the one calibration and every cell are compared with
// the library pieces they are made of (calibration/bundle_state.hpp, portfolio/xccy_fx_scaled.hpp). Header-only
// (swaps_tests), so tools/mutate.py reaches it; the verb end to end stays pinned byte for byte by
// tests/scenario_golden_test.cpp.
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "swaps/derive/scenario_grid.hpp"

namespace cal = swaps::calibration;
namespace crv = swaps::curve;
namespace dv = swaps::derive;
namespace pf = swaps::portfolio;
namespace px = swaps::pricing;

namespace {

class SeedSession {
 public:
  explicit SeedSession(cal::BundleProblem p) : p_(std::move(p)) {}
  const cal::CalibrationResult& calibrate(const Eigen::VectorXd& x0, const cal::RegSpec&) {
    x_ = x0;
    ++calls;
    res_.x = x_;
    res_.converged = true;
    return res_;
  }
  const cal::BundleProblem& problem() const { return p_; }
  const Eigen::VectorXd& x() const { return x_; }
  Eigen::MatrixXd jacobian(const cal::RegSpec&) const { return Eigen::MatrixXd::Zero(p_.n_residuals(), p_.n_knots()); }
  Eigen::MatrixXd risk_operator(const cal::RegSpec&) const { return Eigen::MatrixXd::Zero(p_.n_knots(), p_.n_residuals()); }

  inline static int calls = 0;

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
dv::ShockAxis parallel(std::vector<double> bp) {
  dv::ShockAxis a;
  a.label = "parallel";
  a.values = std::move(bp);
  return a;
}
dv::ShockAxis shift_curve(int role, std::vector<double> bp) {
  dv::ShockAxis a;
  a.kind = dv::ShockAxisKind::ShiftCurve;
  a.label = "curve";
  a.role = role;
  a.values = std::move(bp);
  return a;
}
dv::ShockAxis fx(std::vector<double> rel) {
  dv::ShockAxis a;
  a.kind = dv::ShockAxisKind::Fx;
  a.label = "eurusd";
  a.base = "EUR";
  a.quote = "USD";
  a.values = std::move(rel);
  return a;
}
dv::ScenarioGridRequest three_curve_request() {
  dv::ScenarioGridRequest r;
  r.bundle.curves = {flat_curve(0), flat_curve(0), flat_curve(1)};
  r.x0 = Eigen::Vector3d(0.030, 0.033, 0.020);
  r.book = book_with_xccy();
  return r;
}

}  // namespace

TEST(AddAxisShock, EveryAxisAddsOntoTheCellAndFxCompounds) {
  std::vector<double> delta(3, 0.0);
  double factor = 1.0;
  dv::add_axis_shock(parallel({}), 42.0, delta, factor);
  dv::add_axis_shock(shift_curve(1, {}), -31.75, delta, factor);
  EXPECT_EQ(delta, (std::vector<double>{42.0 / 1e4, 42.0 / 1e4 + -31.75 / 1e4, 42.0 / 1e4}))
      << "a shift_curve axis ADDS to the parallel in a grid (SC1)";
  EXPECT_NE(delta[0], 42.0 * 1e-4) << "bp / 1e4, bit for bit";
  EXPECT_EQ(factor, 1.0);
  dv::add_axis_shock(fx({}), 0.05, delta, factor);
  dv::add_axis_shock(fx({}), -0.02, delta, factor);
  EXPECT_EQ(factor, 1.0 * (1.0 + 0.05) * (1.0 + -0.02));
  EXPECT_EQ(delta[2], 42.0 / 1e4) << "an fx axis moves no curve";
}

TEST(CheckShockAxis, NamesWhatAnAxisIsMissing) {
  EXPECT_NO_THROW(dv::check_shock_axis(parallel({1.0}), 3));
  EXPECT_NO_THROW(dv::check_shock_axis(shift_curve(2, {1.0}), 3));
  dv::ShockAxis no_role = shift_curve(0, {1.0});
  no_role.role.reset();
  EXPECT_THROW(dv::check_shock_axis(no_role, 3), std::invalid_argument) << "a role is required, never 0 by default";
  EXPECT_THROW(dv::check_shock_axis(shift_curve(3, {1.0}), 3), std::invalid_argument);
  EXPECT_THROW(dv::check_shock_axis(shift_curve(-1, {1.0}), 3), std::invalid_argument);
  dv::ShockAxis no_pair = fx({0.01});
  no_pair.quote.clear();
  EXPECT_THROW(dv::check_shock_axis(no_pair, 3), std::invalid_argument);
  EXPECT_THROW(dv::check_shock_axis(parallel({}), 3), std::invalid_argument);
}

TEST(ScenarioGrid, EveryCellIsTheCompiledBookAtItsForkAndItsFxFactor) {
  dv::ScenarioGridRequest r = three_curve_request();
  r.sample_times = {1.0};
  r.axes = {parallel({-36.0, 0.0, 34.5}), fx({-0.05, 0.0, 0.05})};
  const cal::BundleProblem P = r.bundle;
  const Eigen::VectorXd x0 = *r.x0;
  const pf::MultiCurveBook book = *r.book;

  SeedSession::calls = 0;
  const dv::ScenarioGridResult out = dv::scenario_grid<SeedSession>(r);
  EXPECT_EQ(SeedSession::calls, 1) << "one calibration for every cell";
  EXPECT_EQ(out.n0, 3);
  EXPECT_EQ(out.n1, 3);
  EXPECT_EQ(out.n_cells, 9);
  EXPECT_EQ(out.n_positions, 2);
  EXPECT_TRUE(out.x_base == x0);
  ASSERT_EQ(out.base_curves.size(), 3u);
  EXPECT_GE(out.grid_us, 0.0);

  pf::XccyFxScaledBooks books(P.curves, book);
  EXPECT_EQ(out.base_npv, books.at(1.0).npv(x0));
  const double par[] = {-36.0, 0.0, 34.5}, rel[] = {-0.05, 0.0, 0.05};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      const std::vector<double> delta(3, 0.0 + par[i] / 1e4);
      const double want = books.at(1.0 * (1.0 + rel[j])).npv(cal::shift_interp_forwards(P, x0, delta));
      EXPECT_EQ(out.npv[i][j], want) << i << "," << j;
      EXPECT_EQ(out.pnl[i][j], out.npv[i][j] - out.base_npv);
    }
  EXPECT_EQ(out.pnl[1][1], 0.0) << "the zero cell is the base";
  EXPECT_NE(out.npv[1][0], out.npv[1][2]) << "the fx axis reaches the xccy position";
}

TEST(ScenarioGrid, AShiftCurveAxisAfterAParallelAddsAndOneAxisIsAColumn) {
  dv::ScenarioGridRequest r = three_curve_request();
  r.axes = {parallel({42.0}), shift_curve(1, {-31.75, 36.75})};
  const cal::BundleProblem P = r.bundle;
  const Eigen::VectorXd x0 = *r.x0;
  pf::XccyFxScaledBooks books(P.curves, *r.book);
  const dv::ScenarioGridResult out = dv::scenario_grid<SeedSession>(r);
  ASSERT_EQ(out.n0, 1);
  ASSERT_EQ(out.n1, 2);
  const std::vector<double> delta{42.0 / 1e4, 42.0 / 1e4 + -31.75 / 1e4, 42.0 / 1e4};
  EXPECT_EQ(out.npv[0][0], books.at(1.0).npv(cal::shift_interp_forwards(P, x0, delta)));

  dv::ScenarioGridRequest column = three_curve_request();
  column.axes = {shift_curve(2, {-33.25, 14.5})};
  const dv::ScenarioGridResult col = dv::scenario_grid<SeedSession>(column);
  EXPECT_EQ(col.n0, 2);
  EXPECT_EQ(col.n1, 1);
  EXPECT_EQ(col.npv[1][0], books.at(1.0).npv(cal::shift_interp_forwards(P, x0, {0.0, 0.0, 14.5 / 1e4})));
}

TEST(ScenarioGrid, WithoutABookThereIsNoSurfaceAndItChecksItsInputs) {
  dv::ScenarioGridRequest r = three_curve_request();
  r.book.reset();
  r.axes = {parallel({1.0, 2.0})};
  const dv::ScenarioGridResult out = dv::scenario_grid<SeedSession>(r);
  EXPECT_FALSE(out.has_book);
  EXPECT_TRUE(out.npv.empty());
  EXPECT_TRUE(out.pnl.empty());
  EXPECT_EQ(out.n_cells, 0);
  EXPECT_EQ(out.grid_us, 0.0);
  EXPECT_EQ(out.n0, 2);

  SeedSession::calls = 0;
  dv::ScenarioGridRequest none = three_curve_request();
  EXPECT_THROW((void)dv::scenario_grid<SeedSession>(none), std::invalid_argument) << "no axes";
  none.axes = {parallel({1.0}), parallel({1.0}), parallel({1.0})};
  EXPECT_THROW((void)dv::scenario_grid<SeedSession>(none), std::invalid_argument) << "three axes";
  none.axes = {shift_curve(5, {1.0})};
  EXPECT_THROW((void)dv::scenario_grid<SeedSession>(none), std::invalid_argument) << "a role past the bundle";
  EXPECT_EQ(SeedSession::calls, 0) << "the axes are checked before calibrating";
  none.axes = {parallel({1.0})};
  none.x0 = Eigen::VectorXd::Zero(2);
  EXPECT_THROW((void)dv::scenario_grid<SeedSession>(none), std::invalid_argument) << "x0 length";
  EXPECT_THROW((void)dv::scenario_grid<SeedSession>(dv::ScenarioGridRequest{}), std::invalid_argument);
}
