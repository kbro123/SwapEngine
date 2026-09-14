// E5 taxonomy: T6 regression (fails on the reverted bug)
// P12 REPRODUCTION (2026-09-14, found by reading while drafting scenario_oracle_test.cpp): a PARALLEL move shifted a
// SPREAD curve twice. calibration::shift_interp_forwards adds each curve's delta to that curve's own interpolation
// knots, and a spread curve's forward is its base's plus its own (pricing/curve_handle.hpp SpreadHandle), so giving a
// spread curve the parallel delta as well moved its forward by 2x the parallel -- in `scenario` (resolve_scenario_move)
// and in `scenario_grid` (add_axis_shock). The goldens use only outright curves, so nothing saw it; the web composer's
// bundles (FF over SOFR, EURIBOR over ESTR) are exactly this shape. A parallel move of d moves EVERY curve's forward by d.
// A shift KEYED to a base curve still reaches the spread curves built on it (that is what a spread curve is), and a key
// on the spread curve moves only its spread.
#include <vector>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "swaps/calibration/bundle_state.hpp"
#include "swaps/derive/scenario.hpp"
#include "swaps/derive/scenario_grid.hpp"

namespace cal = swaps::calibration;
namespace crv = swaps::curve;
namespace dv = swaps::derive;

namespace {

// Curve 0 outright; curve 1 a spread over curve 0. One flat knot each: forward(t) = x0 and x0 + x1.
cal::BundleProblem outright_and_spread() {
  cal::BundleProblem p;
  crv::CurveModule flat;
  flat.scheme = crv::Scheme::Flat;
  flat.knots = {1.0};
  cal::BundleCurveSpec outright;
  outright.regions = {flat};
  cal::BundleCurveSpec spread;
  spread.base = 0;
  spread.regions = {flat};
  p.curves = {outright, spread};
  return p;
}

// Each curve's forward move at t = 2 when `delta` forks the state.
std::vector<double> forward_moves(const cal::BundleProblem& p, const std::vector<double>& delta) {
  const Eigen::VectorXd x = Eigen::Vector2d(0.03, 0.002);
  const auto before = cal::sample_bundle_curves(p, x, {2.0});
  const auto after = cal::sample_bundle_curves(p, cal::shift_interp_forwards(p, x, delta), {2.0});
  return {after[0].forward[0] - before[0].forward[0], after[1].forward[0] - before[1].forward[0]};
}

}  // namespace

TEST(ScenarioSpreadRepro, AParallelScenarioMovesASpreadCurvesForwardOnce) {
  const cal::BundleProblem p = outright_and_spread();
  dv::ScenarioMove par;
  par.parallel_bp = 25.0;
  const std::vector<double> moved = forward_moves(p, dv::resolve_scenario_move(par, p.curves).curve_delta);
  EXPECT_NEAR(moved[0], 25e-4, 1e-15);
  EXPECT_NEAR(moved[1], 25e-4, 1e-15) << "the spread curve moved by base AND spread: 2x the parallel";
}

TEST(ScenarioSpreadRepro, AParallelGridAxisMovesASpreadCurvesForwardOnce) {
  const cal::BundleProblem p = outright_and_spread();
  dv::ShockAxis axis;
  axis.values = {25.0};
  std::vector<double> delta(2, 0.0);
  dv::add_axis_shock(axis, 25.0, p.curves, delta);
  const std::vector<double> moved = forward_moves(p, delta);
  EXPECT_NEAR(moved[0], 25e-4, 1e-15);
  EXPECT_NEAR(moved[1], 25e-4, 1e-15) << "the spread curve moved by base AND spread: 2x the parallel";
}

TEST(ScenarioSpreadRepro, AKeyedShiftReachesTheSpreadCurvesBuiltOnItAndASpreadKeyMovesOnlyTheSpread) {
  const cal::BundleProblem p = outright_and_spread();
  dv::ScenarioMove base_key;
  base_key.shift_curve_bp = {{0, 10.0}};
  const std::vector<double> via_base = forward_moves(p, dv::resolve_scenario_move(base_key, p.curves).curve_delta);
  EXPECT_NEAR(via_base[0], 10e-4, 1e-15);
  EXPECT_NEAR(via_base[1], 10e-4, 1e-15) << "a spread curve sits on its base";

  dv::ScenarioMove spread_key;
  spread_key.shift_curve_bp = {{1, 10.0}};
  const std::vector<double> spread_only = forward_moves(p, dv::resolve_scenario_move(spread_key, p.curves).curve_delta);
  EXPECT_EQ(spread_only[0], 0.0);
  EXPECT_NEAR(spread_only[1], 10e-4, 1e-15);
}
