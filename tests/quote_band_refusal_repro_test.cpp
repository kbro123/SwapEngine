// E5 taxonomy: T6 regression (fails on the reverted bug)
// K5' REPRODUCTION (2026-09-14; owner). A quote is four numbers {target, lower, upper, decay}; a band exists only to give the solve
// freedom around its target, so a target outside its band -- which turns the quote's squared residual concave at that edge, where
// the streamed and the cold solve can settle on different fits -- must be REFUSED before any solve, as must an inverted band and a
// decay outside [0, 1]. On e20a538 EVERY entry accepts them: session construction, set_market, recalibrate, stream_update,
// set_band, rebind, the hybrid engine's construction and the streamer's own update. Only today's APIs are used here, so this file
// compiles and fails on the unfixed tree. Controls: a target exactly ON an edge, and an unbanded row, are accepted.
#include <stdexcept>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "shape_ladder.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"

namespace api = swaps::api;
namespace cal = swaps::calibration;

namespace {

const swaps::shapes::Shape& banded() {
  static const swaps::shapes::Shape s = swaps::shapes::banded();  // SOFR OIS, a +-1 bp band on every row
  return s;
}

// The first banded row of the fixture and a target just above its band.
int row() {
  const auto& ins = banded().prob.instruments;
  for (int i = 0; i < static_cast<int>(ins.size()); ++i)
    if (ins[static_cast<std::size_t>(i)].band_upper > ins[static_cast<std::size_t>(i)].band_lower) return i;
  return -1;
}
double above() { return banded().prob.instruments[static_cast<std::size_t>(row())].band_upper + 1e-5; }  // 0.1 bp outside

Eigen::VectorXd market_with_row_above() {
  Eigen::VectorXd q = banded().q0;
  q[row()] = above();
  return q;
}

}  // namespace

TEST(QuoteBandRefusalRepro, Premise) {
  ASSERT_GE(row(), 0) << "the fixture carries a banded row";
  const auto& ins = banded().prob.instruments[static_cast<std::size_t>(row())];
  ASSERT_TRUE(ins.market >= ins.band_lower && ins.market <= ins.band_upper) << "the fixture's own target is inside its band";
}

TEST(QuoteBandRefusalRepro, SessionConstructionRefusesATargetOutsideItsBand) {
  cal::BundleProblem p = banded().prob;
  p.instruments[static_cast<std::size_t>(row())].market = above();
  EXPECT_THROW(api::BundleSession{p}, std::invalid_argument);
}

TEST(QuoteBandRefusalRepro, EngineConstructionRefusesATargetOutsideItsBand) {
  cal::BundleProblem p = banded().prob;
  p.instruments[static_cast<std::size_t>(row())].market = above();
  EXPECT_THROW(cal::HybridBundleResidual{p}, std::invalid_argument);
}

TEST(QuoteBandRefusalRepro, SetMarketRefusesAndLeavesTheTargetsUnchanged) {
  api::BundleSession sess(banded().prob);
  EXPECT_THROW(sess.set_market(market_with_row_above()), std::invalid_argument);
  EXPECT_EQ(sess.problem().instruments[static_cast<std::size_t>(row())].market, banded().q0[row()]);
}

TEST(QuoteBandRefusalRepro, RecalibrateRefuses) {
  api::BundleSession sess(banded().prob);
  sess.calibrate(banded().x0);
  EXPECT_THROW(sess.recalibrate(market_with_row_above()), std::invalid_argument);
}

TEST(QuoteBandRefusalRepro, StreamUpdateRefusesAndCommitsNothing) {
  api::BundleSession sess(banded().prob);
  sess.calibrate(banded().x0);
  sess.start_streaming();
  const Eigen::VectorXd x = sess.x();
  EXPECT_THROW(sess.stream_update(market_with_row_above()), std::invalid_argument);
  EXPECT_EQ(sess.x(), x);
}

TEST(QuoteBandRefusalRepro, SetBandRefusesABandThatExcludesTheTarget) {
  api::BundleSession sess(banded().prob);
  const double m = banded().q0[row()];
  EXPECT_THROW(sess.set_band(row(), m + 1e-5, m + 2e-5, 0.5), std::invalid_argument);
}

TEST(QuoteBandRefusalRepro, SetBandRefusesAnInvertedBandAndADecayOutsideZeroOne) {
  api::BundleSession sess(banded().prob);
  const auto& ins = banded().prob.instruments[static_cast<std::size_t>(row())];
  EXPECT_THROW(sess.set_band(row(), ins.band_upper, ins.band_lower, 0.5), std::invalid_argument) << "inverted";
  EXPECT_THROW(sess.set_band(row(), ins.band_lower, ins.band_upper, 1.5), std::invalid_argument) << "decay 1.5";
  EXPECT_THROW(sess.set_band(row(), ins.band_lower, ins.band_upper, -0.1), std::invalid_argument) << "decay -0.1";
}

TEST(QuoteBandRefusalRepro, RebindRefuses) {
  api::BundleSession sess(banded().prob);
  sess.calibrate(banded().x0);
  cal::BundleProblem p = banded().prob;
  p.instruments[static_cast<std::size_t>(row())].market = above();
  EXPECT_THROW(sess.rebind(p), std::invalid_argument);
}

TEST(QuoteBandRefusalRepro, TheStreamersOwnUpdateRefusesAndCommitsNothing) {
  const auto& s = banded();
  const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
  cal::StreamingCalibrator<cal::BundleProblem> sc(s.prob, x, s.prob.market(), {});
  EXPECT_THROW(sc.update(market_with_row_above()), std::invalid_argument);
  EXPECT_EQ(sc.current(), x);
}

TEST(QuoteBandRefusalRepro, ControlATargetOnItsEdgeAndAnUnbandedRowAreAccepted) {
  const auto& s = banded();
  api::BundleSession sess(s.prob);
  Eigen::VectorXd q = s.q0;
  q[row()] = s.prob.instruments[static_cast<std::size_t>(row())].band_upper;  // exactly on the edge
  EXPECT_NO_THROW(sess.set_market(q));
  cal::BundleProblem p = s.prob;
  auto& ins = p.instruments[static_cast<std::size_t>(row())];
  ins.band_lower = ins.band_upper = 0.0;  // no band: any target
  ins.market += 25e-4;
  EXPECT_NO_THROW(api::BundleSession{p});
}
