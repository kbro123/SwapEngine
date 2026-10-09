// E5 taxonomy: T5 properties + value pins (every calibration fixture is over-determined or square, and identified at its solution)
// FIXTURE IDENTIFIABILITY GATE (owner rule, 2026-10-09). A curve-building fixture never has fewer instruments than knots --
// the desk pins the front with a fixing instrument (a deposit / T/N row, the current-month future) and gives every knot
// something that sees it -- so what the engine and its gates are exercised on is an OVER-DETERMINED (or square) fit, never an
// under-determined one. Two checks, on every shape-ladder rung (bench/fixtures/shape_ladder.hpp) and the xccy streaming
// fixture (bench/fixtures/fx_stream_fixture.hpp):
//   1. COUNT: each curve has at least as many rows whose primary curve it is as it has knots.
//   2. RANK: at the state the markets were generated from, the Jacobian has no direction below the streamer's walk threshold
//      (kWalkRankThreshold x sigma_max). That is the bar the walk uses to call a direction weak; a fixture weak at its own
//      solution hands the frozen operator a near-null direction to invert.
// WHY BOTH: the count is the owner's rule and is cheap; the rank is what the count is for, and the count alone does not imply
// it. The xccy fixture that motivated this was SQUARE (24 rows, 24 knots) with rank 22 plus two weak directions
// (sigma / sigma_max 4e-8 and 1.7e-6): USD and EUR had 6 rows for 8 knots (the count failed per curve while the total was
// square), nothing matured at the 0.25y / 0.5y knots and the annual 1y leg telescoped to DF(1y). The streamed tick on it
// spent its steps on those directions -- a 2.9 bp quote move produced a 300 bp first frozen step along a direction the
// quotes see at 1e-6, and where the stall refresh landed (not the step/refresh cost model) decided the tick's cost.
// The rank bar is deliberately the WALK threshold, not the shared rank threshold (1e-10): a fixture that is merely
// non-singular at 1e-10 can still be weak at 1e-4, and the walk keeps anchor-weak directions (a3e469b).
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/SVD>

#include <string>
#include <vector>

#include "fx_stream_fixture.hpp"
#include "shape_ladder.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/streaming.hpp"

namespace cal = swaps::calibration;

namespace {

// The count rule: rows per curve (by Instrument::primary_curve) >= knots per curve.
void expect_counts(const std::string& name, const cal::BundleProblem& p) {
  std::vector<int> rows(p.curves.size(), 0);
  for (const auto& ins : p.instruments) ++rows[static_cast<std::size_t>(ins.primary_curve())];
  EXPECT_GE(p.n_residuals(), p.n_knots()) << name << ": fewer rows than knots";
  for (std::size_t c = 0; c < p.curves.size(); ++c)
    EXPECT_GE(rows[c], p.curves[c].n_knots()) << name << ": curve " << c << " has " << rows[c] << " rows for "
                                                << p.curves[c].n_knots() << " knots";
}

// The rank rule: no singular value below kWalkRankThreshold x sigma_max at the generating state.
void expect_identified(const std::string& name, const cal::BundleProblem& p, const Eigen::VectorXd& x) {
  const cal::HybridBundleResidual hr(p);
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(hr.jacobian(x));
  const auto& sv = svd.singularValues();
  const double ratio = sv(sv.size() - 1) / sv(0);
  EXPECT_GT(ratio, cal::kWalkRankThreshold) << name << ": sigma_min / sigma_max " << ratio
                                            << " -- a direction the walk would call weak at the fixture's own solution";
}

}  // namespace

TEST(FixtureIdentifiability, EveryLadderRungHasAtLeastAsManyRowsAsKnotsPerCurve) {
  for (const auto& s : swaps::shapes::ladder()) expect_counts(s.name, s.prob);
}

TEST(FixtureIdentifiability, EveryLadderRungIsIdentifiedAtItsGeneratingState) {
  for (const auto& s : swaps::shapes::ladder()) expect_identified(s.name, s.prob, s.x_true);
}

TEST(FixtureIdentifiability, TheXccyStreamingFixtureIsOverDeterminedAndIdentified) {
  const auto& f = swaps::fxstream::fixture();
  expect_counts("fx_stream", f.prob);
  EXPECT_GT(f.prob.n_residuals(), f.prob.n_knots()) << "the xccy fixture is over-determined by construction (30 rows, 24 knots)";
  expect_identified("fx_stream (x_true)", f.prob, f.x_true);
  expect_identified("fx_stream (x_solved)", f.prob, f.x_solved);
  // The AAD block under test is still the five compounded-funding MtM rows (the fixture aborts otherwise; pinned here too).
  EXPECT_EQ(static_cast<int>(f.nc_rows.size()), swaps::fxstream::kAadRows);
  // An identified fit recovers the generating state (the markets are its model quotes): the LM solution is x_true.
  EXPECT_LT((f.x_solved - f.x_true).cwiseAbs().maxCoeff(), 1e-8);
}
