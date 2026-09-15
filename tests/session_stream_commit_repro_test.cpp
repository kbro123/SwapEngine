// E5 taxonomy: T6 regression (repro-first: fails on 55ab196, where a converged streamed tick left the session's quotes at the old market)
// P3 (2026-09-15 hot-path audit, A2): BundleSession::stream_update solved the curve to the new market but never wrote that market into the
// session -- not into its problem (problem().instruments[i].market) and not into its shared compiled engine. Everything that reads the
// session's quotes afterwards read the OLD market: resolve() re-solved back to it (desk moved a knot by 4.7e-3, ois_nolag by 6e-5),
// residual() / quote_diagnostics() reported the old gaps, and result().rms_residual was measured against the old targets. A tick that
// FAILED did commit (its LM fallback called set_market), so the two outcomes disagreed about what the session's market was.
#include <algorithm>
#include <cmath>
#include <iostream>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "shape_ladder.hpp"
#include "swaps/api/bundle_api.hpp"

namespace api = swaps::api;
namespace cal = swaps::calibration;
using swaps::shapes::Shape;

namespace {

double inf(const Eigen::VectorXd& a, const Eigen::VectorXd& b) { return (a - b).cwiseAbs().maxCoeff(); }

double stored_gap(const api::BundleSession& s, const Eigen::VectorXd& q) {
  double g = 0.0;
  for (int i = 0; i < q.size(); ++i) g = std::max(g, std::abs(s.problem().instruments[static_cast<std::size_t>(i)].market - q[i]));
  return g;
}

}  // namespace

// The session's quotes after a converged tick ARE the tick's market, and resolve() at that market stays on the streamed curve.
TEST(SessionStreamCommitRepro, AConvergedTickCommitsItsTargetsSoResolveStaysOnTheStreamedCurve) {
  for (const Shape& s : {swaps::shapes::ois_nolag(), swaps::shapes::desk()}) {
    api::BundleSession sess(s.prob);
    sess.calibrate(s.x0);
    sess.start_streaming();
    const Eigen::VectorXd x_stream = sess.stream_update(s.q_small);
    ASSERT_TRUE(sess.last_converged()) << s.name << ": premise -- the 0.1 bp tick converges on the fast path: " << sess.last_reason();
    const double gap = stored_gap(sess, s.q_small);
    sess.resolve();
    const double moved = inf(sess.x(), x_stream);
    std::cout << "  [p3] " << s.name << ": stored-target gap " << gap << ", resolve() moved x by " << moved << "\n";
    EXPECT_EQ(gap, 0.0) << s.name << ": the session's quotes must carry the market the tick solved to";
    EXPECT_LT(moved, 1e-9) << s.name << ": resolve() after a converged tick must re-solve THAT market, not the previous one";
  }
}

// What reads the session's quotes (residual, and the rms the shared engine reports) sees the streamed market. ois_nolag is square and
// hard, so the streamed curve reprices its market exactly: every residual and the rms are at the step-tolerance floor.
TEST(SessionStreamCommitRepro, ResidualsAfterATickAreMeasuredAgainstTheStreamedMarket) {
  const Shape s = swaps::shapes::ois_nolag();
  api::BundleSession sess(s.prob);
  sess.calibrate(s.x0);
  sess.start_streaming();
  sess.stream_update(s.q_small);
  ASSERT_TRUE(sess.last_converged()) << sess.last_reason();
  double worst = 0.0;
  for (const cal::Instrument& ins : sess.problem().instruments) worst = std::max(worst, std::abs(sess.residual(ins)));
  sess.resolve();  // the shared engine's residuals: result().rms_residual is computed from its stored targets
  std::cout << "  [p3] ois_nolag: max |residual()| " << worst << ", resolve rms " << sess.result().rms_residual << "\n";
  EXPECT_LT(worst, 1e-10) << "residual() must price against the streamed market (a 0.1 bp tick is a 1e-5 gap)";
  EXPECT_LT(sess.result().rms_residual, 1e-10) << "the shared engine must carry the streamed targets";
}

// The four-number requote commits its targets as well as its bands.
TEST(SessionStreamCommitRepro, AFourNumberRequoteCommitsItsTargets) {
  const Shape s = swaps::shapes::banded();
  const Shape::Requote r = s.requote(s.q_small);
  api::BundleSession sess(s.prob);
  sess.calibrate(s.x0);
  sess.start_streaming();
  const Eigen::VectorXd x_stream = sess.stream_update(r.target, r.lower, r.upper, r.decay);
  ASSERT_TRUE(sess.last_converged()) << sess.last_reason();
  const double gap = stored_gap(sess, r.target);
  sess.resolve();
  const double moved = inf(sess.x(), x_stream);
  std::cout << "  [p3] banded requote: stored-target gap " << gap << ", resolve() moved x by " << moved << "\n";
  EXPECT_EQ(gap, 0.0);
  EXPECT_EQ(sess.problem().instruments[0].band_upper, r.upper[0]);
  EXPECT_LT(moved, 1e-9);
}
