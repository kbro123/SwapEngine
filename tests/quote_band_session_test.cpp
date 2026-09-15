// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs) | T5 properties + value pins (hand / closed-form literals, identities, FD)
// K5' SESSION REQUOTE (owner 2026-09-14): BundleSession::stream_update(target, lower, upper, decay) -- a four-number requote tick.
// It must equal the other two ways of getting the same quotes in (rebind with those bands, and a fresh session on the requoted
// problem), move bands without a Jacobian refresh, refuse a requote WHOLE when any row is bad (nothing changes), and re-anchor
// (still converging) when the set of banded rows changes.
#include <cmath>
#include <stdexcept>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "shape_ladder.hpp"
#include "swaps/api/bundle_api.hpp"

namespace api = swaps::api;
namespace cal = swaps::calibration;
using swaps::shapes::Shape;

namespace {

cal::BundleProblem as_problem(const Shape& s, const Shape::Requote& r) {
  cal::BundleProblem p = s.prob;
  for (int i = 0; i < p.n_residuals(); ++i) {
    auto& ins = p.instruments[static_cast<std::size_t>(i)];
    ins.market = r.target[i];
    ins.band_lower = r.lower[i];
    ins.band_upper = r.upper[i];
    ins.band_decay = r.decay[i];
  }
  return p;
}

Eigen::VectorXd moved(const Shape& s, double bp) {
  Eigen::VectorXd q = s.q0;
  for (int i = 0; i < q.size(); ++i) q[i] += bp * 1e-4 * std::sin(0.9 * i + 0.1);
  return q;
}

double inf(const Eigen::VectorXd& a, const Eigen::VectorXd& b) { return (a - b).cwiseAbs().maxCoeff(); }

}  // namespace

TEST(QuoteBandSession, AFourNumberRequoteMatchesRebindAndAFreshSessionWithoutARefresh) {
  const Shape s = swaps::shapes::banded();
  const Shape::Requote r = s.requote(moved(s, 0.5));
  const cal::BundleProblem pq = as_problem(s, r);
  api::BundleSession a(s.prob), b(s.prob), fresh(pq);
  a.calibrate(s.x0);
  b.calibrate(s.x0);
  a.start_streaming();
  a.stream_update(r.target, r.lower, r.upper, r.decay);
  EXPECT_TRUE(a.last_converged()) << a.last_reason();
  EXPECT_EQ(a.last_refreshes(), 0) << "moving bands must not cost a Jacobian";
  b.rebind(pq);
  fresh.calibrate(s.x0);
  EXPECT_LT(inf(a.x(), b.x()), 1e-9);
  EXPECT_LT(inf(a.x(), fresh.x()), 1e-8);
  EXPECT_EQ(a.problem().instruments[0].band_upper, r.upper[0]) << "the session's quotes carry the requote";
}

TEST(QuoteBandSession, ARequoteIsRefusedWholeWhenAnyRowIsBadAndNothingChanges) {
  const Shape s = swaps::shapes::banded();
  api::BundleSession sess(s.prob);
  sess.calibrate(s.x0);
  sess.start_streaming();
  const Eigen::VectorXd x = sess.x();
  Shape::Requote r = s.requote(moved(s, 0.5));
  const int bad = s.prob.n_residuals() - 1;
  r.target[bad] = r.upper[bad] + 1e-6;  // the LAST row is outside its band; every earlier row is a valid move
  EXPECT_THROW(sess.stream_update(r.target, r.lower, r.upper, r.decay), std::invalid_argument);
  EXPECT_EQ(sess.x(), x);
  EXPECT_EQ(sess.problem().instruments[0].band_lower, s.prob.instruments[0].band_lower) << "row 0's valid band did not land";
  EXPECT_THROW(sess.stream_update(r.target.head(3), r.lower, r.upper, r.decay), std::runtime_error) << "lengths";
}

TEST(QuoteBandSession, ChangingWhichRowsAreBandedReanchorsAndStillMatchesAFreshSession) {
  const Shape s = swaps::shapes::banded();
  Shape::Requote r = s.requote(moved(s, 0.3));
  r.lower[0] = r.upper[0] = 0.0;  // row 0 becomes a hard quote
  const cal::BundleProblem pq = as_problem(s, r);
  api::BundleSession a(s.prob), fresh(pq);
  a.calibrate(s.x0);
  a.start_streaming();
  a.stream_update(r.target, r.lower, r.upper, r.decay);
  EXPECT_TRUE(a.last_converged()) << a.last_reason();
  fresh.calibrate(s.x0);
  EXPECT_LT(inf(a.x(), fresh.x()), 1e-8);
}
