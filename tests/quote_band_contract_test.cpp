// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD) | T3 cross-path parity (two engine paths, same inputs)
// K5' QUOTE CONTRACT (owner 2026-09-14): a quote is four numbers {target, lower, upper, decay}; a band only gives the solve freedom
// around its target, so the target lies inside it. Header-only (tools/mutate.py reaches it): cal::validate_quote, the wire's
// market::band_target, and the streamer's four-number requote (StreamingCalibrator::set_bands) -- which must be a row re-scale,
// never a Jacobian refresh, and land on the requoted problem's optimum.
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/problem.hpp"
#include "swaps/calibration/streaming.hpp"
#include "swaps/market/quote.hpp"

namespace cal = swaps::calibration;
namespace mk = swaps::market;
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
  for (int i = 0; i < q.size(); ++i)
    if (s.prob.instruments[static_cast<std::size_t>(i)].quote != cal::QuoteKind::FxForward) q[i] += bp * 1e-4 * std::sin(0.9 * i + 0.1);
  return q;
}

}  // namespace

TEST(QuoteBandContract, ValidateQuoteAcceptsTheEdgesAndRefusesEveryBadQuote) {
  EXPECT_NO_THROW(cal::validate_quote(0.030, 0.029, 0.031, 0.5, "t", 0));
  EXPECT_NO_THROW(cal::validate_quote(0.031, 0.029, 0.031, 0.5, "t", 0)) << "on the upper edge";
  EXPECT_NO_THROW(cal::validate_quote(0.029, 0.029, 0.031, 0.0, "t", 0)) << "on the lower edge, decay 0";
  EXPECT_NO_THROW(cal::validate_quote(0.050, 0.0, 0.0, 1.0, "t", 0)) << "no band: any target";
  EXPECT_NO_THROW(cal::validate_quote(0.050, 0.02, 0.02, 7.0, "t", 0)) << "upper == lower is no band: decay unused";
  EXPECT_THROW(cal::validate_quote(0.0311, 0.029, 0.031, 0.5, "t", 0), std::invalid_argument) << "above the band";
  EXPECT_THROW(cal::validate_quote(0.0289, 0.029, 0.031, 0.5, "t", 0), std::invalid_argument) << "below the band";
  EXPECT_THROW(cal::validate_quote(0.030, 0.031, 0.029, 0.5, "t", 0), std::invalid_argument) << "inverted";
  try {
    cal::validate_quote(0.030, 0.031, 0.029, 0.5, "set_band", 3);
    FAIL() << "expected a refusal";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("inverted band"), std::string::npos) << "an inverted band is named as one: " << e.what();
  }
  EXPECT_THROW(cal::validate_quote(0.030, 0.029, 0.031, 1.5, "t", 0), std::invalid_argument) << "decay 1.5";
  EXPECT_THROW(cal::validate_quote(0.030, 0.029, 0.031, -0.1, "t", 0), std::invalid_argument) << "decay -0.1";
  EXPECT_THROW(cal::validate_quote(std::nan(""), 0.0, 0.0, 1.0, "t", 0), std::invalid_argument) << "non-finite";
  try {
    cal::validate_quote(0.0311, 0.029, 0.031, 0.5, "set_market", 7);
    FAIL() << "expected a refusal";
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("set_market row 7"), std::string::npos) << e.what();
  }
}

TEST(QuoteBandContract, AWireQuotesBandMustContainItsTargetAndAPinMustEqualIt) {
  const mk::CalibrationTarget band = mk::band_target(0.030, 0.029, 0.031, std::nullopt);
  EXPECT_EQ(band.target, 0.030);
  EXPECT_EQ(band.band_lower, 0.029);
  EXPECT_EQ(band.band_upper, 0.031);
  EXPECT_EQ(band.band_decay, 1.0) << "absent decay: 1 (parity with server/compile.py)";
  const mk::CalibrationTarget pin = mk::band_target(0.030, 0.030, 0.030, 0.0);
  EXPECT_TRUE(pin.is_hard_pin());
  EXPECT_EQ(pin.target, 0.030);
  EXPECT_THROW(mk::band_target(0.030, 0.031, 0.029, std::nullopt), std::invalid_argument) << "inverted";
  EXPECT_THROW(mk::band_target(0.0301, 0.030, 0.030, std::nullopt), std::invalid_argument) << "pin != target";
  EXPECT_THROW(mk::band_target(0.032, 0.029, 0.031, std::nullopt), std::invalid_argument) << "target outside";
  EXPECT_THROW(mk::band_target(0.030, 0.029, 0.031, 1.2), std::invalid_argument) << "decay 1.2";
}

// A requote that moves every target AND band is a per-row constant change: no Jacobian refresh, and the streamed point is the
// requoted problem's optimum (a cold LM polished from it cannot move it).
TEST(QuoteBandContract, AFourNumberRequoteIsARescaleNotAReanchorOnTheBandedRung) {
  const Shape s = swaps::shapes::banded();
  const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
  cal::HybridBundleResidual eng(s.prob);
  cal::StreamingCalibrator<cal::BundleProblem> st(eng, s.prob, x, s.q0, {});
  const Shape::Requote r = s.requote(moved(s, 0.5));
  for (int i = 0; i < s.prob.n_residuals(); ++i) eng.set_quote(i, r.target[i], r.lower[i], r.upper[i], r.decay[i]);
  ASSERT_TRUE(st.set_bands(r.lower, r.upper, r.decay)) << "the same rows stay banded";
  const cal::StreamTick t = st.update(r.target);
  EXPECT_TRUE(t.converged) << t.reason();
  EXPECT_EQ(t.refreshes, 0) << "a band move is not a Jacobian refresh";
  const Eigen::VectorXd xc = cal::calibrate(as_problem(s, r), st.current()).x;
  EXPECT_LT((st.current() - xc).cwiseAbs().maxCoeff(), 1e-8);
}

// The same on the over-determined desk rung (hard rows, butterflies, a turn): banded model quotes sit away from their targets,
// so the streamer's band table (not just the engine's) must carry the moved bands -- a stale table would weight rows by the old
// sides and stop at a different, worse fixed point.
TEST(QuoteBandContract, AFourNumberRequoteLandsOnTheRequotedOptimumOnTheDeskRung) {
  const Shape s = swaps::shapes::desk();
  const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
  cal::HybridBundleResidual eng(s.prob);
  cal::StreamingCalibrator<cal::BundleProblem> st(eng, s.prob, x, s.q0, {});
  for (double bp : {0.5, -0.8}) {
    const Shape::Requote r = s.requote(moved(s, bp));
    for (int i = 0; i < s.prob.n_residuals(); ++i) eng.set_quote(i, r.target[i], r.lower[i], r.upper[i], r.decay[i]);
    ASSERT_TRUE(st.set_bands(r.lower, r.upper, r.decay));
    const cal::StreamTick t = st.update(r.target);
    ASSERT_TRUE(t.converged) << t.reason();
    const cal::BundleProblem pq = as_problem(s, r);
    const cal::HybridBundleResidual ref(pq);
    const Eigen::VectorXd xc = cal::calibrate(pq, st.current()).x;
    const double f_s = ref.residuals_vs(st.current(), r.target).squaredNorm(), f_c = ref.residuals_vs(xc, r.target).squaredNorm();
    EXPECT_LE(f_s, f_c * (1.0 + 1e-6) + 1e-20) << bp << " bp: streamed objective " << f_s << " vs cold-polished " << f_c;
  }
}

// The streamer's OWN band edges move with a requote (set_bands), not just its refusal table and the engine's bands. With stale edges
// the breakpoint walk and the pins work against the old band: on the desk rung twelve 1 bp requotes failed 7 ticks, paid 46 Jacobian
// refreshes and left a knot 9e-2 from the optimum (probe 2026-09-15; a 0.5 bp move does not reach an edge, so the test above passes).
TEST(QuoteBandContract, ARequoteMovesTheStreamersBandEdgesSoTheWalkUsesTheNewBand) {
  const Shape s = swaps::shapes::desk();
  const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
  cal::HybridBundleResidual eng(s.prob);
  cal::StreamingCalibrator<cal::BundleProblem> st(eng, s.prob, x, s.q0, {});
  int refreshes = 0;
  for (int k = 0; k < 12; ++k) {
    Eigen::VectorXd q = s.q0;
    for (int i = 0; i < q.size(); ++i) q[i] += 1e-4 * std::sin(0.37 * k + 1.3 * i);
    const Shape::Requote r = s.requote(q);
    for (int i = 0; i < s.prob.n_residuals(); ++i) eng.set_quote(i, r.target[i], r.lower[i], r.upper[i], r.decay[i]);
    ASSERT_TRUE(st.set_bands(r.lower, r.upper, r.decay)) << "tick " << k << ": the same rows stay banded";
    const cal::StreamTick t = st.update(r.target);
    ASSERT_TRUE(t.converged) << "tick " << k << ": " << t.reason();
    refreshes += t.refreshes;
    // A cold LM polished from the streamed state moves it by at most 1.9e-8 (ticks 3 and 11), along a direction the desk's over-determined
    // objective is flat in (equal to 1e-10 relative); a stale band table leaves it 9e-2 away, when its ticks converge at all.
    const Eigen::VectorXd polished = cal::calibrate(as_problem(s, r), st.current()).x;
    EXPECT_LT((st.current() - polished).cwiseAbs().maxCoeff(), 1e-7) << "tick " << k << ": the streamed state is not the requoted optimum";
  }
  EXPECT_EQ(refreshes, 0) << "a band move rides the frozen operator (row re-scales), never a Jacobian refresh";
}

TEST(QuoteBandContract, SetBandsRefusesAMembershipChangeAndTheStreamerRefusesATargetOutsideTheMovedBand) {
  const Shape s = swaps::shapes::banded();
  const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
  cal::HybridBundleResidual eng(s.prob);
  cal::StreamingCalibrator<cal::BundleProblem> st(eng, s.prob, x, s.q0, {});
  const Shape::Requote r = s.requote(moved(s, 0.5));
  Eigen::VectorXd up = r.upper;
  up[0] = r.lower[0];  // row 0 would lose its band
  EXPECT_FALSE(st.set_bands(r.lower, up, r.decay)) << "a change in which rows are banded is a re-anchor, not an in-place move";
  for (int i = 0; i < s.prob.n_residuals(); ++i) eng.set_quote(i, r.target[i], r.lower[i], r.upper[i], r.decay[i]);
  ASSERT_TRUE(st.set_bands(r.lower, r.upper, r.decay));
  Eigen::VectorXd q = r.target;
  q[0] = r.upper[0] + 1e-6;  // just outside the MOVED band
  const Eigen::VectorXd before = st.current();
  EXPECT_THROW(st.update(q), std::invalid_argument);
  EXPECT_EQ(st.current(), before) << "a refused tick commits nothing";
}
