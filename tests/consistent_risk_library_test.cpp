// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// calibration/consistent_risk.hpp and null_completed_ladder (E7 stage 3.7): the library behind generate_risk, driven
// by a LINEAR fake session whose calibration is a pseudo-inverse -- so the anchor / re-level / rank-completion logic is
// checked against hand arithmetic, independent of any curve. Header-only (swaps_tests), so tools/mutate.py reaches it.
// The verb end to end stays pinned by tests/consistent_risk_test.cpp.
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "swaps/calibration/consistent_risk.hpp"

namespace cal = swaps::calibration;
namespace crv = swaps::curve;

namespace {

Eigen::MatrixXd pinv(const Eigen::MatrixXd& A) {
  return Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>(A).pseudoInverse();
}

// Quote i = w_i · x, with w_i carried in the instrument's obs.weight. calibrate() is the least-squares solve; the
// book's npv is g · x.
struct FakeReprice {
  double npv = 0.0, pv01 = 0.0;
};
struct FakeRisk {
  Eigen::VectorXd curve_grad;
};
struct FakeBook {
  std::vector<int> positions;
  Eigen::VectorXd g;
};

class FakeSession {
 public:
  explicit FakeSession(cal::BundleProblem p) : p_(std::move(p)) {
    A_.resize(p_.n_residuals(), p_.n_knots());
    for (int i = 0; i < A_.rows(); ++i)
      for (int k = 0; k < A_.cols(); ++k) A_(i, k) = p_.instruments[static_cast<std::size_t>(i)].obs.weight[static_cast<std::size_t>(k)];
  }
  const cal::CalibrationResult& calibrate(const Eigen::VectorXd&, const cal::RegSpec&) {
    x_ = pinv(A_) * p_.market();
    res_.x = x_;
    res_.converged = true;
    return res_;
  }
  const cal::BundleProblem& problem() const { return p_; }
  const Eigen::VectorXd& x() const { return x_; }
  Eigen::MatrixXd jacobian(const cal::RegSpec&) const { return A_; }
  Eigen::MatrixXd risk_operator(const cal::RegSpec&) const { return pinv(A_); }
  FakeReprice price_portfolio(const FakeBook& b) const { return {b.g.dot(x_), 1e-4 * b.g.sum()}; }
  FakeRisk price_portfolio_risk(const FakeBook& b) const { return {b.g}; }
  double model_quote(const cal::Instrument& ins) const {
    return Eigen::Map<const Eigen::VectorXd>(ins.obs.weight.data(), x_.size()).dot(x_);
  }
  bool same_curve_set(const cal::BundleProblem& q) const { return q.curves.size() == p_.curves.size(); }

 private:
  cal::BundleProblem p_;
  Eigen::MatrixXd A_;
  Eigen::VectorXd x_;
  cal::CalibrationResult res_;
};
static_assert(cal::RiskSession<FakeSession, FakeBook>);

// One curve with `n_knots` knots; one instrument per weight row, market = row · x_true + shift.
cal::BundleProblem linear_bundle(const std::vector<std::vector<double>>& rows, const Eigen::VectorXd& x_true,
                                 double shift, int n_curves = 1) {
  cal::BundleProblem p;
  for (int c = 0; c < n_curves; ++c) {
    cal::BundleCurveSpec spec;
    crv::CurveModule m;
    m.scheme = crv::Scheme::Linear;
    for (int k = 0; k < x_true.size() / n_curves; ++k) m.knots.push_back(1.0 + k);
    spec.regions = {m};
    p.curves.push_back(spec);
  }
  for (const std::vector<double>& w : rows) {
    cal::Instrument ins;
    ins.quote = cal::QuoteKind::Rate;
    ins.obs.weight = w;
    ins.market = Eigen::Map<const Eigen::VectorXd>(w.data(), static_cast<Eigen::Index>(w.size())).dot(x_true) + shift;
    p.instruments.push_back(ins);
  }
  return p;
}

}  // namespace

TEST(NullCompletedLadder, OneQuoteOnTwoKnotsSelfQuotesTheUnseenDirection) {
  Eigen::MatrixXd J(1, 2);
  J << 1.0, 1.0;
  Eigen::VectorXd g(2);
  g << 1.0, 3.0;
  const cal::NullCompletedLadder lad = cal::null_completed_ladder(J, g, Eigen::VectorXd::Ones(J.rows()));
  // Jf = [1 1; v] with v = ±(1,-1)/sqrt2. Jfᵀ y = g gives y1 = 2 and |y2| = sqrt2 whatever the sign of v.
  EXPECT_EQ(lad.n_residuals, 1);
  ASSERT_EQ(lad.full.size(), 2);
  ASSERT_EQ(lad.synthetic_knot.size(), 1u);
  EXPECT_NEAR(lad.full[0], 2.0, 1e-14);
  EXPECT_NEAR(std::abs(lad.full[1]), std::sqrt(2.0), 1e-14);
}

TEST(NullCompletedLadder, ASingularValueBelowTheSharedRankThresholdIsUnseen) {
  Eigen::VectorXd g(2);
  g << 1.0, 2.0;
  Eigen::MatrixXd J = Eigen::MatrixXd::Identity(2, 2);
  J(1, 1) = 1e-12;  // sigma ratio 1e-12 < kRankThreshold 1e-10: Eigen's default threshold would call it constrained
  const cal::NullCompletedLadder lad = cal::null_completed_ladder(J, g, Eigen::VectorXd::Ones(J.rows()));
  ASSERT_EQ(lad.synthetic_knot.size(), 1u);
  EXPECT_EQ(lad.synthetic_knot[0], 1);
  J(1, 1) = 1e-6;  // stiff but constrained
  EXPECT_TRUE(cal::null_completed_ladder(J, g, Eigen::VectorXd::Ones(J.rows())).synthetic_knot.empty());
}

TEST(ConsistentRiskLibrary, ReLevelsEveryBundleOntoTheAnchor) {
  Eigen::VectorXd x_true(2);
  x_true << 0.03, 0.01;
  const std::vector<std::vector<double>> rows = {{1.0, 0.0}, {1.0, 1.0}};
  FakeBook book{{0, 1}, Eigen::Vector2d(2.0, -1.0)};
  const cal::ConsistentRisk r = cal::consistent_risk<FakeSession, FakeBook>(
      book, {linear_bundle(rows, x_true, 0.0), linear_bundle(rows, x_true, 25e-4)}, cal::RegSpec{});
  EXPECT_EQ(r.n, 2);
  ASSERT_EQ(r.bundles.size(), 2u);
  EXPECT_NEAR(r.npv, 2.0 * 0.03 - 0.01, 1e-15);
  // Bundle 1's +25 bp market is overwritten by the anchor's model quotes, so it prices the anchor's npv...
  EXPECT_NEAR(r.bundles[1].npv, r.npv, 1e-15);
  // ... where calibrating its own market would move npv by g·pinv(A)·(25bp, 25bp) = 2·25bp.
  EXPECT_GT(std::abs(book.g.dot(pinv(Eigen::Matrix2d{{1.0, 0.0}, {1.0, 1.0}}) * Eigen::Vector2d(25e-4, 25e-4))), 1e-3);

  // Square and full rank: no synthetic pillars; the ladder is pinv(A)ᵀ g and its DV01 is 1 bp times its sum.
  const cal::ConsistentBundleRisk& b0 = r.bundles[0];
  EXPECT_EQ(b0.n_residuals, 2);
  EXPECT_EQ(b0.n_synthetic, 0);
  const Eigen::Vector2d want = pinv(Eigen::Matrix2d{{1.0, 0.0}, {1.0, 1.0}}).transpose() * book.g;
  EXPECT_NEAR(b0.ladder[0], want[0], 1e-14);
  EXPECT_NEAR(b0.ladder[1], want[1], 1e-14);
  EXPECT_EQ(b0.ladder_dv01, 1e-4 * (b0.ladder[0] + b0.ladder[1]));
}

TEST(ConsistentRiskLibrary, AnUnderDeterminedBundleGetsOneSyntheticPillar) {
  Eigen::VectorXd x_true(2);
  x_true << 0.02, 0.02;
  FakeBook book{{0}, Eigen::Vector2d(1.0, 3.0)};
  const cal::ConsistentRisk r =
      cal::consistent_risk<FakeSession, FakeBook>(book, {linear_bundle({{1.0, 1.0}}, x_true, 0.0)}, cal::RegSpec{});
  const cal::ConsistentBundleRisk& b = r.bundles[0];
  EXPECT_EQ(b.n_residuals, 1);
  EXPECT_EQ(b.n_synthetic, 1);
  ASSERT_EQ(b.ladder.size(), 1);
  ASSERT_EQ(b.synthetic.size(), 1);
  EXPECT_NEAR(b.ladder[0], 2.0, 1e-14);  // the hand case of OneQuoteOnTwoKnotsSelfQuotesTheUnseenDirection
  EXPECT_NEAR(std::abs(b.synthetic[0]), std::sqrt(2.0), 1e-14);
}

TEST(ConsistentRiskLibrary, RefusesAnEmptyListOrAMismatchedCurveSet) {
  Eigen::VectorXd x2(2);
  x2 << 0.03, 0.01;
  FakeBook book{{0}, Eigen::Vector2d(1.0, 1.0)};
  EXPECT_THROW((void)(cal::consistent_risk<FakeSession, FakeBook>(book, {}, cal::RegSpec{})), std::invalid_argument);
  const std::vector<std::vector<double>> rows = {{1.0, 0.0}, {0.0, 1.0}};
  EXPECT_THROW((void)(cal::consistent_risk<FakeSession, FakeBook>(
                   book, {linear_bundle(rows, x2, 0.0), linear_bundle(rows, x2, 0.0, /*n_curves=*/2)}, cal::RegSpec{})),
               std::invalid_argument);
}

TEST(ConsistentRiskLibrary, TheReLevelingCalibrationIsTheCallersRegulariserElseLight) {
  Eigen::VectorXd x3(3);
  x3 << 0.03, 0.03, 0.03;
  const cal::BundleProblem p = linear_bundle({{1.0, 0.0, 0.0}}, x3, 0.0, /*n_curves=*/3);
  const cal::RegSpec light = cal::relevel_calibration_reg(cal::RegSpec{}, p);
  const cal::RegSpec want = cal::smoothing_preset(cal::Smoothing::Light, 3);
  EXPECT_EQ(light.lambda, want.lambda);
  EXPECT_EQ(light.curves, want.curves);
  EXPECT_EQ(light.tension, want.tension);
  cal::RegSpec mine;
  mine.lambda = 0.5;
  mine.curves = {1};
  EXPECT_EQ(cal::relevel_calibration_reg(mine, p).lambda, 0.5);
  EXPECT_EQ(cal::relevel_calibration_reg(mine, p).curves, (std::vector<int>{1}));
}

// dP/dq = D ⊙ dP/dr: the library applies the residual market scale of each quote to its real ladder row (1 for a hard
// pin, the decay for a band). The fake's J ignores the band on purpose -- this pins the library's rule, not band physics;
// the physics is pinned end to end by tests/risk_scale_repro_test.cpp against a re-calibrated finite difference.
TEST(ConsistentRiskLibrary, TheLadderIsInQuoteUnitsForABandedRow) {
  Eigen::VectorXd x_true(2);
  x_true << 0.03, 0.01;
  cal::BundleProblem p = linear_bundle({{1.0, 0.0}, {1.0, 1.0}}, x_true, 0.0);
  p.instruments[1].band_lower = p.instruments[1].market - 1e-3;
  p.instruments[1].band_upper = p.instruments[1].market + 1e-3;
  p.instruments[1].band_decay = 0.25;
  FakeBook book{{0}, Eigen::Vector2d(2.0, -1.0)};
  const cal::ConsistentRisk r = cal::consistent_risk<FakeSession, FakeBook>(book, {p}, cal::RegSpec{});
  const Eigen::Vector2d dPdr = pinv(Eigen::Matrix2d{{1.0, 0.0}, {1.0, 1.0}}).transpose() * book.g;
  EXPECT_NEAR(r.bundles[0].ladder[0], dPdr[0], 1e-14);
  EXPECT_NEAR(r.bundles[0].ladder[1], 0.25 * dPdr[1], 1e-14);
  EXPECT_EQ(r.bundles[0].ladder_dv01, 1e-4 * (r.bundles[0].ladder[0] + r.bundles[0].ladder[1]));
  EXPECT_THROW((void)cal::null_completed_ladder(Eigen::MatrixXd::Identity(2, 2), Eigen::VectorXd::Ones(2),
                                                Eigen::VectorXd::Ones(3)),
               std::invalid_argument);
}
