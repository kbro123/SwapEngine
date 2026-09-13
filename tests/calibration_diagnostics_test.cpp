// E5 taxonomy: T5 properties + value pins (hand / closed-form literals, identities, FD)
// calibration/diagnostics.hpp and flat_x0 (E7 stage 3.6): the library behind the calib_report verb, on matrices and
// problems small enough to work out by hand. Header-only (swaps_tests), so tools/mutate.py reaches it. The verb end
// to end stays pinned by tests/calib_report_test.cpp and tests/consistent_risk_test.cpp.
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "swaps/calibration/diagnostics.hpp"

namespace cal = swaps::calibration;
namespace crv = swaps::curve;
namespace px = swaps::pricing;

namespace {
Eigen::MatrixXd pinv(const Eigen::MatrixXd& A) { return Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>(A).pseudoInverse(); }
}  // namespace

TEST(JacobianConditioning, DiagonalTwoByTwo) {
  Eigen::MatrixXd J(2, 2);
  J << 3.0, 0.0, 0.0, 0.5;
  const cal::JacobianConditioning c = cal::jacobian_conditioning(J);
  ASSERT_EQ(c.singular_values.size(), 2);
  EXPECT_NEAR(c.singular_values[0], 3.0, 1e-15);
  EXPECT_NEAR(c.singular_values[1], 0.5, 1e-15);
  EXPECT_NEAR(c.condition_number, 6.0, 1e-14);
}

TEST(JacobianConditioning, AShearHasTheGoldenRatioSpectrum) {
  Eigen::MatrixXd J(2, 2);
  J << 1.0, 1.0, 0.0, 1.0;  // JᵀJ = [[1,1],[1,2]]: eigenvalues phi², 1/phi²
  const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
  const cal::JacobianConditioning c = cal::jacobian_conditioning(J);
  EXPECT_NEAR(c.singular_values[0], phi, 1e-14);
  EXPECT_NEAR(c.singular_values[1], 1.0 / phi, 1e-14);
  EXPECT_NEAR(c.condition_number, phi * phi, 1e-13);
}

TEST(JacobianConditioning, AnExactNullIsTheLargestDoubleAndAnEmptyJacobianIsOne) {
  Eigen::MatrixXd J(2, 2);
  J << 1.0, 0.0, 0.0, 0.0;
  EXPECT_EQ(cal::jacobian_conditioning(J).condition_number, std::numeric_limits<double>::max());
  const cal::JacobianConditioning empty = cal::jacobian_conditioning(Eigen::MatrixXd(0, 3));
  EXPECT_EQ(empty.condition_number, 1.0);
  EXPECT_EQ(empty.singular_values.size(), 0);
}

TEST(HatDiagonal, DuplicateRowsSplitTheirLeverage) {
  Eigen::MatrixXd J(3, 2);
  J << 1.0, 0.0, 0.0, 1.0, 0.0, 1.0;  // quotes 2 and 3 pin the same knot
  const Eigen::VectorXd h = cal::hat_diagonal(J, pinv(J));
  EXPECT_NEAR(h[0], 1.0, 1e-14);
  EXPECT_NEAR(h[1], 0.5, 1e-14);
  EXPECT_NEAR(h[2], 0.5, 1e-14);
  EXPECT_NEAR(h.sum(), 2.0, 1e-14) << "trace of the hat matrix = rank";
}

TEST(HatDiagonal, IsClampedToTheUnitIntervalAndChecksShapes) {
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(2, 2);
  EXPECT_EQ(cal::hat_diagonal(I, 2.0 * I), Eigen::VectorXd::Ones(2));
  EXPECT_EQ(cal::hat_diagonal(I, -1.0 * I), Eigen::VectorXd::Zero(2));
  EXPECT_THROW(cal::hat_diagonal(Eigen::MatrixXd::Zero(3, 2), Eigen::MatrixXd::Zero(3, 2)), std::invalid_argument);
}

TEST(QuoteDiagnostic, HardPinsAndBands) {
  cal::Instrument hard;
  hard.market = 0.03;
  const cal::QuoteDiagnostic h = cal::quote_diagnostic(hard, 0.031);
  EXPECT_EQ(h.model, 0.031);
  EXPECT_EQ(h.target, 0.03);
  EXPECT_EQ(h.residual, 0.031 - 0.03);
  EXPECT_FALSE(h.soft);
  EXPECT_FALSE(h.in_band);
  EXPECT_EQ(h.weight, 1.0);

  cal::Instrument soft = hard;
  soft.band_lower = 0.02;
  soft.band_upper = 0.04;
  soft.band_decay = 0.25;
  const cal::QuoteDiagnostic in = cal::quote_diagnostic(soft, 0.031);
  EXPECT_TRUE(in.soft);
  EXPECT_TRUE(in.in_band);
  EXPECT_EQ(in.weight, 0.25) << "inside the band the residual's slope is the decay";
  EXPECT_EQ(in.lower, 0.02);
  EXPECT_EQ(in.upper, 0.04);
  EXPECT_EQ(in.decay, 0.25);
  const cal::QuoteDiagnostic out = cal::quote_diagnostic(soft, 0.05);
  EXPECT_FALSE(out.in_band);
  EXPECT_EQ(out.weight, 1.0) << "outside the band it pulls to the edge at unit slope";
}

TEST(FlatX0, IsTheMeanOutrightQuoteClampedWithSpreadCurvesAndTurnsAtZero) {
  cal::BundleProblem p;
  cal::BundleCurveSpec outright;
  crv::CurveModule front;
  front.scheme = crv::Scheme::Flat;
  front.knots = {1.0, 2.0};
  outright.regions = {front};
  outright.turns = {px::Turn{0.1, 0.2}};
  cal::BundleCurveSpec spread;
  spread.base = 0;
  crv::CurveModule s;
  s.scheme = crv::Scheme::Linear;
  s.knots = {1.0};
  spread.regions = {s};
  p.curves = {outright, spread};
  const auto quote = [](cal::QuoteKind k, double m) {
    cal::Instrument i;
    i.quote = k;
    i.market = m;
    return i;
  };
  // Rate 2 % and ParRate 4 % average to 3 %; a ParSpread row is not an outright level.
  p.instruments = {quote(cal::QuoteKind::Rate, 0.02), quote(cal::QuoteKind::ParRate, 0.04),
                   quote(cal::QuoteKind::ParSpread, 0.5)};
  Eigen::VectorXd want(4);
  want << 0.03, 0.03, 0.0, 0.0;  // two outright knots at the level, the turn delta and the spread knot at zero
  EXPECT_TRUE(cal::flat_x0(p).isApprox(want, 1e-15));

  p.instruments = {quote(cal::QuoteKind::ParRate, 0.5)};
  EXPECT_EQ(cal::flat_x0(p)[0], 0.20) << "clamped above at 20 %";
  p.instruments = {quote(cal::QuoteKind::ParRate, 1e-5)};
  EXPECT_EQ(cal::flat_x0(p)[0], 1e-3) << "clamped below at 0.1 %";
  p.instruments = {quote(cal::QuoteKind::ParSpread, 0.01)};
  EXPECT_EQ(cal::flat_x0(p)[0], 0.02) << "no outright rows";
  EXPECT_EQ(cal::flat_x0(p, 0.07)[0], 0.07) << "an explicit level";
}

TEST(SeedOrFlat, IsTheCallersSeedLengthCheckedElseTheFlatSeed) {
  cal::BundleProblem p;
  cal::BundleCurveSpec c;
  crv::CurveModule m;
  m.scheme = crv::Scheme::Hermite;
  m.knots = {1.0, 2.0, 5.0};
  c.regions = {m};
  p.curves = {c};
  EXPECT_TRUE(cal::seed_or_flat(p, std::nullopt, "r").isApprox(cal::flat_x0(p)));
  const Eigen::VectorXd mine = Eigen::VectorXd::Constant(3, 0.04);
  EXPECT_EQ(cal::seed_or_flat(p, mine, "r"), mine);
  EXPECT_THROW(cal::seed_or_flat(p, Eigen::VectorXd::Constant(2, 0.04), "r"), std::invalid_argument);
}
