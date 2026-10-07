// E5 taxonomy: T1 unit (one component, no oracle)
// The PenaltyMap (problem.hpp, constraint rows 2026-10-07): the ONE description of a row's piecewise-linear residual.
// Pins (1) the band instance is bit for bit the three band forms it replaced (residual<Scalar>, residual_d, slope);
// (2) the piece conventions: a target on an edge is inside, a value on an edge resolves toward the target's piece;
// (3) a general three-piece map (a band with an outer dead zone, the shape a bound would use) is continuous, zero at
// the target, with the declared slopes; (4) piece_from_residual and invert round-trip the double form, which is what
// the streamer's walk reads its sides and model values from; (5) Instrument::penalty is plain for an FX forward.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/calibration/problem.hpp"

namespace cal = swaps::calibration;
namespace ad = swaps::ad;

TEST(PenaltyMap, BandInstanceIsBitForBitTheThreeOldBandForms) {
  const double lo = 0.030, hi = 0.032, m = 0.0309, decay = 0.25;
  const cal::PenaltyMap pm = cal::PenaltyMap::band(lo, hi, decay);
  for (double q : {0.025, lo, 0.0305, m, 0.0315, hi, 0.040}) {
    // the pre-2026-10-07 expressions, verbatim
    const double want = q > hi ? q + (decay * (hi - m) - hi) : (q < lo ? q + (decay * (lo - m) - lo) : (q - m) * decay);
    const double want_d = q > hi ? decay * (hi - m) + (q - hi) : (q < lo ? decay * (lo - m) + (q - lo) : decay * (q - m));
    const double want_s = (q > hi || q < lo) ? 1.0 : decay;
    EXPECT_EQ(pm.residual<double>(q, m), want) << q;
    EXPECT_EQ(pm.residual_d(q, m).first, want_d) << q;
    EXPECT_EQ(pm.residual_d(q, m).second, want_s) << q;
    EXPECT_EQ(pm.slope_at(q, m), want_s) << q;
    EXPECT_EQ(cal::band_residual<double>(q, m, lo, hi, decay), want) << q;
    EXPECT_EQ(cal::band_slope(q, lo, hi, decay), want_s) << q;
  }
  // the dual form carries the slope as its derivative and keeps q the plain operand
  const ad::Dual q(0.035, 1, 0);
  const ad::Dual r = pm.residual<ad::Dual>(q, m);
  EXPECT_EQ(r.value(), 0.035 + (decay * (hi - m) - hi));
  EXPECT_EQ(r.derivatives()[0], 1.0);
  const ad::Dual r_in = pm.residual<ad::Dual>(ad::Dual(0.0305, 1, 0), m);
  EXPECT_EQ(r_in.derivatives()[0], decay);
  // plain: q − m, slope 1
  EXPECT_EQ(cal::PenaltyMap::plain().residual<double>(0.031, 0.030), 0.031 - 0.030);
  EXPECT_EQ(cal::PenaltyMap::plain().residual_d(0.031, 0.030).second, 1.0);
  EXPECT_TRUE(cal::PenaltyMap::band(0.03, 0.03, 0.5).is_plain()) << "an empty band is no band";
}

// The band's unrolled branch of residual_d and the general piece walk agree to rounding: the SAME operations, but
// the two expression shapes let the compiler fuse a different multiply-add (x86-64-v3 FMA contraction), so a
// non-unit outside slope can differ by an ulp. With unit outside slopes (every shipped band) the product is exact
// and the two are bit for bit; the first test pins the unrolled branch against the old expressions verbatim.
TEST(PenaltyMap, TheUnrolledBandBranchEqualsTheGeneralPieceWalk) {
  for (double outer : {1.0, 0.7}) {
    cal::PenaltyMap pm = cal::PenaltyMap::band(0.030, 0.032, 0.25);
    pm.s[0] = outer; pm.s[2] = outer;
    for (double m : {0.030, 0.0305, 0.0311, 0.032})
      for (double q : {0.020, 0.0299, 0.030, 0.0305, m, 0.0319, 0.032, 0.0321, 0.045}) {
        const auto fast = pm.residual_d(q, m), gen = pm.residual_d_general(q, m);
        if (outer == 1.0) EXPECT_EQ(fast.first, gen.first) << "outer " << outer << " m " << m << " q " << q;
        else EXPECT_NEAR(fast.first, gen.first, 4e-16 * (std::abs(gen.first) + 1e-3)) << "outer " << outer << " m " << m << " q " << q;
        EXPECT_EQ(fast.second, gen.second) << "outer " << outer << " m " << m << " q " << q;
      }
  }
  // a target outside its band (refused upstream) still evaluates, through the general path
  const cal::PenaltyMap pm = cal::PenaltyMap::band(0.030, 0.032, 0.25);
  EXPECT_EQ(pm.residual_d(0.031, 0.035).first, pm.residual_d_general(0.031, 0.035).first);
  // the view's walk helpers agree with the general map's piece walk and inversion
  const cal::PenaltyMap::TwoEdge v = pm.two_edge();
  const double m = 0.0311;
  for (double q : {0.020, 0.0299, 0.0305, 0.0319, 0.0321, 0.045}) {
    const double r = pm.residual_d(q, m).first;
    EXPECT_EQ(v.side_from_residual(r, m), pm.piece_from_residual(r, m, 1) - 1) << q;
    EXPECT_EQ(v.side_from_value(q), pm.piece_of(q, 1) - 1) << q;
    EXPECT_EQ(v.invert(v.side_from_value(q), r, m), pm.invert(pm.piece_of(q, 1), r, m, 1)) << q;
  }
  EXPECT_EQ(v.r_at_edge(+1, m), pm.at_break(1, m, 1));
  EXPECT_EQ(v.r_at_edge(-1, m), pm.at_break(0, m, 1));
  EXPECT_THROW((void)cal::PenaltyMap::plain().two_edge(), std::logic_error) << "a plain row has no two-edge view";
}

TEST(PenaltyMap, PieceConventionsResolveEdgesTowardTheTarget) {
  const cal::PenaltyMap pm = cal::PenaltyMap::band(0.030, 0.032, 0.25);
  EXPECT_EQ(pm.target_piece(0.031), 1);
  EXPECT_EQ(pm.target_piece(0.030), 1) << "a target on the lower edge is inside (validate_quote allows it)";
  EXPECT_EQ(pm.target_piece(0.032), 1) << "a target on the upper edge is inside";
  EXPECT_EQ(pm.piece_of(0.030, 1), 1);
  EXPECT_EQ(pm.piece_of(0.032, 1), 1);
  EXPECT_EQ(pm.piece_of(0.0299, 1), 0);
  EXPECT_EQ(pm.piece_of(0.0321, 1), 2);
}

TEST(PenaltyMap, AThreePieceMapIsContinuousZeroAtTheTargetWithItsSlopes) {
  // {1, 0.2, 0, 1}: a band {0.03, 0.032} whose upper side has a DEAD zone to 0.034 then full slope -- a bound's shape.
  cal::PenaltyMap pm;
  pm.n = 3;
  pm.b[0] = 0.030; pm.b[1] = 0.032; pm.b[2] = 0.034;
  pm.s[0] = 1.0; pm.s[1] = 0.2; pm.s[2] = 0.0; pm.s[3] = 1.0;
  const double m = 0.031;
  EXPECT_EQ(pm.target_piece(m), 1);
  EXPECT_EQ(pm.residual<double>(m, m), 0.0);
  EXPECT_EQ(pm.residual_d(m, m).first, 0.0);
  const double h = 1e-9;
  for (double b : {0.030, 0.032, 0.034}) {
    EXPECT_NEAR(pm.residual<double>(b - h, m), pm.residual<double>(b + h, m), 2e-9) << "continuous at " << b;
    EXPECT_NEAR(pm.residual_d(b - h, m).first, pm.residual_d(b + h, m).first, 2e-9) << "continuous at " << b;
  }
  const std::vector<std::pair<double, double>> probes{{0.029, 1.0}, {0.0315, 0.2}, {0.033, 0.0}, {0.036, 1.0}};
  for (const auto& [q, slope] : probes) {
    EXPECT_EQ(pm.slope_at(q, m), slope) << q;
    const double fd = (pm.residual<double>(q + h, m) - pm.residual<double>(q - h, m)) / (2 * h);
    EXPECT_NEAR(fd, slope, 1e-6) << q;
    EXPECT_NEAR(pm.residual<double>(q, m), pm.residual_d(q, m).first, 1e-15) << "the two numeric forms agree";
  }
  // the dead zone: r is constant across it, equal to the residual at its edges
  EXPECT_EQ(pm.residual<double>(0.033, m), pm.residual<double>(0.0325, m));
  EXPECT_EQ(pm.target_slope(m), 0.2);
  // a target ON the edge between the dead zone and the band joins the smaller slope: the dead zone
  EXPECT_EQ(pm.target_piece(0.032), 2);
}

TEST(PenaltyMap, ResidualInversionRoundTripsOnEveryPieceWithPositiveSlope) {
  const cal::PenaltyMap pm = cal::PenaltyMap::band(0.030, 0.032, 0.25);
  const double m = 0.0311;
  const int tp = pm.target_piece(m);
  for (double q : {0.020, 0.0299, 0.0301, 0.0311, 0.0319, 0.0321, 0.045}) {
    const double r = pm.residual_d(q, m).first;
    const int piece = pm.piece_from_residual(r, m, tp);
    EXPECT_EQ(piece, pm.piece_of(q, tp)) << q;
    EXPECT_NEAR(pm.invert(piece, r, m, tp), q, 1e-15) << q;
  }
  // the old side_of arithmetic, verbatim, for the two outer pieces and the band
  const double r_above = pm.residual_d(0.040, m).first;
  EXPECT_EQ(pm.invert(2, r_above, m, tp), 0.032 + (r_above - 0.25 * (0.032 - m)));
  const double r_in = pm.residual_d(0.0315, m).first;
  EXPECT_EQ(pm.invert(1, r_in, m, tp), m + r_in / 0.25);
}

TEST(PenaltyMap, AnFxForwardCarriesNoPenaltyAndABandedRowCarriesItsBand) {
  cal::Instrument fx;
  fx.quote = cal::QuoteKind::FxForward;
  fx.band_lower = 1.0; fx.band_upper = 1.2; fx.band_decay = 0.5;
  EXPECT_TRUE(fx.penalty().is_plain());
  cal::Instrument par;
  par.band_lower = 0.03; par.band_upper = 0.032; par.band_decay = 0.5; par.market = 0.031;
  EXPECT_EQ(par.penalty().n, 2);
  EXPECT_EQ(par.penalty().target_slope(par.market), 0.5);
  cal::Instrument plain;
  EXPECT_TRUE(plain.penalty().is_plain());
}
