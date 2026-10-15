// E5 taxonomy: T2 calibration (every variant recovers its generating state; the CCP block vanishes at a flat seed) | T4 hot-path invariant (the router partition and the AAD block width on the desk)
// The G4 desk fixture (bench/fixtures/g4_market.hpp): the hardest bundle the engine calibrates -- four currencies,
// fourteen curves, meeting-date Flat fronts on the RFR curves with a value-dependent long end, tenor / CCP / xccy
// basis -- pinned here so the profile bench (bench/g4_profile_bench.cpp) measures a fixture that is what it says.
#include <gtest/gtest.h>

#include <Eigen/Core>

#include "g4_market.hpp"
#include "swaps/ad/dual.hpp"
#include "swaps/calibration/aad_block.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"

namespace cal = swaps::calibration;
namespace cv = swaps::curve;
namespace g4 = swaps::shapes::g4;
using swaps::shapes::Shape;

namespace {
g4::Opts raw(cv::Scheme back, bool everywhere = false) {
  g4::Opts o;
  o.back = back;
  o.vd_everywhere = everywhere;
  o.smoothing = false;
  return o;
}
}  // namespace

TEST(G4Desk, StructureIsTheDesk) {
  const Shape s = swaps::shapes::g4_desk(raw(cv::Scheme::MonotoneConvex));
  const auto& p = s.prob;
  EXPECT_EQ(p.n_curves(), 14);
  EXPECT_EQ(p.n_knots(), 200);
  EXPECT_EQ(p.n_quotes(), 204);
  EXPECT_EQ(p.n_constraints(), 0);
  // the four RFR curves: a Flat front with one knot per DB meeting before the 1Y pillar, then the smooth back, a turn
  const int meetings[] = {8, 8, 4, 4};
  int i = 0;
  for (const g4::Rfr& r : g4::rfrs()) {
    const auto& c = p.curves[static_cast<std::size_t>(r.role)];
    ASSERT_EQ(c.regions.size(), 2u) << r.index;
    EXPECT_EQ(c.regions[0].scheme, cv::Scheme::Flat) << r.index;
    EXPECT_EQ(static_cast<int>(c.regions[0].knots.size()), meetings[i++]) << r.index;
    EXPECT_EQ(c.regions[1].scheme, cv::Scheme::MonotoneConvex) << r.index;
    EXPECT_EQ(c.regions[1].knots.size(), 12u) << r.index;
    EXPECT_EQ(c.turns.size(), 1u) << r.index;
    EXPECT_LT(c.base, 0) << r.index;
    EXPECT_EQ(p.curves[static_cast<std::size_t>(r.ccp)].base, r.role) << r.ccp_note;  // the CCP curve is a spread over its RFR
  }
  for (const g4::Xccy& x : g4::xccys()) EXPECT_LT(p.curves[static_cast<std::size_t>(x.role)].base, 0) << x.pair;
  EXPECT_EQ(p.curves[g4::EUR_3M].base, g4::EUR_6M);
  EXPECT_EQ(p.curves[g4::EUR_6M].base, g4::EUR_ESTR);
  EXPECT_EQ(p.curves[g4::USD_FF].base, g4::USD_SOFR);
  // a smoothed build stamps every non-Flat region and only those
  const Shape sm = swaps::shapes::g4_desk(g4::Opts{});
  for (const auto& c : sm.prob.curves)
    for (const auto& m : c.regions) EXPECT_EQ(m.reg_lambda > 0, m.scheme != cv::Scheme::Flat);
  EXPECT_FALSE(sm.expect_compiled);
  EXPECT_TRUE(swaps::shapes::g4_desk(raw(cv::Scheme::Hermite)).expect_compiled);
}

// Every variant's raw build calibrates from the fixture's cold seed back to its generating state, full rank: fourteen
// curves, four turns and the four weakly identified CCP blocks all identified.
TEST(G4Desk, EveryVariantCalibratesToItsGeneratingState) {
  for (const auto& [sch, everywhere] : {std::pair{cv::Scheme::Hermite, false}, std::pair{cv::Scheme::MonotoneCubic, false},
                                        std::pair{cv::Scheme::MonotoneConvex, false}, std::pair{cv::Scheme::MonotoneConvex, true}}) {
    const Shape s = swaps::shapes::g4_desk(raw(sch, everywhere));
    const cal::CalibrationResult r = cal::calibrate(s.prob, s.x0);
    EXPECT_TRUE(r.converged) << s.name << ": " << r.status;
    EXPECT_EQ(r.rank_deficiency, 0) << s.name;
    EXPECT_LT((r.x - s.x_true).cwiseAbs().maxCoeff(), 1e-7) << s.name;
    EXPECT_LT(r.iterations, 12) << s.name;
  }
}

// THE CCP BLOCK IS IDENTIFIED ONLY THROUGH THE CURVE'S SLOPE (2026-10-15): at a flat forecast curve every float coupon
// is the same rate, so a par rate is independent of its discounting and the CCP columns vanish there (to the second
// order of the accrual, ~1e-6: a compounded coupon is exp(f tau) - 1, not f tau) -- which is why the fixture's cold seed
// carries a slope -- and at x_true they are still three orders below the RFR columns.
TEST(G4Desk, TheCcpBlockIsIdentifiedOnlyThroughTheSlope) {
  const Shape s = swaps::shapes::g4_desk(raw(cv::Scheme::Hermite));
  const auto& p = s.prob;
  const cal::HybridBundleResidual h(p);
  Eigen::VectorXd flat = s.x0;
  for (const g4::Rfr& r : g4::rfrs())
    for (int i = 0; i < p.curves[static_cast<std::size_t>(r.role)].n_interp_knots(); ++i) flat[p.offset(r.role) + i] = 0.03;
  const Eigen::MatrixXd J0 = h.jacobian(flat), J1 = h.jacobian(s.x_true);
  for (const g4::Rfr& r : g4::rfrs()) {
    const int off = p.offset(r.ccp), n = p.curves[static_cast<std::size_t>(r.ccp)].n_knots();
    const double ccp0 = J0.middleCols(off, n).colwise().norm().maxCoeff();
    const double ccp = J1.middleCols(off, n).colwise().norm().minCoeff();
    EXPECT_LT(ccp0, 1e-5) << r.ccp_note << ": vanishes at a flat forecast";
    EXPECT_LT(ccp0, 1e-1 * ccp) << r.ccp_note << ": the flat-seed column is far below the sloped one";
    const double rfr = J1.middleCols(p.offset(r.role), p.curves[static_cast<std::size_t>(r.role)].n_knots()).colwise().norm().maxCoeff();
    EXPECT_GT(ccp, 1e-5) << r.ccp_note;
    EXPECT_LT(ccp, 1e-2 * rfr) << r.ccp_note;
  }
  EXPECT_TRUE(cal::calibrate(p, s.x0).converged);  // the sloped seed
}

// The router's partition on the desk: Hermite and MonotoneCubic (the piecewise-linear tier) compile every row; the
// MonotoneConvex long ends send the rows reading past the RFR horizons to the AAD block, whose touched width is the
// whole state -- above the pooled dual's limit, so the block runs on the heap dual (the profile bench's finding).
TEST(G4Desk, ThePartitionIsWhatTheSchemeSays) {
  for (cv::Scheme sch : {cv::Scheme::Hermite, cv::Scheme::MonotoneCubic}) {
    const Shape s = swaps::shapes::g4_desk(raw(sch));
    const cal::HybridBundleResidual h(s.prob);
    EXPECT_EQ(h.n_compiled_rows(), s.prob.n_residuals()) << s.name;
  }
  const Shape s = swaps::shapes::g4_desk(raw(cv::Scheme::MonotoneConvex));
  const cal::HybridBundleResidual h(s.prob);
  EXPECT_LT(h.n_compiled_rows(), s.prob.n_residuals());
  const cal::AadBlock* blk = nullptr;
  for (const auto& e : h.engines()) if (auto* b = dynamic_cast<const cal::AadBlock*>(e.get())) blk = b;
  ASSERT_NE(blk, nullptr);
  EXPECT_EQ(blk->touched_width(), s.prob.n_knots());
  EXPECT_GT(blk->touched_width(), swaps::ad::kPooledMaxW);
  EXPECT_FALSE(blk->pooled());
}
