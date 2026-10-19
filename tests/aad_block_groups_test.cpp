// E5 taxonomy: T3 cross-path parity (the grouped AAD blocks against one block over the same rows, and against the full AAD Jacobian) | T4 hot-path invariant (every block pooled on the G4 desk)
// ONE AAD BLOCK PER CLOSURE GROUP (2026-10-19). HybridBundleResidual used to put every non-cacheable row in ONE AadBlock,
// whose touched width is the union of its rows' curve closures: on the G4 desk that is the whole state (200 knots against
// kPooledMaxW 64), so the block ran on the heap dual at 16 ms a Jacobian. Rows are now grouped by closure under the pooled
// width (first fit in row order), each block building and updating only its closure's curves. Pinned here: the grouped
// Jacobian is the SAME doubles as one block over the same rows (bit for bit; residuals to an ulp through the DF cache), they match the full AAD
// Jacobian, every block on the desk is pooled, and a rung whose closure fits stays one block.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include <Eigen/Core>

#include "g4_market.hpp"
#include "shape_ladder.hpp"
#include "swaps/ad/dual.hpp"
#include "swaps/calibration/aad_block.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/jacobian.hpp"

namespace cal = swaps::calibration;
namespace cv = swaps::curve;
using swaps::shapes::Shape;

namespace {
// One AadBlock over every row the hybrid's blocks own (the pre-2026-10-19 form), as the parity reference.
struct SingleBlock {
  cal::AadBlock blk;
  std::vector<int> rows;
  explicit SingleBlock(const cal::BundleProblem& p, const cal::HybridBundleResidual& h) {
    std::vector<cal::Instrument> ins;
    for (const cal::AadBlock* b : h.aad_blocks())
      for (int r : b->rows()) { rows.push_back(r); ins.push_back(p.row(r)); }
    blk.init(p.curves, ins, rows, p.n_knots());
  }
};
void expect_same_rows(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const std::vector<int>& rows, const char* what) {
  for (int r : rows)
    for (int k = 0; k < A.cols(); ++k) EXPECT_EQ(A(r, k), B(r, k)) << what << " row " << r << " col " << k;
}
}  // namespace

TEST(AadBlockGroups, GroupedBlocksAreOneBlocksDoublesBitForBit) {
  for (bool everywhere : {false, true}) {
    swaps::shapes::g4::Opts o;
    o.back = cv::Scheme::MonotoneConvex;
    o.vd_everywhere = everywhere;
    o.smoothing = false;
    const Shape s = swaps::shapes::g4_desk(o);
    const auto& p = s.prob;
    const cal::HybridBundleResidual h(p);
    ASSERT_GT(h.n_aad_blocks(), 1) << s.name;
    EXPECT_TRUE(h.aad_pooled()) << s.name << ": a block fell off the pooled dual at width " << h.aad_width();
    EXPECT_LE(h.aad_width(), swaps::ad::kPooledMaxW) << s.name;
    const SingleBlock one(p, h);
    EXPECT_GT(one.blk.touched_width(), swaps::ad::kPooledMaxW) << s.name << ": the single block would be on the heap";
    EXPECT_FALSE(one.blk.pooled()) << s.name;
    const Eigen::VectorXd x = s.x_true.array() + 0.0007;  // off the generating state: nonzero residuals
    // Jacobian (stored mids) and residuals
    const Eigen::MatrixXd Jg = h.jacobian(x);
    Eigen::MatrixXd J1 = Eigen::MatrixXd::Zero(p.n_residuals(), p.n_knots());
    one.blk.jacobian_into(x, J1);
    expect_same_rows(Jg, J1, one.rows, "jacobian");
    // Residuals go through each block's DF cache, whose GEMV now runs over the closure's columns instead of the whole
    // state: the same sum with the zero terms dropped, so the vectorised summation order differs by an ulp.
    const Eigen::VectorXd rg = h.residuals(x);
    Eigen::VectorXd r1 = Eigen::VectorXd::Zero(p.n_residuals());
    one.blk.residuals_into(x, r1);
    for (int r : one.rows) EXPECT_NEAR(rg[r], r1[r], 4e-16 * (1.0 + std::abs(r1[r]))) << s.name << " residual row " << r;
    // the live-market forms the streamer uses
    Eigen::MatrixXd Jvg = Eigen::MatrixXd::Zero(p.n_residuals(), p.n_knots()), Jv1 = Jvg;
    Eigen::VectorXd rvg = Eigen::VectorXd::Zero(p.n_residuals()), rv1 = rvg;
    h.jacobian_vs_into(x, s.q_small, Jvg, &rvg);
    one.blk.jacobian_vs_into(x, s.q_small, Jv1, &rv1);
    expect_same_rows(Jvg, Jv1, one.rows, "jacobian_vs");
    for (int r : one.rows) EXPECT_EQ(rvg[r], rv1[r]) << s.name << " residual_vs row " << r;  // the sweep's value part: no cache
    // and the independent path: the full AAD Jacobian of the problem
    const Eigen::MatrixXd Jaad = cal::aad_jacobian(p, x);
    for (int r : one.rows)
      for (int k = 0; k < p.n_knots(); ++k) EXPECT_NEAR(Jg(r, k), Jaad(r, k), 1e-12 * (1.0 + std::abs(Jaad(r, k)))) << s.name;
  }
}

// A rung whose non-cacheable rows share one closure that fits the pooled width stays ONE block, as before.
TEST(AadBlockGroups, ARungWhoseClosureFitsStaysOneBlock) {
  for (const Shape& s : {swaps::shapes::mixed_convex(), swaps::shapes::desk_mixed(), swaps::shapes::fx_xccy()}) {
    const cal::HybridBundleResidual h(s.prob);
    if (h.n_aad_blocks() == 0) continue;  // fx_xccy compiles whole
    EXPECT_EQ(h.n_aad_blocks(), 1) << s.name;
    EXPECT_TRUE(h.aad_pooled()) << s.name;
  }
}
