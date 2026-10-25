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
#include "swaps/curve/curve_module.hpp"

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

// THE DF TABLE FOR A VALUE-DEPENDENT CURVE (2026-10-25, K51): a block caches a MonotoneConvex curve's discount factors per
// distinct query time per eval, evaluated once on the real curve, and its rows read the table through the cursor replay.
// The residuals are the SAME doubles the generic kernel computes coupon by coupon on freshly built curves -- bit for bit
// wherever every curve a row reads is value-dependent (every row when every curve is). A row that also reads a LINEAR
// curve takes that curve's DFs from the affine-map cache (one GEMV, since 2026-09), which rounds an ulp away from the
// per-coupon integral: a zero FX-forward residual reads 1e-17 there, as it did before this table.
TEST(AadBlockGroups, ValueDependentDfTableIsTheGenericKernelsDoubles) {
  for (bool everywhere : {false, true}) {
    swaps::shapes::g4::Opts o;
    o.back = cv::Scheme::MonotoneConvex;
    o.vd_everywhere = everywhere;
    o.smoothing = false;
    const Shape s = swaps::shapes::g4_desk(o);
    const auto& p = s.prob;
    const cal::HybridBundleResidual h(p);
    ASSERT_GT(h.n_aad_blocks(), 1) << s.name;
    std::vector<bool> linear_curve(static_cast<std::size_t>(p.n_curves()));
    for (int c = 0; c < p.n_curves(); ++c) linear_curve[static_cast<std::size_t>(c)] = cv::make_modular_curve<double>(p.curves[static_cast<std::size_t>(c)].modules()).is_linear_map();
    const auto reads_a_linear_curve = [&](const cal::Instrument& ins) {
      bool any = false;
      ins.for_each_curve_ref([&](int c, const char*) {
        for (int a = c; a >= 0; a = p.curves[static_cast<std::size_t>(a)].base) any = any || linear_curve[static_cast<std::size_t>(a)];
      });
      return any;
    };
    int bitwise_rows = 0;
    for (const Eigen::VectorXd& x : {Eigen::VectorXd(s.x_true), Eigen::VectorXd(s.x_true.array() + 0.0007), Eigen::VectorXd(s.x0)}) {
      const Eigen::VectorXd r = h.residuals(x);
      const auto C = cal::build_bundle_curves<double>(p.curves, [&](int c, int i) { return x[p.offset(c) + i]; });
      const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
      for (const cal::AadBlock* b : h.aad_blocks())
        for (int row : b->rows()) {
          const double want = cal::instrument_residual<double>(p.row(row), curve_of);
          if (reads_a_linear_curve(p.row(row))) EXPECT_NEAR(r[row], want, 1e-15 + 1e-12 * std::abs(want)) << s.name << " row " << row;
          else { EXPECT_EQ(r[row], want) << s.name << " row " << row; ++bitwise_rows; }
        }
    }
    EXPECT_GT(bitwise_rows, 0) << s.name;
    if (everywhere) EXPECT_EQ(bitwise_rows, 3 * (p.n_residuals() - h.n_compiled_rows())) << s.name << ": every AAD row reads value-dependent curves only";
  }
}
