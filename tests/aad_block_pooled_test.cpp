// AadBlock pooled-Dual retype (R11, the width-reduced hybrid path): the block's forward-AAD sweep runs on
// ad::DualPooled<ad::kPooledMaxW> (gradient in-object, allocation-free) whenever the touched width fits,
// and falls back to the heap ad::Dual unchanged beyond that. These tests pin:
//   * pooled == heap BIT-FOR-BIT (DualPooled preserves Dual's dynamic-length / empty-gradient semantics,
//     so the sweep computes the exact same doubles in the exact same order);
//   * both match the full-width aad_jacobian oracle on the non-cacheable rows (touched-set completeness);
//   * a bundle whose touched width exceeds kPooledMaxW engages the heap fallback and still matches.
// QuantLib-free: the FX/MtM bundle is synthetic (Portfolio-nested FX forwards + payment-lagged MtM basis,
// both genuinely non-cacheable per hybrid_residual.hpp's instrument_is_noncacheable).
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <vector>

#include "swaps/ad/dual.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/jacobian.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace ad = swaps::ad;

namespace {

constexpr int USD = 0, EUR = 1, EURUSD = 2;
constexpr double FX_SPOT = 1.10;
constexpr double PAY_LAG = 2.0 / 365.0;

struct Legs {
  std::vector<px::FloatCoupon> flt;
  std::vector<px::FixedCoupon> fix;
};
Legs annual(double T, double lag = 0.0) {
  Legs L;
  double prev = 0.0;
  for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev};
    c.obs.sub_end = {u};
    c.obs.tau_index = u - prev;
    c.pay = u + lag;
    c.tau_pay = u - prev;
    L.flt.push_back(c);
    L.fix.push_back({u + lag, u - prev});
    prev = u;
  }
  return L;
}
cal::Instrument par_inst(double T, int role) {
  Legs L = annual(T);
  cal::Instrument in;
  in.quote = cal::QuoteKind::ParRate;
  in.fwd = {L.flt, role, role};
  in.fixed = {L.fix, role};
  return in;
}
// FX forward nested in a 1-component Portfolio -> non-cacheable (the compiled transforms don't compose).
cal::Instrument fx_portfolio_inst(double T) {
  cal::Instrument fx;
  fx.quote = cal::QuoteKind::FxForward;
  fx.fx_num = EURUSD;
  fx.fx_den = USD;
  fx.fx_spot = FX_SPOT;
  fx.fx_time = T;
  cal::Instrument wrap;
  wrap.quote = cal::QuoteKind::Portfolio;
  wrap.combination = {{1.0, fx}};
  return wrap;
}
// MtM xccy basis whose funding coupons carry the compounded PRODUCT observation -> AAD block (a plain MtM leg compiles).
cal::Instrument mtm_basis_inst(double T) {
  Legs Le = annual(T);
  Legs Lu = annual(T, PAY_LAG);
  cal::Instrument in;
  in.quote = cal::QuoteKind::XccyMtmBasis;
  in.fwd = {Le.flt, EURUSD, EURUSD};
  in.bench = {Le.flt, EUR, EURUSD};
  in.fixed = {Le.fix, EURUSD};
  in.mtm = {Lu.flt, USD, USD};
  in.mtm.reset_num = EURUSD;
  in.mtm.reset_den = USD;
  in.mtm.fx_spot = FX_SPOT;
  // Since 2026-09-09 an MtM leg is W-cacheable; this fixture needs a GENUINELY non-cacheable MtM-shaped row for the
  // AAD block, so its funding coupons carry the compounded PRODUCT form (single sub-period: same value, AAD route).
  for (auto& c : in.mtm.coupons) c.obs.compounded = true;
  return in;
}

struct Fixture {
  cal::BundleProblem prob;
  Eigen::VectorXd x;
  std::vector<cal::Instrument> nc;
  std::vector<int> nc_rows;

  // `back` sets the per-curve knot density: the narrow fixture's touched width is 3*|knots| <= 48, the
  // wide one's exceeds it (every non-cacheable row touches all three curves via FX + spread ancestry).
  explicit Fixture(const std::vector<double>& back) {
    const std::vector<double> meet{0.25};
    prob.curves.resize(3);
    prob.curves[USD] = px::CurveStructure{.base = -1, .currency = 0, .regions = swaps::curve::flat_hermite(meet, back)};
    prob.curves[EUR] = px::CurveStructure{.base = -1, .currency = 1, .regions = swaps::curve::flat_hermite(meet, back)};
    prob.curves[EURUSD] = px::CurveStructure{.base = EUR, .currency = 1, .regions = swaps::curve::flat_hermite(meet, back)};

    for (double T : {1.0, 2.0, 3.0, 5.0, 7.0, 10.0}) prob.instruments.push_back(par_inst(T, USD));
    for (double T : {1.0, 2.0, 3.0, 5.0, 7.0, 10.0}) prob.instruments.push_back(par_inst(T, EUR));
    for (double T : {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}) prob.instruments.push_back(fx_portfolio_inst(T));
    for (double T : {2.0, 3.0, 5.0, 7.0, 10.0}) prob.instruments.push_back(mtm_basis_inst(T));

    const int N = prob.n_knots();
    x.resize(N);
    auto fill = [&](int c, double lvl, double slope) {
      for (int i = 0; i < prob.curves[c].n_knots(); ++i) x[prob.offset(c) + i] = lvl + slope * i;
    };
    fill(USD, 0.0430, 0.0004);
    fill(EUR, 0.0300, 0.0004);
    fill(EURUSD, -0.0015, 0.00002);

    // Self-consistent markets (model quotes at x), so the residuals under differentiation are realistic.
    const auto C = cal::build_bundle_curves<double>(
        prob.curves, [&](int c, int i) { return x[prob.offset(c) + i]; });
    const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
    for (auto& ins : prob.instruments) ins.market = cal::instrument_model_quote<double>(ins, curve_of);

    for (int r = 0; r < prob.n_residuals(); ++r)
      if (cal::instrument_is_noncacheable(prob.instruments[r], prob.curves)) {
        nc.push_back(prob.instruments[r]);
        nc_rows.push_back(r);
      }
  }
};

}  // namespace

// Narrow bundle (touched width 24 <= kPooledMaxW = 48): the block selects the POOLED sweep, and its
// Jacobian equals the force-heap block's ENTRY FOR ENTRY, bit-for-bit -- on both the stored-mid and the
// live-market (streaming `_vs`) forms. Both also match the full-width AAD oracle on the block's rows.
TEST(AadBlockPooled, NarrowBlockPooledMatchesHeapBitForBit) {
  const Fixture f(std::vector<double>{0.5, 1, 2, 3, 5, 7, 10});  // 8 knots/curve -> width 24
  ASSERT_EQ(f.prob.n_knots(), 24);
  ASSERT_FALSE(f.nc_rows.empty()) << "fixture must genuinely exercise the AAD block";
  ASSERT_EQ(static_cast<int>(f.nc_rows.size()), 12);  // 7 Portfolio-FX + 5 lagged MtM

  cal::AadBlock pooled, heap;
  pooled.init(f.prob.curves, f.nc, f.nc_rows, f.prob.n_knots());
  heap.init(f.prob.curves, f.nc, f.nc_rows, f.prob.n_knots(), /*force_heap=*/true);
  EXPECT_TRUE(pooled.pooled()) << "width 24 <= kPooledMaxW must select the pooled dual";
  EXPECT_FALSE(heap.pooled());

  const int n = f.prob.n_residuals(), m = f.prob.n_knots();
  Eigen::MatrixXd Jp = Eigen::MatrixXd::Zero(n, m), Jh = Eigen::MatrixXd::Zero(n, m);
  pooled.jacobian_into(f.x, Jp);
  heap.jacobian_into(f.x, Jh);
  EXPECT_EQ((Jp - Jh).cwiseAbs().maxCoeff(), 0.0)
      << "pooled and heap AAD sweeps must be bit-identical (same doubles, same order)";

  // Streaming form against a live market: same bit-identity (the band/log chain rule included).
  Eigen::VectorXd q = f.prob.market();
  for (int i = 0; i < q.size(); ++i) q[i] += 1e-4 * ((i % 3) - 1);  // ~1bp live move
  Eigen::MatrixXd Jpv = Eigen::MatrixXd::Zero(n, m), Jhv = Eigen::MatrixXd::Zero(n, m);
  pooled.jacobian_vs_into(f.x, q, Jpv);
  heap.jacobian_vs_into(f.x, q, Jhv);
  EXPECT_EQ((Jpv - Jhv).cwiseAbs().maxCoeff(), 0.0)
      << "pooled and heap streaming (vs) sweeps must be bit-identical";

  // Oracle: the full-width AAD Jacobian of the whole problem, on the block's rows (this also proves the
  // touched-knot set captures every nonzero column -- untouched columns of Jp are zero by construction).
  const Eigen::MatrixXd Jfull = cal::aad_jacobian(f.prob, f.x);
  double worst = 0.0;
  for (int r : f.nc_rows) worst = std::max(worst, (Jp.row(r) - Jfull.row(r)).cwiseAbs().maxCoeff());
  EXPECT_LT(worst, 1e-9) << "width-reduced pooled Jacobian matches the full-width AAD oracle";
}

// Wide bundle (touched width 63 > kPooledMaxW): the block must FALL BACK to the heap-Dual path unchanged
// -- and still match the full-width AAD oracle, including through the hybrid engine end to end.
TEST(AadBlockPooled, WideBundleFallsBackToHeapAndMatches) {
  std::vector<double> back;  // 20 back knots/curve -> 21 knots/curve -> touched width 63 > 48
  for (int i = 1; i <= 20; ++i) back.push_back(0.5 * i);
  const Fixture f(back);
  ASSERT_EQ(f.prob.n_knots(), 63);
  ASSERT_GT(f.prob.n_knots(), ad::kPooledMaxW);

  cal::AadBlock blk;
  blk.init(f.prob.curves, f.nc, f.nc_rows, f.prob.n_knots());
  EXPECT_FALSE(blk.pooled()) << "touched width 63 > kPooledMaxW must engage the heap fallback";

  const int n = f.prob.n_residuals(), m = f.prob.n_knots();
  Eigen::MatrixXd Jb = Eigen::MatrixXd::Zero(n, m);
  blk.jacobian_into(f.x, Jb);
  const Eigen::MatrixXd Jfull = cal::aad_jacobian(f.prob, f.x);
  double worst = 0.0;
  for (int r : f.nc_rows) worst = std::max(worst, (Jb.row(r) - Jfull.row(r)).cwiseAbs().maxCoeff());
  EXPECT_LT(worst, 1e-9) << "heap-fallback width-reduced Jacobian matches the full-width AAD oracle";

  // End to end through the hybrid engine (W-cache rows + the fallback AAD block, stitched).
  const cal::HybridBundleResidual hr(f.prob);
  const double dj = (hr.jacobian(f.x) - Jfull).cwiseAbs().maxCoeff();
  EXPECT_LT(dj, 1e-8) << "hybrid Jacobian on the wide bundle matches full AAD";
}
