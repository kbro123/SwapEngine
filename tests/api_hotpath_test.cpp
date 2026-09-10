// E5 taxonomy: T4 hot-path invariant (allocation / determinism / structure) | T3 cross-path parity (two engine paths, same inputs)
// E4.D A4/D7 + D6 (2026-09-10): the ONE-SHOT price_portfolio's PV01 is the all-ones directional derivative of a
// width-1 dual (ad::seed_directional) -- equal to the full-width gradient's sum to rounding -- so the one-shot no
// longer pays a heap-vector dual per knot to compute one sum (71k allocations on chain8x26/book200 in the E3
// census); and risk_operator takes its tension block from the session cache (ensure_reg_R) instead of rebuilding
// it per call (4,449 allocations). Both are pinned by an AllocScope count; the pins may only DECREASE.
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>

#include "malloc_count.hpp"
#include "swaps/ad/dual.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/regularize.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/portfolio/portfolio.hpp"
#include "swaps/pricing/cashflows.hpp"

namespace api = swaps::api;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace ad = swaps::ad;
namespace pf = swaps::portfolio;

namespace {
// A 4-curve spread chain (26 Hermite knots each: 104 knots > ad::kPooledMaxW, so the old one-shot took the heap
// Dual path) with a 40-position swap book -- the census fixture (bench/hotpath_census.cpp) at half width.
struct Legs { std::vector<px::FloatCoupon> flt; std::vector<px::FixedCoupon> fix; };
Legs annual(double T) {
  Legs L; double prev = 0.0;
  for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
    px::FloatCoupon c; c.obs.sub_start = {prev}; c.obs.sub_end = {u}; c.obs.tau_index = u - prev; c.pay = u; c.tau_pay = u - prev;
    L.flt.push_back(c); L.fix.push_back({u, u - prev}); prev = u;
  }
  return L;
}
struct Chain {
  static constexpr int NC = 4, NK = 26;
  cal::BundleProblem prob;
  Eigen::VectorXd x0;
  api::RegSpec tension;
  pf::MultiCurveBook book;
  Chain() {
    std::vector<double> meeting{0.25}, back;
    for (int i = 1; i <= NK - 1; ++i) back.push_back(30.0 * i / (NK - 1));
    prob.curves.resize(NC);
    prob.curves[0] = px::CurveStructure{.base = -1, .regions = swaps::curve::flat_hermite(meeting, back)};
    for (int c = 1; c < NC; ++c) prob.curves[c] = px::CurveStructure{.base = c - 1, .regions = swaps::curve::flat_hermite(meeting, back)};
    for (double T = 1.0; T <= 30.0 + 1e-9; T += 1.0) {
      Legs L = annual(T); cal::Instrument in; in.quote = cal::QuoteKind::ParRate; in.fwd = {L.flt, 0, 0}; in.fixed = {L.fix, 0};
      prob.instruments.push_back(in);
    }
    for (int c = 1; c < NC; ++c)
      for (double T = 1.0; T <= 30.0 + 1e-9; T += 1.0) {
        Legs L = annual(T); cal::Instrument in; in.quote = cal::QuoteKind::ParSpread; in.fwd = {L.flt, c, 0}; in.bench = {L.flt, c - 1, 0}; in.fixed = {L.fix, 0};
        prob.instruments.push_back(in);
      }
    Eigen::VectorXd x_true(NC * NK);
    for (int c = 0; c < NC; ++c) for (int i = 0; i < NK; ++i) x_true[c * NK + i] = (c == 0) ? 0.040 + 0.0005 * i : 0.0020 + 0.0001 * i;
    const Eigen::VectorXd r0 = prob.residuals<double>(x_true);
    for (int i = 0; i < int(prob.instruments.size()); ++i) prob.instruments[i].market += r0[i];
    x0.resize(NC * NK);
    for (int c = 0; c < NC; ++c) for (int i = 0; i < NK; ++i) x0[c * NK + i] = (c == 0) ? 0.040 : 0.0020;
    tension.lambda = 1e-3; tension.tension = true; tension.sigma = 0.5;
    for (int c = 0; c < NC; ++c) tension.curves.push_back(c);
    for (int i = 0; i < 40; ++i) {
      const double T = 1.0 + (i % 30); Legs L = annual(T);
      pf::MultiCurveBook::Position p;
      p.kind = pf::MultiCurveBook::Kind::Swap; p.notional = (i % 2 ? 1.0 : -1.0) * (1.0 + 0.01 * i);
      p.float_coupons = L.flt; p.fixed_coupons = L.fix; p.fwd_curve = i % NC; p.disc_curve = 0; p.fixed_curve = 0; p.fixed_rate = 0.04;
      book.positions.push_back(std::move(p));
    }
  }
};
}  // namespace

TEST(ApiHotPath, OneShotPv01IsTheGradientSumFromOneNarrowPass) {
  const Chain f;
  api::BundleSession s(f.prob);
  s.calibrate(f.x0);
  ASSERT_TRUE(s.result().converged) << s.result().status;
  // Reference: the full-width gradient (one heap Dual per knot), summed.
  const Eigen::VectorXd x = s.x();
  const auto xd = ad::seed(x);
  const auto Cad = cal::build_bundle_curves<ad::Dual>(f.prob.curves, [&](int c, int i) { return xd[f.prob.offset(c) + i]; });
  const auto curve_ad = [&Cad](int i) -> const cal::CurveHandle<ad::Dual>& { return *Cad[i]; };
  const ad::Dual npv_ad = f.book.value<ad::Dual>(curve_ad);
  const double pv01_ref = 1e-4 * npv_ad.derivatives().sum();
  const auto first = s.price_portfolio(f.book);  // warm-up (resolve_book's first pass, telemetry)
  unsigned long allocs = 0;
  api::PortfolioReprice r;
  {
    swaps::testing::AllocScope scope;
    r = s.price_portfolio(f.book);
    allocs = scope.allocs();
  }
  std::cout << "  [hotpath] one-shot price_portfolio: npv " << r.npv << " pv01 " << r.pv01 << " (gradient sum " << pv01_ref
            << ", |diff| " << std::abs(r.pv01 - pv01_ref) << "), " << allocs << " allocs\n";
  EXPECT_NEAR(r.npv, npv_ad.value(), 1e-12 * std::max(1.0, std::abs(r.npv)));
  EXPECT_NEAR(r.pv01, pv01_ref, 1e-12 * std::max(1.0, std::abs(pv01_ref)));
  EXPECT_NEAR(first.pv01, r.pv01, 0.0);
  // PIN (2026-09-10, measured 195): the two curve-handle sets (double + directional) and the seeded vector -- the
  // E3 census measured 71,061 for this one-shot on chain8x26/book200 when the PV01 pass was a heap Dual per
  // knot (208 knots > kPooledMaxW). May only decrease.
  if (swaps::testing::alloc_counting_available()) EXPECT_LE(allocs, 200u) << "the one-shot grew a heap-Dual pass back";
}

TEST(ApiHotPath, RiskOperatorTakesTheTensionBlockFromTheSessionCache) {
  const Chain f;
  api::BundleSession s(f.prob);
  s.calibrate(f.x0, f.tension);
  ASSERT_TRUE(s.result().converged) << s.result().status;
  const Eigen::MatrixXd M1 = s.risk_operator(f.tension);  // warm: J cached, R cached
  unsigned long allocs = 0;
  Eigen::MatrixXd M2;
  {
    swaps::testing::AllocScope scope;
    M2 = s.risk_operator(f.tension);
    allocs = scope.allocs();
  }
  std::cout << "  [hotpath] risk_operator (tension, warm): " << allocs << " allocs\n";
  EXPECT_EQ((M1 - M2).cwiseAbs().maxCoeff(), 0.0) << "same inputs, same operator";
  // Explicit formula off the same J and a freshly built tension block: the cached block is the same block.
  const Eigen::MatrixXd J = s.jacobian(f.tension);
  const Eigen::MatrixXd R = cal::tension_energy_operator(f.prob, f.tension.lambda, f.tension.sigma, f.tension.curves);
  Eigen::MatrixXd S(J.rows() + R.rows(), J.cols()); S << J, R;
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod; cod.setThreshold(cal::kRankThreshold); cod.compute(S);
  const Eigen::MatrixXd Mref = cod.pseudoInverse().leftCols(J.rows()) * s.residual_market_scale().asDiagonal();
  EXPECT_LT((M2 - Mref).cwiseAbs().maxCoeff(), 1e-12 * std::max(1.0, Mref.cwiseAbs().maxCoeff()));
  // PIN (2026-09-10, measured 132): the rank-safe COD's own workspace (item 9 keeps the pseudo-inverse; the E3
  // normal-equation suggestion is not rank-safe) -- E3-D6 measured 4,449 with the block rebuilt per call. May only decrease.
  if (swaps::testing::alloc_counting_available()) EXPECT_LE(allocs, 140u) << "risk_operator rebuilt the tension block";
}
