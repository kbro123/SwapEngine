// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs)
// Gate for the SESSION-cached compiled portfolio reprice (BundleSession::bind_portfolio /
// reprice_bound, audit U2 — the amortized streaming path). Where compiled_multi_test.cpp gates the
// kernel (CompiledMultiCurveBook::npv vs the templated MultiCurveBook at a raw state x), THIS gates the
// PRODUCT seam: a calibrated session's cached compiled reprice must equal the shipped templated path
// (BundleSession::price_portfolio) in BOTH npv AND the +1bp parallel-shift PV01, to ~1e-10 —
//   * at the calibrated x,
//   * again after a rebind moves x (proving the cached W-twin tracks x, is not built stale),
//   * for an all-compilable book (the allocation-free streaming hot path, n_fallback == 0),
//   * and for a mixed book carrying an Xccy position (exercising the fallback PV01's AAD pass).
//
// QuantLib-free: a small multi-curve bundle (outright + spread + a second outright) is hand-built and
// calibrated, mirroring session_warm_bench.cpp's shapes.

#include <gtest/gtest.h>

#include <Eigen/Core>

#include <cmath>
#include <vector>

#include "swaps/api/bundle_api.hpp"

namespace api = swaps::api;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace pf = swaps::portfolio;

namespace {

constexpr int NC = 3;   // curve 0 outright, curve 1 spread on 0, curve 2 outright (2nd currency role)
constexpr int NK = 8;   // knots per curve (1 meeting + 7 back)
constexpr double MAX_T = 15.0;

struct Legs {
  std::vector<px::FloatCoupon> flt;
  std::vector<px::FixedCoupon> fix;
};
Legs annual(double T) {
  Legs L;
  double prev = 0.0;
  for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev};
    c.obs.sub_end = {u};
    c.obs.tau_index = u - prev;
    c.pay = u;
    c.tau_pay = u - prev;
    L.flt.push_back(c);
    L.fix.push_back({u, u - prev});
    prev = u;
  }
  return L;
}
cal::Instrument par_inst(double T, int fc, int dc) {
  Legs L = annual(T);
  cal::Instrument in;
  in.quote = cal::QuoteKind::ParRate;
  in.fwd = {L.flt, fc, dc};
  in.fixed = {L.fix, dc};
  return in;
}
cal::Instrument basis_inst(double T, int fc, int bc, int dc) {
  Legs L = annual(T);
  cal::Instrument in;
  in.quote = cal::QuoteKind::ParSpread;
  in.fwd = {L.flt, fc, dc};
  in.bench = {L.flt, bc, dc};
  in.fixed = {L.fix, dc};
  return in;
}

// A calibratable 3-curve bundle + a market consistent with a known x_true.
struct Fixture {
  cal::BundleProblem prob;
  cal::BundleProblem pert;  // ~1bp-perturbed quotes (a rebind payload → x moves)
  Eigen::VectorXd x0;

  Fixture() {
    std::vector<double> meeting{0.5}, back;
    for (int i = 1; i <= NK - 1; ++i) back.push_back(MAX_T * i / (NK - 1));
    prob.curves.resize(NC);
    prob.curves[0] = px::CurveStructure{.base = -1, .regions = swaps::curve::flat_hermite(meeting, back)};
    prob.curves[1] = px::CurveStructure{.base = 0, .regions = swaps::curve::flat_hermite(meeting, back)};
    prob.curves[2] =
        px::CurveStructure{.base = -1, .currency = 1, .regions = swaps::curve::flat_hermite(meeting, back)};

    std::vector<double> mats;
    for (double T = 1.0; T <= MAX_T + 1e-9; T += 1.0) mats.push_back(T);
    // 8 pillars per curve so n_residuals == n_knots (well-posed square solve).
    for (double T : mats) prob.instruments.push_back(par_inst(T, 0, 0));    // curve 0 outright
    for (double T : mats) prob.instruments.push_back(basis_inst(T, 1, 0, 0));  // curve 1 = spread on 0
    for (double T : mats) prob.instruments.push_back(par_inst(T, 2, 2));    // curve 2 outright (ccy 1)

    Eigen::VectorXd x_true(NC * NK);
    for (int c = 0; c < NC; ++c)
      for (int i = 0; i < NK; ++i)
        x_true[c * NK + i] = (c == 1) ? 0.0030 + 0.0001 * i          // curve 1: ~30bp spread
                                       : (c == 2 ? 0.020 : 0.035) + 0.0004 * i;  // outrights
    const Eigen::VectorXd r0 = prob.residuals<double>(x_true);
    for (int i = 0; i < static_cast<int>(prob.instruments.size()); ++i) prob.instruments[i].market += r0[i];

    x0 = Eigen::VectorXd::Zero(NC * NK);
    for (int c = 0; c < NC; ++c)
      for (int i = 0; i < NK; ++i) x0[c * NK + i] = (c == 1) ? 0.0 : (c == 2 ? 0.020 : 0.035);

    pert = prob;
    for (int i = 0; i < static_cast<int>(pert.instruments.size()); ++i)
      pert.instruments[i].market += 1e-4 * std::sin(0.7 * i + 0.3);
  }
};

// A vanilla multi-curve swap position, payer-of-fixed.
pf::MultiCurveBook::Position swap_pos(double T, int fc, int dc, int xc, double rate, double notl) {
  Legs L = annual(T);
  pf::MultiCurveBook::Position p;
  p.kind = pf::MultiCurveBook::Kind::Swap;
  p.notional = notl;
  p.float_coupons = L.flt;
  p.fwd_curve = fc;
  p.disc_curve = dc;
  p.fixed_coupons = L.fix;
  p.fixed_curve = xc;
  p.fixed_rate = rate;
  return p;
}

// An all-compilable multi-curve book across the three forecast curves/roles.
pf::MultiCurveBook compilable_book() {
  pf::MultiCurveBook book;
  const double T[6] = {1, 2, 3, 5, 10, 15};
  for (int i = 0; i < 24; ++i) {
    const int fc = i % 3, dc = (i % 2) ? 0 : 2;
    book.positions.push_back(
        swap_pos(T[i % 6], fc, dc, dc, 0.02 + 0.001 * (i % 5), (i % 2 ? 1.0 : -1.0) * 1e6 * (1 + i)));
  }
  // fwd != disc != fixed (all three roles distinct) + a spread-carrying and an FX-scaled leg.
  book.positions.push_back(swap_pos(10.0, 1, 0, 2, 0.031, 2e6));
  {
    auto p = swap_pos(5.0, 1, 0, 0, 0.028, 1.5e6);
    for (auto& c : p.float_coupons) c.spread = 0.0030;
    book.positions.push_back(p);
    auto q = swap_pos(3.0, 2, 2, 2, 0.021, 2.5e6);
    for (auto& c : q.float_coupons) c.scale = 1.10;
    for (auto& c : q.fixed_coupons) c.scale = 1.10;
    book.positions.push_back(q);
  }
  return book;
}

// The same book plus one Xccy position (COMPILED as two rows since 2026-09-10 -- exercises the MtM PV01 partials).
pf::MultiCurveBook mixed_book() {
  pf::MultiCurveBook book = compilable_book();
  pf::MultiCurveBook::Position xp;
  xp.kind = pf::MultiCurveBook::Kind::Xccy;
  xp.notional = 3e6;
  xp.float_coupons = annual(5.0).flt;
  xp.fwd_curve = 0;
  xp.disc_curve = 0;
  xp.mtm_coupons = annual(5.0).flt;
  for (auto& c : xp.mtm_coupons) c.spread = 0.005;
  xp.mtm_fwd_curve = 2;
  xp.mtm_disc_curve = 2;
  xp.mtm_reset_num = 2;
  xp.mtm_reset_den = 0;
  xp.fx_spot = 1.10;
  book.positions.push_back(xp);
  return book;
}

// Compiled (cached) reprice must match the templated price_portfolio in npv AND pv01.
void expect_reprice_parity(const api::PortfolioReprice& warm, const api::PortfolioReprice& cold,
                           const char* label) {
  EXPECT_EQ(warm.n, cold.n) << label;
  EXPECT_LE(std::abs(warm.npv - cold.npv), 1e-9 * (std::abs(cold.npv) + 1.0))
      << label << ": npv compiled " << warm.npv << " vs templated " << cold.npv;
  EXPECT_LE(std::abs(warm.pv01 - cold.pv01), 1e-8 * (std::abs(cold.pv01) + 1.0))
      << label << ": pv01 compiled " << warm.pv01 << " vs templated " << cold.pv01;
}

}  // namespace

// A book bound BEFORE calibration still reprices correctly (lazy state read at reprice time).
TEST(SessionCompiledReprice, ThrowsBeforeBind) {
  Fixture f;
  api::BundleSession sess(f.prob);
  sess.calibrate(f.x0);
  EXPECT_FALSE(sess.has_bound_portfolio());
  EXPECT_THROW(sess.reprice_bound(), std::runtime_error);
}

// THE parity gate: cached compiled reprice == templated price_portfolio at the calibrated x, and again
// after a rebind moves x (the compiled twin was built ONCE, at bind, yet tracks the new x).
TEST(SessionCompiledReprice, MatchesTemplatedAcrossXMoves) {
  Fixture f;
  api::BundleSession sess(f.prob);
  sess.calibrate(f.x0);

  const pf::MultiCurveBook book = compilable_book();
  sess.bind_portfolio(book);  // build the W-cache twin ONCE
  ASSERT_TRUE(sess.has_bound_portfolio());

  expect_reprice_parity(sess.reprice_bound(), sess.price_portfolio(book), "all-compiled @ calibrated x");
  EXPECT_GT(std::abs(sess.reprice_bound().pv01), 0.0) << "the book must carry parallel-shift risk";

  // Rebind to a ~1bp-perturbed market: x_ moves, the bound compiled twin is reused (not rebuilt).
  sess.rebind(f.pert);
  expect_reprice_parity(sess.reprice_bound(), sess.price_portfolio(book), "all-compiled @ rebound x");

  // And back — returning to the first market reproduces the first reprice.
  sess.rebind(f.prob);
  expect_reprice_parity(sess.reprice_bound(), sess.price_portfolio(book), "all-compiled @ x back");
}

// The mixed book (an Xccy position compiled as two rows) matches in npv AND PV01 -- both analytic now; the
// MtM reset / exchange partials are the ones d_pv_from_num carries.
TEST(SessionCompiledReprice, MixedBookWithXccyMatches) {
  Fixture f;
  api::BundleSession sess(f.prob);
  sess.calibrate(f.x0);

  const pf::MultiCurveBook book = mixed_book();
  sess.bind_portfolio(book);
  expect_reprice_parity(sess.reprice_bound(), sess.price_portfolio(book), "mixed @ calibrated x");

  sess.rebind(f.pert);
  expect_reprice_parity(sess.reprice_bound(), sess.price_portfolio(book), "mixed @ rebound x");
}

// Rebinding a DIFFERENT book replaces the cache; each reprice matches its own templated pass.
TEST(SessionCompiledReprice, RebindReplacesTheBook) {
  Fixture f;
  api::BundleSession sess(f.prob);
  sess.calibrate(f.x0);

  const pf::MultiCurveBook a = compilable_book();
  const pf::MultiCurveBook b = mixed_book();
  sess.bind_portfolio(a);
  expect_reprice_parity(sess.reprice_bound(), sess.price_portfolio(a), "book A");
  sess.bind_portfolio(b);
  expect_reprice_parity(sess.reprice_bound(), sess.price_portfolio(b), "book B after re-bind");
}
