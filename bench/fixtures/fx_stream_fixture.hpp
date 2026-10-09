// The synthetic USD + EUR + EUR-in-USD xccy bundle behind the usd_eur_xccy_3c8k_* perf metrics (bench/fx_stream_bench.cpp)
// and the fixture-identifiability gate (tests/fixture_identifiability_test.cpp). QuantLib-free; shared so the test that pins
// its identifiability reads the SAME bundle the metrics time.
//
// Three curves, 8 knots each (a 0.25y leading flat knot, then 0.5 .. 10y Hermite): USD outright, EUR outright, EURUSD = EUR + spread.
// Rows: per outright curve 6 annual par swaps (1y..10y) + 3 SHORT single-period par rows (an overnight front pin at 1/360y, 0.25y,
// 0.5y); 7 Portfolio-wrapped FX forwards (0.1y..2y, compiled); 5 MtM xccy basis swaps whose funding coupons carry the compounded
// PRODUCT observation, so they are non-cacheable and ride the AadBlock (the block under test). 30 rows for 24 knots.
//
// WHY THE SHORT ROWS (owner rule, 2026-10-09: a fixture never has fewer instruments than knots, and is identified at its solution).
// Until then USD and EUR had 6 rows for 8 knots: the annual 1y leg telescopes to DF(1y), nothing matured at 0.25y or 0.5y, and the
// FX forwards only see EUR + spread - USD. The 24 x 24 Jacobian had rank 22 plus two weak directions (sigma / sigma_max 4e-8 and
// 1.7e-6), and the streamed tick spent its steps on them: a 2.9 bp quote move produced a 300 bp first frozen step along a direction
// the quotes see at 1e-6, and where the stall refresh then landed decided the tick (6 steps before the kink damper e20a538, 9-10
// after). The overnight row pins the leading flat segment the way a deposit / T/N fixing instrument pins a real curve; the 0.25y and
// 0.5y rows give the two short knots an instrument each. The 5 AAD rows are unchanged, and the fixture aborts if the block empties.
#pragma once

#include <Eigen/Core>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/lm.hpp"

namespace swaps::fxstream {

namespace cal = swaps::calibration;
namespace px = swaps::pricing;

constexpr int USD = 0, EUR = 1, EURUSD = 2;
constexpr double FX_SPOT = 1.10;
constexpr double PAY_LAG = 2.0 / 365.0;  // funding-leg payment lag -> the MtM funding term is NOT negligible
constexpr int kAadRows = 5;              // the compounded-funding MtM rows: the AAD block under test

struct Legs {
  std::vector<px::FloatCoupon> flt;
  std::vector<px::FixedCoupon> fix;
};
// An annual OIS to T as generic legs (one telescoped sub-period per coupon), optional payment lag.
inline Legs annual(double T, double lag = 0.0) {
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

inline cal::Instrument par_inst(double T, int role) {
  Legs L = annual(T);
  cal::Instrument in;
  in.quote = cal::QuoteKind::ParRate;
  in.fwd = {L.flt, role, role};
  in.fixed = {L.fix, role};
  return in;
}

// A SINGLE-PERIOD par row [0, T] on one curve: the front pin (T = one day, the deposit / T/N shape) and the short swaps.
inline cal::Instrument short_inst(double T, int role) {
  px::FloatCoupon c;
  c.obs.sub_start = {0.0};
  c.obs.sub_end = {T};
  c.obs.tau_index = T;
  c.pay = T;
  c.tau_pay = T;
  cal::Instrument in;
  in.quote = cal::QuoteKind::ParRate;
  in.fwd = {{c}, role, role};
  in.fixed = {{{T, T}}, role};
  return in;
}

// FX forward wrapped in a 1-component Portfolio. Since the compiled row model (8f9343d, 2026-09-22) a Portfolio of FX forwards
// COMPILES (an FxRatio source on the W-cache), so these rows ride the compiled engine.
inline cal::Instrument fx_portfolio_inst(double T) {
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

// MtM xccy basis whose USD funding leg pays with a LAG. A plain MtM leg compiles too (the signed-leg batch, 972caab), so to keep
// an AAD block under test the funding coupons carry the compounded PRODUCT observation (single sub-period: the same value, the
// AAD route) -- Instrument::noncacheable answers true for these five rows exactly as in tests/aad_block_pooled_test.cpp. Without
// this the block is EMPTY and the metric measured nothing (it read 1 ns on 2026-10-06; the fixture aborts on an empty block).
inline cal::Instrument mtm_basis_inst(double T) {
  Legs Le = annual(T);           // EUR legs (self + benchmark share the schedule)
  Legs Lu = annual(T, PAY_LAG);  // USD funding leg, payment-lagged
  cal::Instrument in;
  in.quote = cal::QuoteKind::XccyMtmBasis;
  in.fwd = {Le.flt, EURUSD, EURUSD};  // self-forecast (primary = EUR-in-USD)
  in.bench = {Le.flt, EUR, EURUSD};   // EUR benchmark forecast
  in.fixed = {Le.fix, EURUSD};
  in.mtm = {Lu.flt, USD, USD};
  in.mtm.reset_num = EURUSD;
  in.mtm.reset_den = USD;
  in.mtm.fx_spot = FX_SPOT;
  for (auto& c : in.mtm.coupons) c.obs.compounded = true;
  return in;
}

struct Fixture {
  cal::BundleProblem prob;
  Eigen::VectorXd x_true, x0, x_solved, q0, q1;
  std::vector<cal::Instrument> nc;  // the non-cacheable instruments (for the AadBlock-only probes)
  std::vector<int> nc_rows;

  Fixture() {
    const std::vector<double> meet{0.25}, back{0.5, 1, 2, 3, 5, 7, 10};
    prob.curves.resize(3);
    prob.curves[USD] = px::CurveStructure{.base = -1, .currency = 0, .regions = swaps::curve::flat_hermite(meet, back)};
    prob.curves[EUR] = px::CurveStructure{.base = -1, .currency = 1, .regions = swaps::curve::flat_hermite(meet, back)};
    prob.curves[EURUSD] = px::CurveStructure{.base = EUR, .currency = 1, .regions = swaps::curve::flat_hermite(meet, back)};

    for (int role : {USD, EUR}) {
      for (double T : {1.0, 2.0, 3.0, 5.0, 7.0, 10.0}) prob.instruments.push_back(par_inst(T, role));
      for (double T : {1.0 / 360.0, 0.25, 0.5}) prob.instruments.push_back(short_inst(T, role));  // the front pin + the short knots
    }
    for (double T : {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}) prob.instruments.push_back(fx_portfolio_inst(T));
    for (double T : {2.0, 3.0, 5.0, 7.0, 10.0}) prob.instruments.push_back(mtm_basis_inst(T));

    const int N = prob.n_knots();  // 24: 8 knots x 3 curves -> touched width 24 <= kPooledMaxW
    x_true.resize(N);
    auto fill = [&](int c, double lvl, double slope) {
      for (int i = 0; i < prob.curves[c].n_knots(); ++i) x_true[prob.offset(c) + i] = lvl + slope * i;
    };
    fill(USD, 0.0430, 0.0004);
    fill(EUR, 0.0300, 0.0004);
    fill(EURUSD, -0.0015, 0.00002);

    // Self-consistent market = the model quote at x_true (valid for every quote kind here; the FX rows are Portfolio-wrapped,
    // so their residual is the plain q - market, not the log transform).
    set_markets(x_true);

    x0.resize(N);
    x0.segment(prob.offset(USD), prob.curves[USD].n_knots()).setConstant(0.043);
    x0.segment(prob.offset(EUR), prob.curves[EUR].n_knots()).setConstant(0.030);
    x0.segment(prob.offset(EURUSD), prob.curves[EURUSD].n_knots()).setConstant(-0.0015);
    x_solved = cal::calibrate(prob, x0).x;

    // Live-feed pair for the streaming tick: q0 = the anchor market, q1 = a FEASIBLE ~5bp market move (model quotes at a
    // perturbed x, so the frozen-Newton fixed point exists for the FX/MtM couplings).
    q0 = prob.market();
    Eigen::VectorXd xp = x_true;
    for (int k = 0; k < N; ++k) xp[k] += 5e-4 * ((k % 2) ? 1.0 : -1.0);
    q1 = model_quotes(xp);

    // The non-cacheable partition (mirrors HybridBundleResidual's), for the AadBlock-only probes.
    for (int r = 0; r < prob.n_residuals(); ++r)
      if ((prob.instruments[r]).noncacheable()) {
        nc.push_back(prob.instruments[r]);
        nc_rows.push_back(r);
      }
    if (static_cast<int>(nc_rows.size()) != kAadRows) {  // the five compounded-funding MtM rows; an empty block would time nothing
      std::fprintf(stderr, "fx_stream fixture: expected %d AAD-block rows, got %zu -- the metric would measure nothing\n", kAadRows,
                   nc_rows.size());
      std::abort();
    }
  }

  Eigen::VectorXd model_quotes(const Eigen::VectorXd& x) const {
    const auto C = cal::build_bundle_curves<double>(prob.curves, [&](int c, int i) { return x[prob.offset(c) + i]; });
    const auto curve_of = [&C](int i) -> const cal::CurveHandle<double>& { return *C[i]; };
    Eigen::VectorXd q(prob.n_residuals());
    for (int i = 0; i < prob.n_residuals(); ++i) q[i] = cal::instrument_model_quote<double>(prob.instruments[i], curve_of);
    return q;
  }
  void set_markets(const Eigen::VectorXd& x) {
    const Eigen::VectorXd q = model_quotes(x);
    for (int i = 0; i < prob.n_residuals(); ++i) prob.instruments[i].market = q[i];
  }
};

inline const Fixture& fixture() {
  static const Fixture f;
  return f;
}

}  // namespace swaps::fxstream
