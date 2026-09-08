// Portfolio-analytics baseline: our vectorized CompiledPortfolio reprice vs QuantLib's per-swap
// NPV() loop, both off the SAME calibrated curve (gate metric sofr_23k_book1000_ois_reprice; QuantLib is the informational reference).
//
// Repricing a large book as the curve moves is the real-time workload. QuantLib walks every swap's
// coupons through its object model each reprice; we do DF = exp(-Wx) then gathered elementwise
// coupon math + sparse reductions. To force QuantLib to actually re-price (not return cached NPVs)
// we relink the term-structure handle between two structures each iteration.
//
// Run on a quiesced machine (CLAUDE.md §4).

#include <benchmark/benchmark.h>
#include <ql/quantlib.hpp>

#include <Eigen/Core>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/api/bundle_api.hpp"  // BundleSession — the cached (warm) vs one-shot (cold) reprice compare
#include "swaps/calibration/lm.hpp"
#include "swaps/ql/ql_term_structure.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/portfolio/compiled.hpp"
#include "swaps/portfolio/portfolio.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;
namespace rm = swaps::refmkt;

namespace {

constexpr int kBook = 1000;

struct Fixture {
  mutable RelinkableHandle<YieldTermStructure> h;  // relinked during the QuantLib benchmark
  rb::Market mk = rb::build_market(h);
  cal::CalibrationProblem prob = rb::build_square_problem(mk);
  Eigen::VectorXd x = cal::calibrate(prob, Eigen::VectorXd::Constant(prob.n_knots(), 0.035), true).x;

  swaps::curve::ModularCurve<double> curve =
      swaps::curve::make_modular_curve<double>(swaps::curve::flat_hermite(prob.meeting_times, prob.back_times));
  ext::shared_ptr<swaps::qlx::CurveTermStructure<swaps::curve::ModularCurve<double>>> ts_a, ts_b;

  swaps::portfolio::Portfolio pf;
  std::vector<ext::shared_ptr<OvernightIndexedSwap>> ql_book;
  std::unique_ptr<swaps::portfolio::CompiledPortfolio> cp;

  Fixture() {
    curve.set_forwards(x);
    ts_a = ext::make_shared<swaps::qlx::CurveTermStructure<swaps::curve::ModularCurve<double>>>(mk.today, mk.dc, &curve);
    ts_b = ext::make_shared<swaps::qlx::CurveTermStructure<swaps::curve::ModularCurve<double>>>(mk.today, mk.dc, &curve);
    ts_a->enableExtrapolation();
    ts_b->enableExtrapolation();
    h.linkTo(ts_a);

    for (int b = 0; b < kBook; ++b) {
      const auto& s = rm::swaps[b % rm::swaps.size()];
      const double fixed = s.par_rate + 0.0005 * ((b % 7) - 3);
      const double notl = ((b % 2) ? 1.0 : -1.0) * (1.0 + (b % 5));
      ext::shared_ptr<OvernightIndexedSwap> sw =
          MakeOIS(Period(s.tenor_years, Years), mk.sofr, fixed).withNominal(notl).withDiscountingTermStructure(h);
      ql_book.push_back(sw);
      pf.positions.push_back({swaps::qlx::extract_float_leg(sw->overnightLeg(), mk.today, mk.dc),
                              swaps::qlx::extract_fixed_leg(sw->fixedLeg(), mk.today, mk.dc), fixed,
                              notl});
    }
    cp = std::make_unique<swaps::portfolio::CompiledPortfolio>(swaps::curve::flat_hermite(prob.meeting_times, prob.back_times), pf);
  }
};

const Fixture& fx() {
  static const Fixture f;
  return f;
}

}  // namespace

static void BM_Portfolio_QuantLib(benchmark::State& state) {
  const auto& f = fx();
  bool flip = false;
  for (auto _ : state) {
    f.h.linkTo(flip ? f.ts_a : f.ts_b);  // invalidate every swap's cached NPV
    flip = !flip;
    double total = 0;
    for (const auto& s : f.ql_book) total += s->NPV();
    benchmark::DoNotOptimize(total);
  }
}
BENCHMARK(BM_Portfolio_QuantLib);

static void BM_Portfolio_Ours(benchmark::State& state) {
  const auto& f = fx();
  for (auto _ : state) {
    double total = f.cp->total_npv(f.x);
    benchmark::DoNotOptimize(total);
  }
}
BENCHMARK(BM_Portfolio_Ours);

// ============================================================================================
// SESSION reprice: the AMORTIZED (warm/streaming) path vs the one-shot COLD path, at desk scale.
//
// The audit-U2 lesson (a reverted regression): wiring the compiled multi-curve book into the ONE-SHOT
// price_portfolio was a NET LOSS — it sped up the NPV pass (not the bottleneck) while ADDING a per-call
// W-cache build. The win exists ONLY when that build is AMORTIZED across repeated repricings, i.e. the
// STREAMING path where the SAME book is repriced every tick against a recalibrating curve. These two
// benches make that concrete on an 8-curve / 26-knot bundle and the 200-swap multi-curve book of the
// gated BM_Session_PricePortfolio:
//   * COLD (BM_Session_Reprice_Cold) = session.price_portfolio(book) per tick — builds the double curve
//     handles + walks the virtual CurveHandle kernel + one AAD PV01 pass EVERY call. Unchanged shipped
//     path; must NOT regress (it is the gated baseline).
//   * WARM (BM_Session_Reprice_Warm) = bind_portfolio(book) ONCE, then reprice_bound() per tick — the
//     compiled W-cache twin (DF = exp(-W_all x) matvec + gathered reduce) + an ANALYTIC parallel PV01,
//     the W built once at bind. This is the number that must BEAT cold to justify the compiled kernel.
// QuantLib-free (ours-only); shares this binary with the QuantLib benches above but touches none of them.
namespace sess_bench {

constexpr int NC = 8;           // 1 outright + 7 spread curves (a spread chain)
constexpr int NK = 26;          // knots per curve
constexpr double MAX_T = 30.0;  // longest tenor (years)

namespace api = swaps::api;
namespace px = swaps::pricing;

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

struct Fixture {
  cal::BundleProblem prob;
  Eigen::VectorXd x0;
  swaps::portfolio::MultiCurveBook book;

  Fixture() {
    std::vector<double> meeting{0.25}, back;
    for (int i = 1; i <= NK - 1; ++i) back.push_back(MAX_T * i / (NK - 1));
    prob.curves.resize(NC);
    prob.curves[0] = px::CurveStructure{.base = -1, .regions = swaps::curve::flat_hermite(meeting, back)};
    for (int c = 1; c < NC; ++c)
      prob.curves[c] = px::CurveStructure{.base = c - 1, .regions = swaps::curve::flat_hermite(meeting, back)};

    std::vector<double> mats;
    for (double T = 1.0; T <= MAX_T + 1e-9; T += 1.0) mats.push_back(T);
    for (double T : mats) prob.instruments.push_back(par_inst(T, 0, 0));
    for (int c = 1; c < NC; ++c)
      for (double T : mats) prob.instruments.push_back(basis_inst(T, c, c - 1, 0));

    Eigen::VectorXd x_true(NC * NK);
    for (int c = 0; c < NC; ++c)
      for (int i = 0; i < NK; ++i)
        x_true[c * NK + i] = (c == 0) ? 0.040 + 0.0005 * i : 0.0020 + 0.0001 * i;
    const Eigen::VectorXd r0 = prob.residuals<double>(x_true);
    for (int i = 0; i < static_cast<int>(prob.instruments.size()); ++i) prob.instruments[i].market += r0[i];

    x0.resize(NC * NK);
    for (int c = 0; c < NC; ++c)
      for (int i = 0; i < NK; ++i) x0[c * NK + i] = (c == 0) ? 0.040 : 0.0020;

    // The 200-swap multi-curve book of BM_Session_PricePortfolio.
    for (int i = 0; i < 200; ++i) {
      const double T = 1.0 + (i % 30);
      Legs L = annual(T);
      swaps::portfolio::MultiCurveBook::Position p;
      p.kind = swaps::portfolio::MultiCurveBook::Kind::Swap;
      p.notional = (i % 2 ? 1.0 : -1.0) * (1.0 + 0.01 * i);
      p.float_coupons = L.flt;
      p.fixed_coupons = L.fix;
      p.fwd_curve = i % NC;
      p.disc_curve = 0;
      p.fixed_curve = 0;
      p.fixed_rate = 0.04;
      book.positions.push_back(std::move(p));
    }
  }
};

const Fixture& fx() {
  static const Fixture f;
  return f;
}

}  // namespace sess_bench

// COLD: the shipped one-shot path repeated (builds curves + AAD PV01 every call). The gated baseline.
static void BM_Session_Reprice_Cold(benchmark::State& state) {
  const auto& f = sess_bench::fx();
  sess_bench::api::BundleSession sess(f.prob);
  sess.calibrate(f.x0);
  for (auto _ : state) {
    sess_bench::api::PortfolioReprice r = sess.price_portfolio(f.book);
    benchmark::DoNotOptimize(r.npv);
    benchmark::DoNotOptimize(r.pv01);
  }
}
BENCHMARK(BM_Session_Reprice_Cold);

// WARM: bind ONCE (amortizes the W build), then reprice the cached compiled twin every tick. Must beat cold.
static void BM_Session_Reprice_Warm(benchmark::State& state) {
  const auto& f = sess_bench::fx();
  sess_bench::api::BundleSession sess(f.prob);
  sess.calibrate(f.x0);
  sess.bind_portfolio(f.book);  // build the compiled W-cache twin ONCE, outside the loop
  for (auto _ : state) {
    sess_bench::api::PortfolioReprice r = sess.reprice_bound();
    benchmark::DoNotOptimize(r.npv);
    benchmark::DoNotOptimize(r.pv01);
  }
}
BENCHMARK(BM_Session_Reprice_Warm);

BENCHMARK_MAIN();
