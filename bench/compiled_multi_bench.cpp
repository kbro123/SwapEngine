// Compiled multi-curve book reprice vs the templated virtual-handle path (audit U2).
//
// The fixture mirrors session_warm_bench.cpp's desk scale: an 8-curve spread chain (1 outright + 7
// spreads), 26 knots per curve, and a 200-swap multi-curve book spread across all 8 forecast curves --
// the SAME book shape as the gated BM_Session_PricePortfolio baseline. Both benches flip between two
// nearby curve states every iteration so neither path can serve a cached result: the templated side
// set_forwards's its reused handles and walks the virtual CurveHandle kernel (what price_portfolio does
// today); the compiled side is one DF = exp(-W_all x) matvec + the gathered coupon/annuity reduce.
//
// QuantLib-free, ours-only. Run on a quiesced machine (CLAUDE.md §4).

#include <benchmark/benchmark.h>

#include <Eigen/Core>

#include <cmath>
#include <vector>

#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/portfolio/compiled_multi.hpp"
#include "swaps/portfolio/portfolio.hpp"

namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace pf = swaps::portfolio;

namespace {

constexpr int NC = 8;           // 1 outright + 7 spread curves (a spread chain)
constexpr int NK = 26;          // knots per curve
constexpr double MAX_T = 30.0;  // longest tenor (years)

px::FloatCoupon ois_coupon(double a, double b) {
  px::FloatCoupon c;
  c.obs.sub_start = {a};
  c.obs.sub_end = {b};
  c.obs.tau_index = b - a;
  c.pay = b;
  c.tau_pay = b - a;
  return c;
}

struct Fixture {
  std::vector<px::CurveStructure> curves;
  pf::MultiCurveBook book;
  Eigen::VectorXd x0, x1;  // two nearby states, alternated per iteration

  Fixture() {
    std::vector<double> meeting{0.25}, back;
    for (int i = 1; i <= NK - 1; ++i) back.push_back(MAX_T * i / (NK - 1));
    curves.resize(NC);
    curves[0] = px::CurveStructure{.base = -1, .regions = swaps::curve::flat_hermite(meeting, back)};
    for (int c = 1; c < NC; ++c)
      curves[c] = px::CurveStructure{.base = c - 1, .regions = swaps::curve::flat_hermite(meeting, back)};

    // The 200-swap multi-curve book of BM_Session_PricePortfolio: annual legs, forecast curve i % NC,
    // everything discounted on curve 0, alternating payer/receiver.
    for (int i = 0; i < 200; ++i) {
      const double T = 1.0 + (i % 30);
      pf::MultiCurveBook::Position p;
      p.kind = pf::MultiCurveBook::Kind::Swap;
      p.notional = (i % 2 ? 1.0 : -1.0) * (1.0 + 0.01 * i);
      double prev = 0.0;
      for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
        p.float_coupons.push_back(ois_coupon(prev, u));
        p.fixed_coupons.push_back({u, u - prev});
        prev = u;
      }
      p.fwd_curve = i % NC;
      p.disc_curve = 0;
      p.fixed_curve = 0;
      p.fixed_rate = 0.04;
      book.positions.push_back(std::move(p));
    }

    x0.resize(NC * NK);
    for (int c = 0; c < NC; ++c)
      for (int i = 0; i < NK; ++i)
        x0[c * NK + i] = (c == 0) ? 0.040 + 0.0005 * i : 0.0020 + 0.0001 * i;
    x1 = x0;
    for (int i = 0; i < x1.size(); ++i) x1[i] += 1e-4 * std::sin(0.7 * i + 0.3);  // ~1bp move
  }
};

const Fixture& fx() {
  static const Fixture f;
  return f;
}

}  // namespace

// TODAY's path: the templated virtual-CurveHandle kernel behind BundleSession::price_portfolio. The
// handles are built ONCE and set_forwards'd per tick (charitable: the session also rebuilds nothing),
// so what is measured is purely the per-coupon virtual discount()/spread-chain recursion.
static void BM_MultiCurveBook_Templated(benchmark::State& state) {
  const Fixture& f = fx();
  cal::BundleCurveSet<double> curves;
  curves.build(f.curves);
  const auto cof = [&curves](int i) -> const cal::CurveHandle<double>& { return curves[i]; };
  bool flip = false;
  for (auto _ : state) {
    const Eigen::VectorXd& x = flip ? f.x1 : f.x0;
    flip = !flip;
    curves.update([&](int c, int i) { return x[c * NK + i]; });
    double npv = f.book.value<double>(cof);
    benchmark::DoNotOptimize(npv);
  }
}
BENCHMARK(BM_MultiCurveBook_Templated);

// The COMPILED kernel: DF_all = exp(-W_all x) once, then the role-aware gathered coupon/annuity reduce.
static void BM_MultiCurveBook_Compiled(benchmark::State& state) {
  const Fixture& f = fx();
  const pf::CompiledMultiCurveBook cmb(f.curves, f.book);
  bool flip = false;
  for (auto _ : state) {
    double npv = cmb.npv(flip ? f.x1 : f.x0);
    flip = !flip;
    benchmark::DoNotOptimize(npv);
  }
}
BENCHMARK(BM_MultiCurveBook_Compiled);

// One-time construction cost (W build + batch registration) -- context, not a gate: it is paid once per
// book/curve-structure, then every tick rides the reprice above.
static void BM_MultiCurveBook_CompileCost(benchmark::State& state) {
  const Fixture& f = fx();
  for (auto _ : state) {
    const pf::CompiledMultiCurveBook cmb(f.curves, f.book);
    benchmark::DoNotOptimize(cmb.n_times());
  }
}
BENCHMARK(BM_MultiCurveBook_CompileCost);
