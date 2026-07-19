// Coherent parallel portfolio reprice (Q2): serial full-book CompiledPortfolio vs an N-slice
// ParallelPortfolio, all off ONE pinned curve. Engine-only, ours-only probe (not a gated metric); the
// point is the reprice speedup on a large book and that the split is bit-identical (proven in tests).
#include <benchmark/benchmark.h>

#include <Eigen/Core>

#include <vector>

#include "swaps/portfolio/compiled.hpp"
#include "swaps/portfolio/parallel.hpp"
#include "swaps/portfolio/portfolio.hpp"

namespace pf = swaps::portfolio;

namespace {
const std::vector<double> kMeeting{0.5};
const std::vector<double> kBack{1, 2, 3, 5, 7, 10, 15, 20, 30};

pf::Portfolio make_book(int P) {
  pf::Portfolio book;
  for (int p = 0; p < P; ++p) {
    pf::Portfolio::Position pos;
    const double T = 1.0 + (p % 30);
    double prev = 0.0;
    for (double u = 1.0; u <= T + 1e-9; u += 1.0) {
      swaps::pricing::FloatCoupon c;
      c.obs.sub_start = {prev};
      c.obs.sub_end = {u};
      c.obs.tau_index = u - prev;
      c.pay = u;
      c.tau_pay = u - prev;
      pos.float_coupons.push_back(c);
      pos.fixed_coupons.push_back({u, u - prev});
      prev = u;
    }
    pos.fixed_rate = 0.03 + 0.0001 * (p % 20);
    pos.notional = 1.0e6 * (1 + p % 5);
    book.positions.push_back(pos);
  }
  return book;
}
Eigen::VectorXd forwards() {
  Eigen::VectorXd x(kMeeting.size() + kBack.size());
  for (int i = 0; i < x.size(); ++i) x[i] = 0.030 + 0.0008 * i;
  return x;
}
constexpr int kBookSize = 20000;
}  // namespace

static void BM_PortfolioReprice_Serial(benchmark::State& s) {
  const pf::CompiledPortfolio book(kMeeting, kBack, make_book(kBookSize));
  const Eigen::VectorXd x = forwards();
  for (auto _ : s) benchmark::DoNotOptimize(book.npv(x).data());
}
BENCHMARK(BM_PortfolioReprice_Serial)->Unit(benchmark::kMicrosecond);

static void BM_PortfolioReprice_Parallel8(benchmark::State& s) {
  const pf::ParallelPortfolio book(kMeeting, kBack, make_book(kBookSize), 8);
  const Eigen::VectorXd x = forwards();
  for (auto _ : s) benchmark::DoNotOptimize(book.reprice(x).data());
}
BENCHMARK(BM_PortfolioReprice_Parallel8)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
