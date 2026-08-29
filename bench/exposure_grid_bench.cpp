// MC-exposure kernel throughput (hot-path design R3+R12, VOL_XVA_ROADMAP §2.1): reprice a whole swap book at
// a GRID of curve-states X (n_knots x n_states) via CompiledPortfolio::npv_grid — one W·X GEMM discounts every
// simulated state, then the cheap sparse coupon/annuity reduce runs per state. This IS "our microsecond book
// reprice in a loop": the number a desk quotes for exposure-sim wall-clock (EPE/ENE/PFE). Engine-only, ours-
// only, no QuantLib, no JSON. Target (design): 10k paths x 100 nodes in < 1 s  => > 1M node-reprices/sec.
//
// Reported metric: items/sec == node-reprices/sec (SetItemsProcessed). Column 0 is pinned to the calibrated
// state and asserted bit-identical to the serial single-state npv(x) — same kernel, so exact equality holds.
#include <benchmark/benchmark.h>

#include <Eigen/Core>

#include <cstdlib>
#include <random>
#include <vector>

#include "swaps/portfolio/compiled.hpp"
#include "swaps/portfolio/portfolio.hpp"
#include "swaps/xva/exposure.hpp"

namespace pf = swaps::portfolio;
namespace xva = swaps::xva;

namespace {
const std::vector<double> kMeeting{0.5};
const std::vector<double> kBack{1, 2, 3, 5, 7, 10, 15, 20, 30};  // 10 knots total

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
Eigen::VectorXd x_calibrated() {
  Eigen::VectorXd x(kMeeting.size() + kBack.size());
  for (int i = 0; i < x.size(); ++i) x[i] = 0.030 + 0.0008 * i;
  return x;
}

// A grid of simulated curve-states around the calibrated one (a Gaussian proxy for LGM/HW1F-evolved x — the
// kernel is indifferent to where x comes from). Column 0 is pinned to x_cal for the exactness check.
Eigen::MatrixXd make_grid(int n_states) {
  const int nk = static_cast<int>(kMeeting.size() + kBack.size());
  const Eigen::VectorXd x_cal = x_calibrated();
  Eigen::MatrixXd X(nk, n_states);
  std::mt19937 rng(0xE7050);
  std::normal_distribution<double> shock(0.0, 0.0050);  // 50 bp curve-state diffusion
  for (int j = 0; j < n_states; ++j)
    for (int i = 0; i < nk; ++i) X(i, j) = x_cal[i] + shock(rng);
  X.col(0) = x_cal;
  return X;
}
}  // namespace

static void BM_ExposureGrid(benchmark::State& state) {
  const int n_states = static_cast<int>(state.range(0));  // e.g. paths * time-nodes
  const pf::CompiledPortfolio book(swaps::curve::flat_hermite(kMeeting, kBack), make_book(50));
  const Eigen::MatrixXd X = make_grid(n_states);

  // Validate: the batched grid must reproduce the serial single-state reprice on the pinned column. Not
  // bit-exact — the batch uses a GEMM (W·X) vs the single-state GEMV (W·x), which round differently in the
  // last ULP — so check a tight RELATIVE tolerance (NPVs are ~notional in scale).
  const Eigen::VectorXd ref = book.npv(x_calibrated());
  const double err = (book.npv_grid(X).col(0) - ref).cwiseAbs().maxCoeff();
  const double scale = std::max(1.0, ref.cwiseAbs().maxCoeff());
  if (err > 1e-9 * scale) {
    state.SkipWithError(("npv_grid col 0 vs single-state: rel err " + std::to_string(err / scale)).c_str());
    return;
  }
  for (auto _ : state) benchmark::DoNotOptimize(book.npv_grid(X).data());
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n_states));  // node-reprices/sec
}
BENCHMARK(BM_ExposureGrid)->Arg(10000)->Arg(100000)->Unit(benchmark::kMillisecond);

// End-to-end EPE/ENE/PFE profile: a Gaussian curve-state proxy grid (node-major; node 0 = today so every path
// shares x_cal and the profile there is deterministic) -> npv_grid -> exposure_profile. Measures the full
// exposure-sim wall-clock (reprice + net + per-node reduce) — the number a desk quotes for a nightly/intraday
// counterparty run. n_paths x n_nodes states; column (path p, node j) at j*n_paths + p.
static Eigen::MatrixXd profile_grid(int n_paths, int n_nodes, const std::vector<double>& t) {
  const int nk = static_cast<int>(kMeeting.size() + kBack.size());
  const Eigen::VectorXd x_cal = x_calibrated();
  Eigen::MatrixXd X(nk, n_paths * n_nodes);
  std::mt19937 rng(0x5E0Fu);
  std::normal_distribution<double> z(0.0, 1.0);
  const double sigma = 0.010;  // 100 bp/yr curve-state vol (illustrative; not a calibrated LGM)
  for (int j = 0; j < n_nodes; ++j) {
    const double sd = sigma * std::sqrt(t[j]);  // Brownian: node 0 (t=0) -> sd 0 -> deterministic
    for (int p = 0; p < n_paths; ++p)
      for (int i = 0; i < nk; ++i) X(i, j * n_paths + p) = x_cal[i] + (sd > 0.0 ? sd * z(rng) : 0.0);
  }
  return X;
}

static void BM_ExposureProfile(benchmark::State& state) {
  const int n_paths = 1000, n_nodes = 100;
  std::vector<double> t(n_nodes);
  for (int j = 0; j < n_nodes; ++j) t[j] = 30.0 * j / (n_nodes - 1);  // 0..30y horizon
  const pf::CompiledPortfolio book(swaps::curve::flat_hermite(kMeeting, kBack), make_book(50));
  const Eigen::MatrixXd X = profile_grid(n_paths, n_nodes, t);

  // Validate: node 0 is today (deterministic), so EPE(0) == max(book MtM at x_cal, 0), PFE(0) == that MtM.
  const double mtm = book.total_npv(x_calibrated());
  const xva::ExposureProfile chk = xva::exposure_profile(book.npv_grid(X), n_paths, n_nodes, t);
  const double scale = std::max(1.0, std::abs(mtm));
  if (std::abs(chk.epe[0] - std::max(mtm, 0.0)) > 1e-9 * scale ||
      std::abs(chk.pfe[0] - mtm) > 1e-9 * scale) {
    state.SkipWithError("exposure profile node-0 != deterministic MtM");
    return;
  }
  for (auto _ : state) {
    const xva::ExposureProfile pr = xva::exposure_profile(book.npv_grid(X), n_paths, n_nodes, t);
    benchmark::DoNotOptimize(pr.epe.data());
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n_paths) * n_nodes);
}
BENCHMARK(BM_ExposureProfile)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
