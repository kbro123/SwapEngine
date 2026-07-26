// Simulate a full real-time trading day on the realistic 4-curve bundle (SOFR + FF + PRIME + PRIME2) and
// emit JSON for visualization. 1 tick/second for a 23-hour day = 82,800 ticks. Each tick the DRIVING
// curve random-walks by a LEVEL + gentle SLOPE shift (per-tick move ~ mean 0.15bp / sd 0.25bp, random
// sign -> a mean-zero walk); because level+slope leave 2nd differences invariant, butterflies stay stable
// and intermediate knots move sensibly with their neighbours. Quotes are generated from that curve (so
// they are always achievable), the StreamingCalibrator recovers it each tick, and we record the
// calibration time (min/max/avg) plus periodic forward snapshots on a knot-aware grid.
//
// Build (not in CMake; links the vendored QuantLib static lib, like tools/stream_sim.cpp). If the CLT
// libc++ headers are missing (CLT 16.2 breakage, see CLAUDE.md §4), point at the active SDK's libc++ first:
//   export CPLUS_INCLUDE_PATH="$(xcrun --show-sdk-path)/usr/include/c++/v1"
//   c++ -std=c++20 -O3 -march=native -DEIGEN_ENABLE_AVX512 \
//       -I include -I tests -I third_party/eigen -I third_party/boost -I third_party/quantlib/install/include \
//       tools/bundle_day_sim.cpp third_party/quantlib/install/lib/libQuantLib.a -o bundle_day_sim
//   ./bundle_day_sim            # writes bundle_day.json in the cwd
#include <ql/quantlib.hpp>

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <vector>

#include "reference_bundle.hpp"
#include "reference_curve.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;

int main() {
  RelinkableHandle<YieldTermStructure> h;
  rb::Market mk = rb::build_market(h);
  rb::RealisticBundle b = rb::build_realistic_bundle(mk, h, /*n_basis=*/3, rb::BundleTopology::Chain);
  const cal::BundleProblem& prob = b.prob;
  const int NC = b.n_curves();
  const int N = prob.n_knots();
  const char* names[] = {"SOFR", "FF", "PRIME", "PRIME2"};

  // Start from the calibrated solution; drive a "true" curve that random-walks; feed model quotes from it.
  cal::CompiledBundleResidual engine(prob);
  Eigen::VectorXd x_true = cal::calibrate(prob, b.x0, true).x;
  const Eigen::VectorXd q0 = engine.model_rates(x_true);
  cal::StreamingCalibrator<cal::BundleProblem> sc(prob, x_true, q0,
                                                  cal::StreamingCalibrator<cal::BundleProblem>::Options{});

  // Per-curve knot times (for the level/slope shift normalization and the display grid).
  std::vector<std::vector<double>> knots(NC);
  double maxT = 0;
  for (int c = 0; c < NC; ++c) {
    for (double t : prob.curves[c].meeting) knots[c].push_back(t);
    for (double t : prob.curves[c].back) knots[c].push_back(t);
    for (double t : knots[c]) maxT = std::max(maxT, t);
  }

  // ---- Display grid: a day either side of every knot (so the flat-forward jumps at meeting dates render
  // sharp) and weekly points in between. Union across curves, sorted & de-duplicated. ----
  const double DAY = 1.0 / 365.0, WEEK = 7.0 / 365.0;
  std::vector<double> allk;
  for (int c = 0; c < NC; ++c)
    for (double t : knots[c]) allk.push_back(t);
  std::sort(allk.begin(), allk.end());
  allk.erase(std::unique(allk.begin(), allk.end(), [](double a, double b) { return std::abs(a - b) < 1e-9; }),
             allk.end());
  std::vector<double> grid{DAY};
  for (std::size_t k = 0; k < allk.size(); ++k) {
    const double kt = allk[k];
    if (kt <= 1.6) {  // the piecewise-FLAT front: a day either side pins the sharp jump at the meeting date
      grid.push_back(std::max(DAY, kt - DAY));
      grid.push_back(kt + DAY);
    } else {
      grid.push_back(kt);  // smooth back: the knot itself is enough
    }
    const double next = (k + 1 < allk.size()) ? allk[k + 1] : maxT;
    const double gap = next - kt;
    // weekly between the near knots (sharp front), coarser as gaps widen (the smooth back).
    const double step = gap < 0.6 ? WEEK : (gap < 2.5 ? 3 * WEEK : gap / 2.2);
    for (double t = kt + step; t < next - 0.6 * step; t += step) grid.push_back(t);
  }
  std::sort(grid.begin(), grid.end());
  grid.erase(std::unique(grid.begin(), grid.end(), [](double a, double b) { return std::abs(a - b) < 1e-9; }),
             grid.end());

  auto forwards_on_grid = [&](const Eigen::VectorXd& x) {
    // Actual forward (base+spread) for each curve on the grid, in PERCENT.
    auto handles = cal::build_bundle_curves<double>(prob.curves, [&](int c, int i) { return x[b.off[c] + i]; });
    std::vector<std::vector<double>> out(NC, std::vector<double>(grid.size()));
    for (int c = 0; c < NC; ++c)
      for (std::size_t g = 0; g < grid.size(); ++g) out[c][g] = 100.0 * handles[c]->forward(grid[g]);
    return out;
  };

  // ---- Run the day ----
  const int TICKS = 23 * 3600;      // 23h at 1 tick/s
  const int SNAPS = 32;
  const int snap_every = TICKS / SNAPS;
  const int SERIES = 828;           // downsampled per-tick timing/level timeline
  const int series_every = TICKS / SERIES;

  std::mt19937 rng(20260719u);
  std::normal_distribution<double> level_move(0.15, 0.25);  // per-tick |move| stats (bp), random sign
  std::normal_distribution<double> slope_move(0.0, 0.05);
  std::normal_distribution<double> spread_move(0.0, 0.04);
  std::uniform_int_distribution<int> sign(0, 1);
  const double bp = 1e-4;
  // Mild mean reversion (Ornstein-Uhlenbeck) so the day stays SENSIBLE (a realistic intraday range) while
  // each tick still moves ~mean 0.15 / sd 0.25 bp. Without it a pure walk of 82,800 steps wanders >150bp.
  const double kappa = 0.0006;
  double level = 0.0;                    // cumulative SOFR level (bp)
  std::vector<double> spread(NC, 0.0);   // cumulative per-curve spread offset (bp)

  std::vector<double> cal_us_all;
  cal_us_all.reserve(TICKS);
  struct Snap { double hour, level_bp, cal_us; std::vector<std::vector<double>> fwd; };
  std::vector<Snap> snaps;
  std::vector<double> series_hour, series_cal_us, series_level_bp;

  double sum_us = 0, min_us = 1e18, max_us = 0;
  for (int t = 0; t < TICKS; ++t) {
    // SOFR: a mean-reverting LEVEL + gentle SLOPE. Level+slope leave 2nd differences (butterflies) fixed,
    // so intermediate knots track their neighbours. Each spread curve gets its own small reverting move.
    const double dL = (sign(rng) ? 1.0 : -1.0) * level_move(rng) - kappa * level;
    const double dS = slope_move(rng);
    level += dL;
    for (int i = 0; i < prob.curves[0].n_knots(); ++i) {
      const double tn = knots[0][i] / maxT;  // 0..1 along the curve
      x_true[b.off[0] + i] += (dL + dS * (tn - 0.5) * 2.0) * bp;
    }
    for (int c = 1; c < NC; ++c) {
      const double ds = spread_move(rng) - kappa * spread[c], dss = 0.4 * slope_move(rng);
      spread[c] += ds;
      for (int i = 0; i < prob.curves[c].n_knots(); ++i) {
        const double tn = knots[c][i] / maxT;
        x_true[b.off[c] + i] += (ds + dss * (tn - 0.5) * 2.0) * bp;
      }
    }
    const double sofr_level_bp = level;

    const Eigen::VectorXd q = engine.model_rates(x_true);
    const auto t0 = std::chrono::steady_clock::now();
    sc.update(q);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    cal_us_all.push_back(us);
    sum_us += us;
    min_us = std::min(min_us, us);
    max_us = std::max(max_us, us);

    if (t % snap_every == 0 && static_cast<int>(snaps.size()) < SNAPS)
      snaps.push_back({23.0 * t / TICKS, sofr_level_bp, us, forwards_on_grid(sc.current())});
    if (t % series_every == 0) {
      series_hour.push_back(23.0 * t / TICKS);
      series_cal_us.push_back(us);
      series_level_bp.push_back(sofr_level_bp);
    }
  }

  std::vector<double> sorted = cal_us_all;
  std::sort(sorted.begin(), sorted.end());
  auto pct = [&](double p) { return sorted[std::min<std::size_t>(sorted.size() - 1, (std::size_t)(p * sorted.size()))]; };

  // ---- Emit JSON ----
  std::ofstream f("bundle_day.json");
  f.setf(std::ios::fixed);
  f << "{\n";
  f << "  \"ticks\":" << TICKS << ",\"hours\":23,\"tick_hz\":1,\n";
  f << "  \"shock\":{\"level_mean_bp\":0.15,\"level_sd_bp\":0.25},\n";
  f << "  \"curve_names\":[\"SOFR\",\"FF\",\"PRIME\",\"PRIME2\"],\n";
  f.precision(3);
  f << "  \"metrics\":{\"min_us\":" << min_us << ",\"max_us\":" << max_us
    << ",\"avg_us\":" << sum_us / TICKS << ",\"median_us\":" << pct(0.5)
    << ",\"p99_us\":" << pct(0.99) << ",\"refreshes\":" << sc.refresh_count() - 1 << "},\n";
  f << "  \"day_range_bp\":" << (*std::max_element(series_level_bp.begin(), series_level_bp.end()) -
                                 *std::min_element(series_level_bp.begin(), series_level_bp.end()))
    << ",\n";
  f.precision(5);
  auto arr = [&](const std::vector<double>& v) {
    f << "[";
    for (std::size_t i = 0; i < v.size(); ++i) f << (i ? "," : "") << v[i];
    f << "]";
  };
  f << "  \"grid\":";
  arr(grid);
  f << ",\n  \"knots\":{";
  for (int c = 0; c < NC; ++c) { f << (c ? "," : "") << "\"" << names[c] << "\":"; arr(knots[c]); }
  f << "},\n";
  f.precision(4);  // forwards in percent -> 4 dp = 0.01bp resolution, and a smaller payload
  f << "  \"snapshots\":[\n";
  for (std::size_t s = 0; s < snaps.size(); ++s) {
    f << "    {\"hour\":" << snaps[s].hour << ",\"level_bp\":" << snaps[s].level_bp
      << ",\"cal_us\":" << snaps[s].cal_us << ",\"fwd\":[";
    for (int c = 0; c < NC; ++c) { f << (c ? "," : ""); arr(snaps[s].fwd[c]); }
    f << "]}" << (s + 1 < snaps.size() ? "," : "") << "\n";
  }
  f << "  ],\n";
  f << "  \"series\":{\"hour\":";
  arr(series_hour);
  f << ",\"cal_us\":";
  arr(series_cal_us);
  f << ",\"level_bp\":";
  arr(series_level_bp);
  f << "}\n}\n";
  f.close();

  std::printf("ticks=%d  cal_us min=%.2f avg=%.2f max=%.2f median=%.2f p99=%.2f  refreshes=%d  day_range=%.1fbp\n",
              TICKS, min_us, sum_us / TICKS, max_us, pct(0.5), pct(0.99), sc.refresh_count() - 1,
              *std::max_element(series_level_bp.begin(), series_level_bp.end()) -
                  *std::min_element(series_level_bp.begin(), series_level_bp.end()));
  std::printf("grid points=%zu snapshots=%zu -> bundle_day.json\n", grid.size(), snaps.size());
  return 0;
}
