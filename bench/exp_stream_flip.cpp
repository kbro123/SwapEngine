// EXPERIMENT (exp/piecewise-linear-w): the piecewise-linear W tier under a STREAMING feed.
//
// The tier re-takes W only when x's Hyman branch pattern changes, so its value depends on how often that
// happens on a realistic feed. Both engines -- the shipped router (value-dependent rows on AAD) and the PWL
// tier -- stream the SAME quote paths through StreamingCalibrator; we report per-tick time, pattern flips
// (W re-takes), Jacobian refreshes, and whether the two engines stay on the same solution.
//
// Feed per tick on every rate row: a COMMON level move (sigma_level) + an IDIOSYNCRATIC move (sigma_idio).
// FX forwards stay at market (covered interest parity, as the shape ladder's q_small). A banded row's band
// travels with its target (Shape::requote), exactly as a desk requote does.
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/streaming.hpp"

namespace cal = swaps::calibration;
using swaps::shapes::Shape;
using Clock = std::chrono::steady_clock;

namespace {

struct Scenario {
  const char* name;
  double level_bp, slope_bp, jitter_bp;  // per-tick: level + slope RANDOM WALKS, and iid (non-accumulating) jitter
  bool accumulate_idio = false;          // the first run's feed: per-row noise that WALKS (jagged; unrealistic)
};

std::vector<Eigen::VectorXd> make_path(const Shape& s, const Scenario& sc, int ticks, unsigned seed) {
  std::mt19937_64 g(seed);
  std::normal_distribution<double> n;
  Eigen::VectorXd mask = (s.q_small - s.q0).cwiseAbs();
  for (int i = 0; i < mask.size(); ++i) mask[i] = mask[i] > 0.0 ? 1.0 : 0.0;
  // A row's slope loading: its position among the rate rows, centred on [-1, 1] (a proxy for tenor order --
  // the ladder lists each curve's strip short to long).
  const int nr = static_cast<int>(mask.sum());
  Eigen::VectorXd load = Eigen::VectorXd::Zero(mask.size());
  for (int i = 0, k = 0; i < mask.size(); ++i)
    if (mask[i] > 0.0) load[i] = nr > 1 ? -1.0 + 2.0 * (k++) / (nr - 1) : 0.0;
  std::vector<Eigen::VectorXd> path;
  Eigen::VectorXd q = s.q0, walk = Eigen::VectorXd::Zero(mask.size());
  double L = 0, Sl = 0;
  for (int t = 0; t < ticks; ++t) {
    L += sc.level_bp * 1e-4 * n(g);
    Sl += sc.slope_bp * 1e-4 * n(g);
    for (int i = 0; i < q.size(); ++i) {
      if (mask[i] == 0.0) continue;
      const double e = sc.jitter_bp * 1e-4 * n(g);
      if (sc.accumulate_idio) walk[i] += e;
      q[i] = s.q0[i] + L + Sl * load[i] + (sc.accumulate_idio ? walk[i] : e);
    }
    path.push_back(q);
  }
  return path;
}

struct Result {
  double us_per_tick = 0, worst_us = 0;
  int converged = 0, refreshes = 0, steps = 0, rebuilds = 0, distinct = 0, analytic = 0;
  double w_err = 0;  // max |r_pwl(x) - r_shipped(x)| over every committed x: validates the rank-k W
  std::vector<Eigen::VectorXd> xs;
  std::vector<char> conv;
};

Result stream(const Shape& s, bool pwl, const std::vector<Eigen::VectorXd>& path, bool check = false) {
  std::vector<cal::Instrument> ins = s.prob.instruments, ins_j = s.prob.instruments;
  cal::HybridBundleResidual eng(s.prob, pwl);
  // The W check's judge: a SHIPPED engine that receives the SAME band requotes (a banded row's residual reads
  // its band, so a judge with stale bands would disagree on every banded row regardless of W).
  cal::HybridBundleResidual judge_eng(s.prob, false);
  const cal::HybridBundleResidual* judge = check ? &judge_eng : nullptr;
  bool has_band_j = s.has_bands;
  using SC = cal::StreamingCalibrator<cal::BundleProblem>;
  SC sc(eng, s.prob, s.x_true, s.q0, SC::Options{});
  bool has_band = s.has_bands;
  Result r;
  double total = 0;
  const int rb0 = eng.pwl_rebuilds();
  for (const auto& q : path) {
    if (s.has_bands) {
      const auto rq = s.requote(q);
      if (cal::requote_bands(ins, &eng, &sc, rq.lower, rq.upper, rq.decay, has_band))
        std::fprintf(stderr, "  (re-anchor requested -- not expected on a band shift)\n");
      if (check)
        cal::requote_bands(ins_j, &judge_eng, static_cast<SC*>(nullptr), rq.lower, rq.upper, rq.decay, has_band_j);
    }
    const auto t0 = Clock::now();
    const cal::StreamTick tk = sc.update(q);
    const double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    total += us;
    r.worst_us = std::max(r.worst_us, us);
    r.converged += tk.converged;
    r.refreshes += tk.refreshes;
    r.steps += tk.newton_steps;
    r.xs.push_back(sc.current());
    if (judge)  // OUTSIDE the timed region: the engine's residual at its own x vs the shipped engine's
      r.w_err = std::max(r.w_err, (eng.residuals_vs(sc.current(), q) - judge->residuals_vs(sc.current(), q)).cwiseAbs().maxCoeff());
    r.conv.push_back(tk.converged ? 1 : 0);
  }
  r.us_per_tick = total / static_cast<double>(path.size());
  r.rebuilds = eng.pwl_rebuilds() - rb0;
  r.distinct = eng.pwl_distinct();
  r.analytic = eng.pwl_analytic();
  return r;
}

// Cost of ONE W re-take in isolation: alternate between two states in different branch cells, so every
// residuals() call flips the pattern; subtract the no-flip cost of the same call.
void retake_cost(const Shape& s) {
  const cal::HybridBundleResidual e(s.prob, true);
  std::mt19937_64 g(99);
  std::normal_distribution<double> n;
  Eigen::VectorXd x1 = s.x_true;
  (void)e.residuals(s.x_true);
  for (int attempt = 0; attempt < 200; ++attempt) {  // find a state in another cell
    x1 = s.x_true;
    for (int i = 0; i < x1.size(); ++i) x1[i] += 5e-4 * n(g);
    const int before = e.pwl_rebuilds();
    (void)e.residuals(x1);
    (void)e.residuals(s.x_true);
    if (e.pwl_rebuilds() - before == 2) break;
  }
  const int N = 2000;
  const int rb = e.pwl_rebuilds();
  auto t0 = Clock::now();
  double sink = 0;
  for (int k = 0; k < N; ++k) sink += e.residuals((k & 1) ? x1 : s.x_true)[0];
  const double flip_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / N;
  const int flips = e.pwl_rebuilds() - rb;
  t0 = Clock::now();
  for (int k = 0; k < N; ++k) sink += e.residuals(s.x_true)[0];
  const double same_us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / N;
  std::printf("  W re-take: %.1f us per flipping residual call vs %.2f us same-cell (%d/%d calls flipped)  [sink %.1e]\n",
              flip_us, same_us, flips, N, sink);
}

}  // namespace

int main() {
  const std::vector<Scenario> scenarios = {
      {"quiet   walk .10 lvl .03 slp, jit .03", 0.10, 0.03, 0.03},
      {"desk    walk .15 lvl .05 slp, jit .10", 0.15, 0.05, 0.10},
      {"stress  walk .50 lvl .20 slp, jit .50", 0.50, 0.20, 0.50},
      {"jagged  (run-1 feed: idio WALKS .10)", 0.15, 0.00, 0.10, true},
  };
  const int ticks = 2000;
  for (const Shape& s : {swaps::shapes::mixed_scheme(), swaps::shapes::desk_mixed()}) {
    std::printf("\n=== %s (%d rows, %d knots) ===\n", s.name.c_str(), s.prob.n_residuals(), s.prob.n_knots());
    retake_cost(s);
    std::printf("  %-38s %6s %9s %9s %7s %7s %7s %9s %11s %11s\n", "scenario", "engine", "us/tick", "worst us", "conv",
                "refr", "steps", "W retake", "|dx| vs base", "max|r(x)|");
    const cal::HybridBundleResidual judge(s.prob, false);  // the SHIPPED residual, used to grade both engines' x
    for (const auto& sc : scenarios) {
      const auto path = make_path(s, sc, ticks, 12345);
      const Result b = stream(s, false, path), p = stream(s, true, path, /*check=*/true);
      // |dx| only over ticks where BOTH converged (an unconverged tick keeps the old x: the paths legitimately differ).
      // max|r(x)|: the shipped engine's residual at each engine's committed x, against that tick's market, on
      // SQUARE rungs only (an over-determined fit has r != 0 by design; there we report n/a).
      const bool square = s.prob.n_residuals() == s.prob.n_knots();
      double dx = 0, rb = 0, rp = 0;
      int both = 0;
      for (int t = 0; t < ticks; ++t) {
        if (!(b.conv[t] && p.conv[t])) continue;
        ++both;
        dx = std::max(dx, (p.xs[t] - b.xs[t]).cwiseAbs().maxCoeff());
        if (square) {
          rb = std::max(rb, judge.residuals_vs(b.xs[t], path[t]).cwiseAbs().maxCoeff());
          rp = std::max(rp, judge.residuals_vs(p.xs[t], path[t]).cwiseAbs().maxCoeff());
        }
      }
      char rbs[16], rps[16];
      std::snprintf(rbs, sizeof rbs, square ? "%.1e" : "n/a", rb);
      std::snprintf(rps, sizeof rps, square ? "%.1e" : "n/a", rp);
      std::printf("  %-38s %6s %9.2f %9.1f %7d %7d %7d %9s %11s %11s\n", sc.name, "base", b.us_per_tick, b.worst_us,
                  b.converged, b.refreshes, b.steps, "-", "-", rbs);
      std::printf("  %-38s %6s %9.2f %9.1f %7d %7d %7d %9d %11.1e %11s   (both conv: %d)\n", "", "pwl", p.us_per_tick,
                  p.worst_us, p.converged, p.refreshes, p.steps, p.rebuilds, dx, rps, both);
      std::printf("  %-38s %6s re-takes: %d analytic + %d AAD; W check max|r_pwl - r_shipped| = %.1e%s\n", "", "",
                  p.analytic, p.rebuilds - p.analytic, p.w_err,
                  p.distinct ? (" ; distinct cells " + std::to_string(p.distinct)).c_str() : "");
    }
  }
  return 0;
}
