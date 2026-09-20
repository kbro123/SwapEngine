// EXPERIMENT (exp/piecewise-linear-w): WHY does the PWL streamer converge fewer ticks than the shipped one on
// desk_mixed's desk feed (1647 vs 1896 of 2000)? Three probes:
//   1. failure-reason histogram per engine;
//   2. the first tick the two runs disagree on convergence, and the state gap before it;
//   3. REPLAY: every tick the PWL run failed is re-run by BOTH engines from the SAME start (the shipped run's
//      previous committed x and market). Engine fault => PWL still fails there; path divergence => both pass.
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/streaming.hpp"

namespace cal = swaps::calibration;
using swaps::shapes::Shape;
using SC = cal::StreamingCalibrator<cal::BundleProblem>;

namespace {

// Same feed as exp_stream_flip's "desk" scenario: level + slope random walks, iid per-row jitter.
std::vector<Eigen::VectorXd> desk_path(const Shape& s, int ticks, unsigned seed, double level, double slope, double jit) {
  std::mt19937_64 g(seed);
  std::normal_distribution<double> n;
  Eigen::VectorXd mask = (s.q_small - s.q0).cwiseAbs();
  for (int i = 0; i < mask.size(); ++i) mask[i] = mask[i] > 0.0 ? 1.0 : 0.0;
  const int nr = static_cast<int>(mask.sum());
  Eigen::VectorXd load = Eigen::VectorXd::Zero(mask.size());
  for (int i = 0, k = 0; i < mask.size(); ++i)
    if (mask[i] > 0.0) load[i] = nr > 1 ? -1.0 + 2.0 * (k++) / (nr - 1) : 0.0;
  std::vector<Eigen::VectorXd> path;
  Eigen::VectorXd q = s.q0;
  double L = 0, Sl = 0;
  for (int t = 0; t < ticks; ++t) {
    L += level * 1e-4 * n(g);
    Sl += slope * 1e-4 * n(g);
    for (int i = 0; i < q.size(); ++i)
      if (mask[i] > 0.0) q[i] = s.q0[i] + L + Sl * load[i] + jit * 1e-4 * n(g);
    path.push_back(q);
  }
  return path;
}

// The problem as it stands at market q: targets AND the requoted bands (a fresh engine/streamer starts here).
cal::BundleProblem at_market(const Shape& s, const Eigen::VectorXd& q) {
  cal::BundleProblem p = s.prob;
  const auto rq = s.requote(q);
  for (int i = 0; i < p.n_residuals(); ++i) {
    auto& in = p.instruments[static_cast<std::size_t>(i)];
    in.market = rq.target[i];
    in.band_lower = rq.lower[i];
    in.band_upper = rq.upper[i];
    in.band_decay = rq.decay[i];
  }
  return p;
}

struct Run {
  std::vector<cal::StreamTick> ticks;
  std::vector<Eigen::VectorXd> xs;
};

Run run(const Shape& s, bool pwl, const std::vector<Eigen::VectorXd>& path, double nudge = 0.0) {
  std::vector<cal::Instrument> ins = s.prob.instruments;
  cal::HybridBundleResidual eng(s.prob, pwl);
  Eigen::VectorXd x0 = s.x_true;
  x0[0] += nudge;  // a rounding-level perturbation of the start: tests SENSITIVITY, not the engine
  SC sc(eng, s.prob, x0, s.q0, SC::Options{});
  bool has_band = s.has_bands;
  Run r;
  for (const auto& q : path) {
    if (s.has_bands) {
      const auto rq = s.requote(q);
      cal::requote_bands(ins, &eng, &sc, rq.lower, rq.upper, rq.decay, has_band);
    }
    r.ticks.push_back(sc.update(q));
    r.xs.push_back(sc.current());
  }
  return r;
}

// One tick from a GIVEN start: a fresh engine + streamer anchored at (x0, q_prev), then update to q.
cal::StreamTick one_tick(const Shape& s, bool pwl, const Eigen::VectorXd& x0, const Eigen::VectorXd& q_prev,
                         const Eigen::VectorXd& q) {
  const cal::BundleProblem p = at_market(s, q_prev);
  std::vector<cal::Instrument> ins = p.instruments;
  cal::HybridBundleResidual eng(p, pwl);
  SC sc(eng, p, x0, q_prev, SC::Options{});
  bool has_band = s.has_bands;
  const auto rq = s.requote(q);
  cal::requote_bands(ins, &eng, &sc, rq.lower, rq.upper, rq.decay, has_band);
  return sc.update(q);
}

void histogram(const char* who, const Run& r) {
  std::map<std::string, int> h;
  for (const auto& t : r.ticks) ++h[t.reason()];
  std::printf("  %-8s", who);
  for (const auto& [k, v] : h) std::printf("  %s: %d", k.c_str(), v);
  std::printf("\n");
}

}  // namespace

int main() {
  const Shape s = swaps::shapes::desk_mixed();
  const int T = 2000;
  const auto path = desk_path(s, T, 12345, 0.15, 0.05, 0.10);
  const Run b = run(s, false, path), p = run(s, true, path);

  std::printf("1. failure reasons over %d ticks (desk feed)\n", T);
  histogram("shipped", b);
  histogram("pwl", p);

  std::printf("\n2. first disagreement\n");
  for (int t = 0; t < T; ++t)
    if (b.ticks[t].converged != p.ticks[t].converged) {
      const double gap = t ? (b.xs[t - 1] - p.xs[t - 1]).cwiseAbs().maxCoeff() : 0.0;
      std::printf("  tick %d: shipped %s (%d steps, %d refr), pwl %s (%d steps, %d refr); state gap before it %.2e\n", t,
                  b.ticks[t].reason(), b.ticks[t].newton_steps, b.ticks[t].refreshes, p.ticks[t].reason(),
                  p.ticks[t].newton_steps, p.ticks[t].refreshes, gap);
      break;
    }
  int only_b = 0, only_p = 0, neither = 0;
  for (int t = 0; t < T; ++t) {
    only_b += b.ticks[t].converged && !p.ticks[t].converged;
    only_p += !b.ticks[t].converged && p.ticks[t].converged;
    neither += !b.ticks[t].converged && !p.ticks[t].converged;
  }
  std::printf("  ticks only shipped converged: %d, only pwl: %d, neither: %d\n", only_b, only_p, neither);

  std::printf("\n3. replay every PWL-failed tick from the SHIPPED run's previous state (same start for both)\n");
  int n = 0, both = 0, bonly = 0, ponly = 0, none = 0;
  std::map<std::string, int> preason, breason;
  for (int t = 1; t < T; ++t) {
    if (p.ticks[t].converged || !b.ticks[t - 1].converged) continue;
    ++n;
    const cal::StreamTick tb = one_tick(s, false, b.xs[t - 1], path[t - 1], path[t]);
    const cal::StreamTick tp = one_tick(s, true, b.xs[t - 1], path[t - 1], path[t]);
    both += tb.converged && tp.converged;
    bonly += tb.converged && !tp.converged;
    ponly += !tb.converged && tp.converged;
    none += !tb.converged && !tp.converged;
    if (!tp.converged) ++preason[tp.reason()];
    if (!tb.converged) ++breason[tb.reason()];
  }
  std::printf("  %d replayed: both converge %d | shipped only %d | pwl only %d | neither %d\n", n, both, bonly, ponly, none);
  for (const auto& [k, v] : preason) std::printf("    pwl replay failure: %s x%d\n", k.c_str(), v);
  for (const auto& [k, v] : breason) std::printf("    shipped replay failure: %s x%d\n", k.c_str(), v);

  std::printf("\n4. is the count a chaotic statistic? converged ticks of %d, per feed seed\n", T);
  std::printf("  %-6s %9s %9s %18s %18s\n", "seed", "shipped", "pwl", "shipped +1e-13", "pwl +1e-13");
  int sb = 0, sp = 0;
  for (unsigned seed : {12345u, 1u, 2u, 3u, 4u, 5u, 6u, 7u}) {
    const auto pth = desk_path(s, T, seed, 0.15, 0.05, 0.10);
    const auto conv = [](const Run& r) { int c = 0; for (const auto& t : r.ticks) c += t.converged; return c; };
    const int cb = conv(run(s, false, pth)), cp = conv(run(s, true, pth));
    const int cbn = conv(run(s, false, pth, 1e-13)), cpn = conv(run(s, true, pth, 1e-13));
    sb += cb; sp += cp;
    std::printf("  %-6u %9d %9d %18d %18d\n", seed, cb, cp, cbn, cpn);
  }
  std::printf("  total  %9d %9d\n", sb, sp);
  return 0;
}
