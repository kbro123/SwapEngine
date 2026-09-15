// Step 3 diagnosis probe (desk_mixed). Header-only, run on a quiet machine AFTER the guard commits land.
//  S1 which knot carries the 4.69e-3 first step of a 0.1 bp tick: per-knot |x(q_small) - x(q0)|, the Jacobian's singular values at the
//     committed state, the weakest right-singular vector's top knots, and d x / d q along the tick direction (the solution's sensitivity).
//  S2 the kink half-step feeding the adaptive stall: run the 0.1 bp ticks (break-even pinned 64) with the stall OFF, and with the step cap
//     raised, reporting steps / refreshes / converged per tick -- does the frozen iteration converge on its own after the half step?
//  S3 the +-1 bp hard-row walk sequence (4/20 ticks fail): per failing tick the status, steps, refreshes, rescales, pins, releases.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"
namespace cal = swaps::calibration;
using swaps::shapes::Shape;
using SC = cal::StreamingCalibrator<cal::BundleProblem>;

static const Shape& rung(const char* n) {
  static const auto L = swaps::shapes::ladder();
  for (const auto& s : L) if (s.name == n) return s;
  throw 0;
}

int main() {
  const Shape& s = rung("desk_mixed");
  const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
  cal::HybridBundleResidual eng(s.prob);

  std::printf("== S1 sensitivity of the solution to the 0.1 bp tick\n");
  {
    SC::Options o; o.breakeven_steps = 64;
    SC st(eng, s.prob, x, s.q0, o);
    st.update(s.q_small); const Eigen::VectorXd xs = st.current();
    st.update(s.q0); const Eigen::VectorXd x0 = st.current();
    Eigen::VectorXd d = (xs - x0).cwiseAbs();
    std::vector<int> idx(d.size()); for (int i = 0; i < d.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return d[a] > d[b]; });
    std::printf("  |x(q_small) - x(q0)|_inf %.3e ; |q_small - q0|_inf %.3e\n  top knots:", d.maxCoeff(), (s.q_small - s.q0).cwiseAbs().maxCoeff());
    for (int k = 0; k < 6; ++k) std::printf(" [%d] %.2e", idx[k], d[idx[k]]);
    std::printf("\n  curve offsets:");
    for (std::size_t c = 0; c < s.prob.curves.size(); ++c) std::printf(" c%zu@%d", c, s.prob.offset(static_cast<int>(c)));
    std::printf("\n");
    const Eigen::MatrixXd J = eng.jacobian_vs(x0, s.q0);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeThinV);
    const Eigen::VectorXd sv = svd.singularValues();
    std::printf("  J %dx%d  sigma_max %.3e  sigma_min %.3e  cond %.3e\n  smallest 5 sigma:", (int)J.rows(), (int)J.cols(), sv[0], sv[sv.size() - 1],
                sv[0] / sv[sv.size() - 1]);
    for (int k = 0; k < 5; ++k) std::printf(" %.2e", sv[sv.size() - 1 - k]);
    const Eigen::VectorXd v = svd.matrixV().col(sv.size() - 1).cwiseAbs();
    std::vector<int> vi(v.size()); for (int i = 0; i < v.size(); ++i) vi[i] = i;
    std::sort(vi.begin(), vi.end(), [&](int a, int b) { return v[a] > v[b]; });
    std::printf("\n  weakest direction top knots:");
    for (int k = 0; k < 6; ++k) std::printf(" [%d] %.2f", vi[k], v[vi[k]]);
    std::printf("\n");
  }

  std::printf("== S2 the half step and the adaptive stall (10 ticks each way after 4 warm; break-even 64)\n");
  struct V { const char* name; bool stall; int max_steps; };
  for (const V v : {V{"default", true, 64}, V{"stall off", false, 64}, V{"stall off, steps 400", false, 400}}) {
    SC::Options o; o.breakeven_steps = 64; o.adaptive_stall = v.stall; o.max_steps = v.max_steps;
    SC st(eng, s.prob, x, s.q0, o);
    bool f = false;
    for (int i = 0; i < 4; ++i) { f = !f; st.update(f ? s.q_small : s.q0); }
    int R = 0, N = 0, fails = 0;
    std::printf("  %-22s", v.name);
    for (int i = 0; i < 10; ++i) {
      f = !f;
      const cal::StreamTick t = st.update(f ? s.q_small : s.q0);
      R += t.refreshes; N += t.newton_steps; fails += !t.converged;
      std::printf(" %d/%d%s", t.newton_steps, t.refreshes, t.converged ? "" : "!");
    }
    std::printf("  | steps %d refreshes %d failed %d\n", N, R, fails);
  }

  std::printf("== S3 +-1 bp and +-3 bp hard-row walk sequence (20 ticks after 6 warm; break-even 64)\n");
  for (double amp : {1e-4, 3e-4}) {
    SC::Options o; o.breakeven_steps = 64;
    SC st(eng, s.prob, x, s.q0, o);
    std::vector<Eigen::VectorXd> seq;
    for (int k = 0; k < 26; ++k) {
      Eigen::VectorXd q = s.q0;
      for (int i = 0; i < q.size(); ++i) {
        const auto& in = s.prob.instruments[static_cast<std::size_t>(i)];
        if (in.band_upper > in.band_lower || in.quote == cal::QuoteKind::FxForward || in.quote == cal::QuoteKind::TurnJump) continue;
        q[i] += amp * std::sin(0.8 * k + 0.9 * i);
      }
      seq.push_back(q);
    }
    for (int k = 0; k < 6; ++k) st.update(seq[static_cast<std::size_t>(k)]);
    for (int k = 6; k < 26; ++k) {
      const int p0 = st.pin_count(), r0 = st.release_count();
      const cal::StreamTick t = st.update(seq[static_cast<std::size_t>(k)]);
      if (!t.converged)
        std::printf("  amp %.0e tick %d FAILED %s: steps %d refreshes %d rescales %d pins %d releases %d drift %.2e\n", amp, k, t.reason(),
                    t.newton_steps, t.refreshes, t.rescales, st.pin_count() - p0, st.release_count() - r0, t.drift);
    }
  }
}
