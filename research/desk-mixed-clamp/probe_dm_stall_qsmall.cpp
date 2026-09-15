// Step 3a (owner 2026-09-15: FIXTURE FIRST). Which part of q_small is unrealistic, and what does desk_mixed do on an honest 0.1 bp tick?
// q_small today: every row + 1e-5 * sin(0.7 i + 0.3), an FX forward RELATIVELY (market * (1 + 1e-5 sin)). A relative 1e-5 on a 1W forward
// implies a ~5 bp move in the rate differential (dF/F = d(r_d - r_f) T), so the short FX rows are the suspect. Variants:
//   V0 today's q_small
//   V1 FX forwards unchanged (covered interest parity, as q_big)
//   V2 FX forwards unchanged, xccy basis rows a tenth (as q_big)
//   V3 FX forwards moved CIP-consistently: F * exp(1e-5 * sin * T) (a 0.1 bp differential move, noisy per tenor)
//   V4 realistic parallel: +0.1 bp x q_big's tilt on rate rows, basis a tenth, FX / spread quotes unchanged (no per-row noise)
// For desk, desk_mixed, fx_xccy (break-even pinned 64 AND measured): |x(q)-x(q0)|_inf and its knot, steps / refreshes / allocs per tick over
// 20 ticks after 6 warm (alternating q0 <-> q), and the frozen-map spectral radius at the tick.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "malloc_count.hpp"
#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"
namespace cal = swaps::calibration;
using swaps::shapes::Shape;
using SC = cal::StreamingCalibrator<cal::BundleProblem>;

static double maturity(const cal::Instrument& ins) {
  switch (ins.quote) {
    case cal::QuoteKind::FxForward: return ins.fx_time;
    case cal::QuoteKind::Rate: return ins.obs.sub_end.empty() ? 0.0 : ins.obs.sub_end.back();
    default: return ins.fixed.coupons.empty() ? 0.0 : ins.fixed.coupons.back().pay;
  }
}

static Eigen::VectorXd variant(const Shape& s, int v) {
  Eigen::VectorXd q = s.q0;
  for (int i = 0; i < q.size(); ++i) {
    const auto& ins = s.prob.instruments[static_cast<std::size_t>(i)];
    const bool fx = ins.quote == cal::QuoteKind::FxForward;
    const bool basis = ins.quote == cal::QuoteKind::XccyMtmBasis || ins.quote == cal::QuoteKind::ParSpread;
    const bool spread = ins.quote == cal::QuoteKind::Portfolio || ins.quote == cal::QuoteKind::TurnJump;
    const double bump = std::sin(0.7 * i + 0.3);
    switch (v) {
      case 0: q[i] = s.q_small[i]; break;
      case 1: q[i] = fx ? ins.market : ins.market + 1e-5 * bump; break;
      case 2: q[i] = fx ? ins.market : ins.market + (ins.quote == cal::QuoteKind::XccyMtmBasis ? 1e-6 : 1e-5) * bump; break;
      case 3: q[i] = fx ? ins.market * std::exp(1e-5 * bump * maturity(ins)) : ins.market + 1e-5 * bump; break;
      case 4: {
        const double tilt = 0.9 + 0.2 * std::min(maturity(ins), 30.0) / 30.0;
        q[i] = (fx || spread) ? ins.market : ins.market + (basis ? 1e-6 : 1e-5) * tilt;
        break;
      }
    }
  }
  return q;
}

int main() {
  static const auto L = swaps::shapes::ladder();
  for (const char* name : {"fx_xccy", "desk", "desk_mixed"}) {
    const Shape* sp = nullptr;
    for (const auto& s : L) if (s.name == name) sp = &s;
    const Shape& s = *sp;
    const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
    cal::HybridBundleResidual eng(s.prob);
    for (int v = 0; v < 5; ++v) {
      const Eigen::VectorXd q = variant(s, v);
      for (double be : {64.0, 0.0}) {
        SC::Options o; o.breakeven_steps = be;
        SC st(eng, s.prob, x, s.q0, o);
        bool f = false;
        for (int i = 0; i < 6; ++i) { f = !f; st.update(f ? q : s.q0); }
        int R = 0, N = 0, fails = 0; unsigned long A = 0;
        {
          swaps::testing::AllocScope a;
          for (int i = 0; i < 20; ++i) { f = !f; const cal::StreamTick t = st.update(f ? q : s.q0); R += t.refreshes; N += t.newton_steps; fails += !t.converged; }
          A = a.allocs();
        }
        std::printf("  %-10s V%d be %-8s steps/tick %.2f refreshes/tick %.2f allocs/20 %lu failed %d", name, v, be == 0 ? "measured" : "64", N / 20.0,
                    R / 20.0, A, fails);
        if (be == 64.0) {
          st.update(q); const Eigen::VectorXd xq = st.current();
          st.update(s.q0); const Eigen::VectorXd x0 = st.current();
          Eigen::VectorXd d = (xq - x0).cwiseAbs();
          Eigen::Index k = 0; d.maxCoeff(&k);
          const Eigen::MatrixXd J0 = eng.jacobian_vs(x0, s.q0), J1 = eng.jacobian_vs(xq, q);
          Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(J0);
          const Eigen::MatrixXd T = Eigen::MatrixXd::Identity(x.size(), x.size()) - cod.pseudoInverse() * J1;
          Eigen::EigenSolver<Eigen::MatrixXd> es(T, false);
          std::printf("  | |dx|_inf %.2e @knot %d  frozen radius %.3f", d.maxCoeff(), (int)k, es.eigenvalues().cwiseAbs().maxCoeff());
        }
        std::printf("\n");
      }
    }
  }
}
