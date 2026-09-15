// Step 3, second probe (counts only; run on a quiet machine after G4 lands). The first probe found the 0.1 bp tick moves knot 54 (curve 3,
// EUR-in-USD xccy, its last = 30Y knot) by 4.69e-3, the solution's weakest direction is the SOFR long end (knots 5-11), and every tick
// refreshes even with the adaptive stall off (frozen steps contract at rho 0.577 -> the max_frozen cap).
//  X1 desk vs desk_mixed: per-knot |x(q_small) - x(q0)| top 6 on BOTH (is the 30Y xccy swing desk_mixed-specific?), and d x_54 / d q per row
//     (the solution sensitivity column via J+: which quote rows drive knot 54).
//  X2 per-row q_small bump split: move ONLY the SOFR rows / ONLY the xccy rows / ONLY the rest by their q_small bump; report knot 54's move
//     and steps / refreshes per tick on desk_mixed.
//  X3 the frozen contraction: J(x_q0) vs J(x_qsmall) relative change ||dJ|| / ||J|| on desk vs desk_mixed, and rho of the frozen iteration
//     (spectral radius of I - M(x0) J(x1)) -- the number that forces the max_frozen refresh.
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
  for (const char* name : {"desk", "desk_mixed"}) {
    const Shape& s = rung(name);
    const Eigen::VectorXd x = cal::calibrate(s.prob, s.x0).x;
    cal::HybridBundleResidual eng(s.prob);
    SC::Options o; o.breakeven_steps = 64;
    SC st(eng, s.prob, x, s.q0, o);
    st.update(s.q_small); const Eigen::VectorXd xs = st.current();
    st.update(s.q0); const Eigen::VectorXd x0 = st.current();
    Eigen::VectorXd d = (xs - x0).cwiseAbs();
    std::vector<int> idx(d.size()); for (int i = 0; i < d.size(); ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return d[a] > d[b]; });
    std::printf("== %s X1: |x(q_small)-x(q0)|_inf %.3e  top:", name, d.maxCoeff());
    for (int k = 0; k < 6; ++k) std::printf(" [%d] %.2e", idx[k], d[idx[k]]);
    std::printf("\n");
    const Eigen::MatrixXd J0 = eng.jacobian_vs(x0, s.q0), J1 = eng.jacobian_vs(xs, s.q_small);
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(J0);
    const Eigen::MatrixXd P = cod.pseudoInverse();  // n_knots x n_res
    Eigen::VectorXd col = P.row(54).transpose().cwiseAbs();
    std::vector<int> ri(col.size()); for (int i = 0; i < col.size(); ++i) ri[i] = i;
    std::sort(ri.begin(), ri.end(), [&](int a, int b) { return col[a] > col[b]; });
    std::printf("   d x54 / d q (J+ row 54) top rows:");
    for (int k = 0; k < 8; ++k) std::printf(" r%d %.1f", ri[k], P(54, ri[k]));
    std::printf("\n   X3 ||J(x_small)-J(x0)||/||J(x0)|| %.3e", (J1 - J0).norm() / J0.norm());
    const Eigen::MatrixXd T = Eigen::MatrixXd::Identity(x.size(), x.size()) - P * J1;
    Eigen::EigenSolver<Eigen::MatrixXd> es(T);
    std::printf("  spectral radius of I - J0+ J1: %.3f\n", es.eigenvalues().cwiseAbs().maxCoeff());

    if (std::string(name) != "desk_mixed") continue;
    std::printf("== desk_mixed X2: bump only one group by its q_small move (10 ticks each way after 2 warm)\n");
    struct Grp { const char* g; int lo, hi; };
    // row ranges from desk_impl: SOFR par swaps first (curve 0), then FF futures+basis (1), ESTR (2), FX fwd + xccy (3), EURIBOR (4), extras
    std::vector<int> curve_of(s.prob.n_residuals(), -1);
    for (int i = 0; i < s.prob.n_residuals(); ++i) {
      const auto& in = s.prob.instruments[static_cast<std::size_t>(i)];
      curve_of[i] = (in.quote == cal::QuoteKind::FxForward || in.quote == cal::QuoteKind::XccyMtmBasis) ? 3 : in.fwd.forecast;
    }
    for (int grp = -1; grp < 5; ++grp) {
      Eigen::VectorXd qb = s.q0;
      for (int i = 0; i < qb.size(); ++i) if (grp < 0 || curve_of[i] == grp) qb[i] = s.q_small[i];
      SC st2(eng, s.prob, x, s.q0, o);
      bool f = false;
      for (int i = 0; i < 2; ++i) { f = !f; st2.update(f ? qb : s.q0); }
      int R = 0, N = 0, fails = 0; double m54 = 0.0;
      for (int i = 0; i < 10; ++i) {
        f = !f;
        const Eigen::VectorXd prev = st2.current();
        const cal::StreamTick t = st2.update(f ? qb : s.q0);
        R += t.refreshes; N += t.newton_steps; fails += !t.converged;
        m54 = std::max(m54, std::abs(st2.current()[54] - prev[54]));
      }
      std::printf("   group %-3s steps/tick %.1f refreshes/tick %.1f failed %d  max |dx54| %.2e\n", grp < 0 ? "all" : std::to_string(grp).c_str(),
                  N / 10.0, R / 10.0, fails, m54);
    }
  }
}
