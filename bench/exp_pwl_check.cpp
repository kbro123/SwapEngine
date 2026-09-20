// EXPERIMENT (exp/piecewise-linear-w): differential check of the ANALYTIC re-take. One long-lived PWL engine
// jumps between random states (small and LARGE moves); after every jump its residuals and Jacobian must equal
// a FRESH engine's at the same x (whose first sync is a full AAD re-take, exact for x's cell).
#include <cstdio>
#include <random>

#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"

namespace cal = swaps::calibration;

int main() {
  for (const auto& s : {swaps::shapes::mixed_scheme(), swaps::shapes::desk_mixed()}) {
    const cal::HybridBundleResidual live(s.prob, true);
    std::mt19937_64 g(7);
    std::normal_distribution<double> n;
    double wr = 0, wj = 0;
    int worst_step = -1;
    const double sizes[] = {1e-6, 1e-5, 1e-4, 1e-3, 5e-3};
    Eigen::VectorXd x = s.x_true;
    for (int step = 0; step < 400; ++step) {
      const double sz = sizes[step % 5];
      for (int i = 0; i < x.size(); ++i) x[i] = s.x_true[i] + sz * n(g);
      const cal::HybridBundleResidual fresh(s.prob, true);
      const double er = (live.residuals(x) - fresh.residuals(x)).cwiseAbs().maxCoeff();
      const Eigen::MatrixXd Jl = live.jacobian(x), Jf = fresh.jacobian(x);
      const double ej = (Jl - Jf).cwiseAbs().maxCoeff() / std::max(1e-300, Jf.cwiseAbs().maxCoeff());
      if (er > wr || ej > wj) worst_step = step;
      wr = std::max(wr, er);
      wj = std::max(wj, ej);
      if ((er > 1e-12 || ej > 1e-10) && step < 60)
        std::printf("  %s step %d (move %.0e): |dr| %.2e  |dJ| rel %.2e   analytic %d / re-takes %d\n", s.name.c_str(), step,
                    sz, er, ej, live.pwl_analytic(), live.pwl_rebuilds());
    }
    std::printf("%-14s max |dr| %.2e   max |dJ| rel %.2e (worst step %d)   analytic %d of %d re-takes\n", s.name.c_str(), wr,
                wj, worst_step, live.pwl_analytic(), live.pwl_rebuilds());
  }
  return 0;
}
