// EXPERIMENT (exp/piecewise-linear-w): reproduce ShapeLadder.EveryRungConvergesOnTheGateTicks on mixed_scheme
// step by step. Run with SWAPS_EXP_PWL=1 (the session's engine is then the PWL tier) and without (control).
#include <cstdio>

#include "shape_ladder.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/calibration/hybrid_residual.hpp"

namespace cal = swaps::calibration;
namespace api = swaps::api;

int main() {
  const auto s = swaps::shapes::mixed_scheme();
  api::BundleSession sess(s.prob);
  const auto& cr = sess.calibrate(s.x0);
  {
    const cal::HybridBundleResidual judge(s.prob, false);
    std::printf("cold calibrate: |r| (shipped judge) at the solved x = %.2e\n", judge.residuals(cr.x).cwiseAbs().maxCoeff());
  }
  sess.start_streaming();
  const char* names[] = {"q_big", "q0", "q_small", "q0"};
  for (int rep = 0; rep < 3; ++rep) {
    int k = 0;
    for (const Eigen::VectorXd* q : {&s.q_big, &s.q0, &s.q_small, &s.q0}) {
      const Eigen::VectorXd x = sess.stream_update(*q);
      const cal::HybridBundleResidual judge(s.prob, false);
      std::printf("rep %d %-8s converged %d  steps %3d  reason %-14s  |r| judge %.2e\n", rep, names[k++],
                  sess.last_converged(), sess.last_newton_steps(), sess.last_reason(),
                  judge.residuals_vs(x, *q).cwiseAbs().maxCoeff());
    }
  }
  return 0;
}
