// How stale does the cached Jacobian actually get as the market moves away from where it was taken?
// M = J(x0)^{-1}. For a parallel move of `bp` from the anchor we report:
//   rho = ||I - M*J(x*)||_2   -- the frozen-Newton contraction factor (converges iff rho<1; it is
//                                also the RELATIVE error of M as an inverse-Jacobian at x*)
//   steps one-shot            -- frozen-Newton steps to solve the WHOLE move from x0 in one go
//   steps local               -- frozen-Newton steps for one more 0.15bp tick, warm-started at x*
// The point: M's *accuracy* degrades steadily (rho grows), but it stays a *contraction* far past
// 5-10bp, and a per-tick 0.15bp correction stays cheap because it is LOCAL, not one-shot.

#include <ql/quantlib.hpp>

#include <Eigen/Dense>
#include <cstdio>

#include "reference_curve.hpp"
#include "swaps/calibration/compiled_residual.hpp"
#include "swaps/calibration/lm.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;

int main() {
  RelinkableHandle<YieldTermStructure> h;
  auto mk = rb::build_market(h);
  auto prob = rb::build_square_problem(mk);
  cal::CompiledResidual cr(prob);
  const int R = prob.n_residuals();
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(R, R);

  const Eigen::VectorXd x0 = cal::calibrate(prob, Eigen::VectorXd::Constant(prob.n_knots(), 0.035), true).x;
  const Eigen::VectorXd q0 = cr.model_rates(x0);
  const Eigen::MatrixXd M =
      Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(cr.jacobian(x0)).solve(I);  // frozen at the anchor

  auto exact = [&](const Eigen::VectorXd& q, Eigen::VectorXd x) {
    for (int k = 0; k < 60; ++k) {
      const Eigen::VectorXd dx = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(cr.jacobian(x)).solve(cr.model_rates(x) - q);
      x -= dx;
      if (dx.cwiseAbs().maxCoeff() < 1e-14) break;
    }
    return x;
  };
  auto frozen_steps = [&](const Eigen::VectorXd& q, Eigen::VectorXd x) {
    int s = 0;
    for (; s < 99;) {
      const Eigen::VectorXd dx = M * (cr.model_rates(x) - q);
      x -= dx;
      ++s;
      if (dx.cwiseAbs().maxCoeff() < 1e-9) break;
    }
    return s;
  };

  std::printf("move_bp | rho=||I-M*J(x*)||2 | M err%% | steps one-shot(x0) | steps local(+0.15bp @ x*)\n");
  for (double bp : {1., 2., 5., 10., 20., 35., 50., 75., 100., 150., 200., 300.}) {
    const Eigen::VectorXd q = q0.array() + bp * 1e-4;  // parallel level move
    const Eigen::VectorXd xs = exact(q, x0);
    const double rho = (I - M * cr.jacobian(xs)).jacobiSvd().singularValues()(0);
    const int s_oneshot = frozen_steps(q, x0);
    const int s_local = frozen_steps((q.array() + 0.15e-4).matrix(), xs);
    std::printf("%7.0f | %17.4f | %6.1f | %18d | %d%s\n", bp, rho, rho * 100, s_oneshot, s_local,
                rho >= 1.0 ? "   <-- frozen-Newton DIVERGES (must recompute)" : "");
  }
  return 0;
}
