#pragma once
// Warm re-calibration with automatic Jacobian-reuse envelope detection (Stage 2).
//
// A full LM re-solve spends ~all its time re-computing the AAD Jacobian every iteration. Under a
// SMALL market perturbation, warm-started from a solved curve, the Jacobian barely moves, so we
// cache J0 (and its factorization) at the base solution and take frozen-Jacobian Gauss-Newton steps
// -- each step is one residual evaluation plus a back-substitution (microseconds), no AAD.
//
// But frozen-J Newton only converges fast inside an ENVELOPE around the base: its error grows ~like
// (perturbation)^2. So the calibrator watches its OWN convergence -- if the step size is not falling
// below tolerance within `max_frozen` steps, J0 is stale (perturbation outside the envelope) and it
// automatically REFRESHES the Jacobian at the current point and continues. The result reports
// `jacobian_refreshes`: 0 means the fast path held; >0 means the perturbation was auto-detected as
// outside the reuse envelope and handled. Refresh costs one AAD Jacobian; frozen steps are ~free.
//
// dq is the market perturbation (q_new - q_base) in rate units, residual order (averaged futures,
// compounded futures, swaps): residual(x) = residual_base(x) - dq. No problem copy needed.

#include <Eigen/Dense>

#include <limits>

#include "swaps/calibration/compiled_residual.hpp"
#include "swaps/calibration/lm.hpp"

namespace swaps::calibration {

struct WarmResult {
  Eigen::VectorXd x;
  int steps = 0;               // total Gauss-Newton steps taken
  int jacobian_refreshes = 0;  // 0 => stayed inside the J0-reuse envelope (fast path)
  bool converged = false;
  double final_step = 0.0;     // last ||dx||_inf
};

// Specialised to CalibrationProblem: it uses the vectorized CompiledResidual (DF = exp(-Wx)) for the
// hot inner-loop residual evaluations, and the AAD problem only for the (rare) Jacobian refresh.
class WarmCalibrator {
 public:
  struct Options {
    double step_tol = 1e-8;  // converged when ||dx||_inf < step_tol
    int max_frozen = 3;      // frozen-J steps before declaring J0 stale and refreshing
    int max_refresh = 3;     // cap on Jacobian refreshes (then give up / fall through)
  };

  WarmCalibrator(const CalibrationProblem& prob, const Eigen::VectorXd& x_base)
      : prob_(&prob), x0_(x_base), cr_(prob), qr0_(cr_.jacobian(x_base)) {
    // J0 via the ANALYTIC Jacobian (no AAD). First-order curve sensitivity M = (J^T J)^{-1} J^T
    // (= J^{-1} when square) -- the SAME operator as the analytic risk ladder dx/dq. Precomputed once.
    M_ = qr0_.solve(Eigen::MatrixXd::Identity(prob.n_residuals(), prob.n_residuals()));
  }

  int n_knots() const { return static_cast<int>(x0_.size()); }
  const Eigen::VectorXd& base() const { return x0_; }
  const Eigen::MatrixXd& sensitivity() const { return M_; }

  // FIRST-ORDER live-tick update: x = x0 + M*dq, a single matvec, no residual eval. Because
  // r(x0) = -dq exactly at the base, this IS the first frozen Newton step. Error is O(|dq|^2) --
  // sub-microsecond, ideal for small real-time market moves where ~1e-6 curve accuracy is enough.
  Eigen::VectorXd recalibrate_linear(const Eigen::VectorXd& dq) const { return x0_ + M_ * dq; }

  WarmResult recalibrate(const Eigen::VectorXd& dq) const { return recalibrate(dq, Options{}); }

  // Adaptive re-calibration with automatic envelope detection (the safe default). The residual is the
  // vectorized CompiledResidual; only a refresh touches the AAD Jacobian.
  WarmResult recalibrate(const Eigen::VectorXd& dq, const Options& opt) const {
    WarmResult res;
    Eigen::VectorXd x = x0_;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> refreshed;  // used only if we leave the envelope
    const Eigen::ColPivHouseholderQR<Eigen::MatrixXd>* J = &qr0_;
    int frozen = 0;

    while (true) {
      const Eigen::VectorXd dx = J->solve(cr_.residuals(x) - dq);
      x.noalias() -= dx;
      ++res.steps;
      ++frozen;
      res.final_step = dx.cwiseAbs().maxCoeff();

      if (res.final_step < opt.step_tol) {
        res.converged = true;
        break;
      }
      // Envelope detection: too many frozen steps without hitting tolerance => J0 is stale.
      if (frozen >= opt.max_frozen) {
        if (res.jacobian_refreshes >= opt.max_refresh) break;  // give up (caller may fall back)
        refreshed = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(cr_.jacobian(x));  // analytic, no AAD
        J = &refreshed;
        ++res.jacobian_refreshes;
        frozen = 0;
      }
    }
    res.x = std::move(x);
    return res;
  }

  // Fixed-step fast path: `iters` frozen-J0 steps, no adaptivity. For the hot loop when the caller
  // KNOWS the perturbation is inside the envelope (e.g. a live tick). Prefer recalibrate() otherwise.
  Eigen::VectorXd recalibrate_fixed(const Eigen::VectorXd& dq, int iters = 2) const {
    Eigen::VectorXd x = x0_;
    for (int k = 0; k < iters; ++k) x.noalias() -= qr0_.solve(cr_.residuals(x) - dq);
    return x;
  }

 private:
  const CalibrationProblem* prob_;
  Eigen::VectorXd x0_;
  CompiledResidual cr_;                              // vectorized residual + analytic Jacobian
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr0_;  // factorization of J0 (base Jacobian)
  Eigen::MatrixXd M_;                                // first-order sensitivity (J^T J)^{-1} J^T
};

}  // namespace swaps::calibration
