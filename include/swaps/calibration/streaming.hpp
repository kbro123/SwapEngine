#pragma once
// Streaming re-calibrator for a live market feed (Stage 2). Holds an ANCHOR (x_anchor solving market
// q_anchor, plus M = J(x_anchor)^{-1}). Each tick decides linear vs exact by how far the CURRENT
// market has drifted from the anchor market -- exactly the point where the Jacobian was last taken:
//
//   drift = ||q_new - q_anchor||_inf
//     drift < envelope :  x = x_anchor + M*(q_new - q_anchor)      one matvec, ~168 ns, error O(drift^2)
//     drift >= envelope:  RE-ANCHOR: refine to the exact solution at q_new, recompute J there, reset
//                         the anchor to (x*, q_new, J*^{-1}). Costs one AAD Jacobian.
//
// So the linear fast path runs as long as the market stays inside the envelope of the last Jacobian;
// a re-anchor happens only when it doesn't -- and resets the drift to zero.

#include <Eigen/Dense>

#include "swaps/calibration/compiled_residual.hpp"
#include "swaps/calibration/lm.hpp"

namespace swaps::calibration {

struct StreamTick {
  bool refreshed = false;  // this tick recomputed the Jacobian (re-anchored)
  int newton_steps = 0;    // refinement steps taken during a re-anchor (0 on the linear path)
  int refreshes = 0;       // AAD Jacobian recomputations during a re-anchor (>=1 if refreshed)
  double drift = 0;        // ||q_new - q_anchor||_inf before this tick
};

class StreamingCalibrator {
 public:
  struct Options {
    double envelope = 1e-4;   // drift (rate units) that triggers a re-anchor; ~1bp
    double step_tol = 1e-9;   // re-anchor refinement converged when ||dx||_inf < step_tol
    int max_frozen = 4;       // frozen-M refinement steps before refreshing M mid-re-anchor
    int max_refresh = 4;
  };

  StreamingCalibrator(const CalibrationProblem& prob, const Eigen::VectorXd& x0,
                      const Eigen::VectorXd& q0, const Options& opt)
      : prob_(&prob), cr_(prob), opt_(opt) {
    set_anchor(x0, q0);
    x_cur_ = x0;
  }

  const Eigen::VectorXd& current() const { return x_cur_; }
  const Eigen::VectorXd& anchor_market() const { return q_anchor_; }
  const Eigen::MatrixXd& sensitivity() const { return M_; }
  int refresh_count() const { return refresh_count_; }

  StreamTick update(const Eigen::VectorXd& q_new) {
    StreamTick t;
    const Eigen::VectorXd d = q_new - q_anchor_;
    t.drift = d.cwiseAbs().maxCoeff();
    if (t.drift < opt_.envelope) {
      x_cur_.noalias() = x_anchor_ + M_ * d;  // LINEAR fast path
      return t;
    }
    return reanchor(q_new, d);  // drifted out of the envelope -> exact re-solve + refresh
  }

 private:
  // M = (J^T J)^{-1} J^T at x (= J^{-1} when square). One ANALYTIC Jacobian + a factor-solve.
  void set_anchor(const Eigen::VectorXd& x, const Eigen::VectorXd& q) {
    x_anchor_ = x;
    q_anchor_ = q;
    const Eigen::MatrixXd J = cr_.jacobian(x);
    M_ = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(J).solve(
        Eigen::MatrixXd::Identity(prob_->n_residuals(), prob_->n_residuals()));
    ++refresh_count_;
  }

  StreamTick reanchor(const Eigen::VectorXd& q_new, const Eigen::VectorXd& d0) {
    StreamTick t;
    t.refreshed = true;
    t.drift = d0.cwiseAbs().maxCoeff();

    // Warm start from the linear prediction, then refine to the exact solution at q_new with the
    // current M (refreshing M if it stalls), before taking the fresh anchor Jacobian.
    Eigen::VectorXd x = x_anchor_ + M_ * d0;
    Eigen::MatrixXd Mref;
    const Eigen::MatrixXd* M = &M_;
    int frozen = 0, refreshes = 0;
    for (;;) {
      const Eigen::VectorXd r = cr_.model_rates(x) - q_new;
      const Eigen::VectorXd dx = (*M) * r;
      x.noalias() -= dx;
      ++t.newton_steps;
      ++frozen;
      if (dx.cwiseAbs().maxCoeff() < opt_.step_tol) break;
      if (frozen >= opt_.max_frozen) {
        if (refreshes >= opt_.max_refresh) break;
        const Eigen::MatrixXd J = cr_.jacobian(x);  // analytic, no AAD
        Mref = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(J).solve(
            Eigen::MatrixXd::Identity(prob_->n_residuals(), prob_->n_residuals()));
        M = &Mref;
        ++refreshes;
        ++refresh_count_;
        frozen = 0;
      }
    }
    set_anchor(x, q_new);  // fresh anchor Jacobian at the converged solution
    x_cur_ = x;
    t.refreshes = refreshes + 1;  // +1 for the anchor Jacobian
    return t;
  }

  const CalibrationProblem* prob_;
  CompiledResidual cr_;
  Options opt_;
  Eigen::VectorXd x_anchor_, q_anchor_, x_cur_;
  Eigen::MatrixXd M_;
  int refresh_count_ = 0;
};

}  // namespace swaps::calibration
