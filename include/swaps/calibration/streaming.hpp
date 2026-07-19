#pragma once
// Streaming re-calibrator for a live market feed (Stage 2). Holds a cached inverse Jacobian
// M = J(x_ref)^{-1} at the last point where J was computed, and the current solution x_cur.
//
// Two modes (Options::exact):
//
//  EXACT (default -- accuracy-first, for a live pricer to sharp clients):
//    Every tick runs frozen-Jacobian Gauss-Newton to convergence, REUSING the cached M:
//        x <- x - M*(model_rates(x) - q)   until ||dx||_inf < step_tol
//    The reused M is a Newton PRECONDITIONER, valid over a far larger market move than a single
//    linear step is accurate -- so each tick is solved EXACTLY (~step_tol), independent of move size.
//    J is recomputed ONLY when the frozen iteration stalls (M stale as a preconditioner), i.e. on the
//    staleness envelope (several bp), NOT the tiny linear-accuracy envelope. That is how Jacobian
//    recalcs are eliminated without any accuracy give-up.
//
//  LINEAR (legacy, for comparison): a single linear step x = x_anchor + M*dq while drift < envelope,
//    re-anchoring (recompute J) on every envelope crossing. Fast (one matvec) but O(drift^2) error.
//
// In both modes the streamed curve reprices the calibration instruments back to the input quotes; in
// EXACT mode the round-trip residual ||model_rates(x) - q||_inf is at the Newton tolerance every tick.

#include <Eigen/Dense>

#include <memory>

#include "swaps/calibration/background_jacobian.hpp"
#include "swaps/calibration/residual_engine.hpp"

namespace swaps::calibration {

struct StreamTick {
  bool refreshed = false;  // this tick recomputed the analytic Jacobian (>=1 refresh)
  int newton_steps = 0;    // total frozen Gauss-Newton steps taken this tick
  int refreshes = 0;       // analytic Jacobian recomputations this tick (0 on the pure fast path)
  int prefetched = 0;      // refreshes served from the BACKGROUND worker (no inline Jacobian this tick)
  double drift = 0;        // ||q_new - q_ref||_inf: market move since J was last computed
};

template <class Problem = CalibrationProblem>
class StreamingCalibrator {
 public:
  struct Options {
    double envelope = 1e-4;  // LINEAR mode: drift (rate units) that forces a re-anchor
    double step_tol = 1e-9;  // EXACT mode: converged when ||dx||_inf < step_tol
    int max_frozen = 4;      // EXACT mode: frozen steps without convergence -> refresh M (staleness)
    int max_refresh = 6;     // safety cap on refreshes within one tick
    bool exact = true;       // EXACT (iterate to step_tol) vs LINEAR (single step + drift re-anchor)
    // SPECULATIVE background Jacobian (EXACT mode): a dedicated thread pre-computes M as drift grows, so
    // a refresh swaps in a ready M (~µs) instead of computing J+factorize inline (0.5-20ms). Pure
    // tail-latency win; correctness is identical (frozen-Newton is exact for any invertible M).
    bool prefetch = false;
    double prefetch_drift = 5e-4;  // request a background M once drift from the anchor exceeds this (5bp)
  };

  StreamingCalibrator(const Problem& prob, const Eigen::VectorXd& x0,
                      const Eigen::VectorXd& q0, const Options& opt)
      : n_res_(prob.n_residuals()), engine_(prob), opt_(opt) {
    if (opt_.prefetch && opt_.exact) bg_ = std::make_unique<BackgroundJacobian<Problem>>(prob);
    set_anchor(x0, q0);
    x_cur_ = x0;
  }

  const Eigen::VectorXd& current() const { return x_cur_; }
  const Eigen::VectorXd& anchor_market() const { return q_anchor_; }
  const Eigen::MatrixXd& sensitivity() const { return M_; }
  int refresh_count() const { return refresh_count_; }
  int prefetch_hits() const { return prefetch_hits_; }  // refreshes served from the background worker

  StreamTick update(const Eigen::VectorXd& q_new) {
    return opt_.exact ? update_exact(q_new) : update_linear(q_new);
  }

 private:
  // EXACT: frozen-Newton to step_tol, reusing M; recompute J only when the iteration stalls.
  StreamTick update_exact(const Eigen::VectorXd& q_new) {
    StreamTick t;
    t.drift = (q_new - q_anchor_).cwiseAbs().maxCoeff();
    x_ = x_cur_;              // warm start from the last exact solution (tick-to-tick move is tiny)
    Eigen::VectorXd& x = x_;  // reused scratch: the frozen-Newton loop below allocates nothing
    int frozen = 0;
    for (;;) {
      r_.noalias() = engine_.model_rates(x) - q_new;  // engine reprice writes into its own scratch
      dx_.noalias() = M_ * r_;
      x.noalias() -= dx_;
      ++t.newton_steps;
      if (dx_.cwiseAbs().maxCoeff() < opt_.step_tol) break;  // EXACT reprice reached
      if (++frozen >= opt_.max_frozen) {                     // M stale as a preconditioner -> refresh
        if (t.refreshes >= opt_.max_refresh) break;          // safety (never hit on smooth feeds)
        // Prefer a Jacobian the BACKGROUND worker already computed (a ~µs matrix swap); fall back to an
        // inline compute only if none is ready. Adopting a slightly-stale background M is exact -- it
        // just sets the frozen-Newton rate -- so no freshness check is needed (a too-stale M merely
        // triggers another refresh, bounded by max_refresh).
        if (bg_ && bg_->try_take(bg_x_, bg_M_)) {
          M_ = std::move(bg_M_);
          x_anchor_ = std::move(bg_x_);
          q_anchor_ = q_new;  // reset drift from here
          ++refresh_count_;
          ++t.prefetched;
          ++prefetch_hits_;
        } else {
          set_anchor(x, q_new);  // inline compute (the spike this feature exists to hide)
        }
        t.refreshed = true;
        ++t.refreshes;
        frozen = 0;
      }
    }
    x_cur_ = x;
    // Speculatively pre-compute the NEXT M once the market has drifted enough that a refresh is plausibly
    // near -- so the worker is done (on a spare core) before the envelope is hit. Coalesces: a request
    // while one is in flight just updates the target x.
    if (bg_ && !bg_->computing() && t.drift > opt_.prefetch_drift) bg_->request(x_cur_);
    return t;
  }

  // LINEAR (legacy): single linear step inside the envelope, re-anchor (recompute J) outside it.
  StreamTick update_linear(const Eigen::VectorXd& q_new) {
    StreamTick t;
    const Eigen::VectorXd d = q_new - q_anchor_;
    t.drift = d.cwiseAbs().maxCoeff();
    if (t.drift < opt_.envelope) {
      x_cur_.noalias() = x_anchor_ + M_ * d;  // one matvec, O(drift^2) error
      return t;
    }
    t.refreshed = true;
    Eigen::VectorXd x = x_anchor_ + M_ * d;
    int frozen = 0;
    for (;;) {
      const Eigen::VectorXd r = engine_.model_rates(x) - q_new;
      const Eigen::VectorXd dx = M_ * r;
      x.noalias() -= dx;
      ++t.newton_steps;
      if (dx.cwiseAbs().maxCoeff() < opt_.step_tol) break;
      if (++frozen >= opt_.max_frozen) {
        if (t.refreshes >= opt_.max_refresh) break;
        set_anchor(x, q_new);
        ++t.refreshes;
        frozen = 0;
      }
    }
    set_anchor(x, q_new);  // legacy always refreshes at the converged solution
    x_cur_ = x;
    t.refreshes += 1;
    return t;
  }

  // M = (J^T J)^{-1} J^T (= J^{-1} when square). One engine Jacobian + a factor-solve.
  void set_anchor(const Eigen::VectorXd& x, const Eigen::VectorXd& q) {
    x_anchor_ = x;
    q_anchor_ = q;
    const Eigen::MatrixXd J = engine_.jacobian(x);
    M_ = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(J).solve(
        Eigen::MatrixXd::Identity(n_res_, n_res_));
    ++refresh_count_;
  }

  int n_res_;
  residual_engine_t<Problem> engine_;
  Options opt_;
  Eigen::VectorXd x_anchor_, q_anchor_, x_cur_;
  Eigen::MatrixXd M_;
  int refresh_count_ = 0;
  int prefetch_hits_ = 0;
  // Speculative background Jacobian (null unless opt_.prefetch): a dedicated thread computes the next M.
  std::unique_ptr<BackgroundJacobian<Problem>> bg_;
  Eigen::VectorXd bg_x_;  // scratch for a taken background result
  Eigen::MatrixXd bg_M_;
  // Per-tick scratch so the exact frozen-Newton loop allocates nothing (sized on first use).
  Eigen::VectorXd x_, r_, dx_;
};

}  // namespace swaps::calibration
