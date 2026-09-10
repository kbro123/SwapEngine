#pragma once
// Speculative background Jacobian for the streaming calibrator (research → build, CLAUDE.md §7a). The
// streaming refresh (recompute J, factorize M = J^{-1}) is the one expensive spike on an otherwise ~µs
// tick: 0.5 ms (non-linear single curve, AAD) up to ~20 ms (an 8-curve bundle). Done SYNCHRONOUSLY it
// freezes a live pricer on the refresh tick. This worker moves that compute to a DEDICATED thread: as
// drift approaches the staleness envelope the main thread `request`s an M at the current x; the worker
// computes it off the critical path; when the refresh actually fires the main thread `try_take`s the
// ready M and swaps it in (~µs) instead of computing inline.
//
// SAFETY comes for free from the frozen-Newton fixed point: `x ← x − M·(model_rates(x) − q)` converges to
// the EXACT solution for ANY invertible M -- M only sets the convergence RATE. So adopting a slightly
// stale background M is never wrong; worst case it needs an extra frozen step (or a follow-up refresh),
// which the streaming loop already handles. No accuracy is traded for the latency win.
//
// The worker owns its OWN residual engine (residual_engine_t<Problem>) -- the compiled engines carry
// mutable per-instance scratch, so it must NOT share the calibrator's engine (same rule as the async
// pricer needing its own CompiledResidual). The problem must outlive the worker.

#include <Eigen/Dense>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>

#include "swaps/calibration/residual_engine.hpp"

namespace swaps::calibration {

template <class Problem>
class BackgroundJacobian {
 public:
  explicit BackgroundJacobian(const Problem& p)
      : engine_(p), n_res_(p.n_residuals()), worker_([this] { loop(); }) {}

  ~BackgroundJacobian() {
    {
      std::lock_guard<std::mutex> lk(req_m_);
      stop_ = true;
    }
    req_cv_.notify_one();
    worker_.join();
  }

  BackgroundJacobian(const BackgroundJacobian&) = delete;
  BackgroundJacobian& operator=(const BackgroundJacobian&) = delete;

  // MAIN THREAD. Ask the worker to compute M = J(x)^{-1} at x. Non-blocking; coalescing (only the LATEST
  // pending x is kept -- an in-flight compute finishes, then the worker picks up the newest request).
  void request(const Eigen::VectorXd& x) {
    {
      std::lock_guard<std::mutex> lk(req_m_);
      req_x_ = x;
      have_req_ = true;
    }
    req_cv_.notify_one();
  }

  // MAIN THREAD. If a computed M is ready, move it out (M and the x it was computed at) and return true.
  // Non-blocking (a brief lock; the worker holds it only to hand off the result).
  bool try_take(Eigen::VectorXd& x_out, Eigen::MatrixXd& M_out) {
    std::lock_guard<std::mutex> lk(res_m_);
    if (!have_res_) return false;
    x_out = std::move(res_x_);
    M_out = std::move(res_M_);
    have_res_ = false;
    return true;
  }

  // A compute is currently running (so a request now would just coalesce behind it).
  bool computing() const { return computing_.load(std::memory_order_acquire); }
  bool ready() const {
    std::lock_guard<std::mutex> lk(res_m_);
    return have_res_;
  }

 private:
  void loop() {
    for (;;) {
      Eigen::VectorXd x;
      {
        std::unique_lock<std::mutex> lk(req_m_);
        req_cv_.wait(lk, [this] { return have_req_ || stop_; });
        if (stop_) return;
        x = std::move(req_x_);
        have_req_ = false;
      }
      computing_.store(true, std::memory_order_release);
      const Eigen::MatrixXd J = engine_.jacobian(x);  // the expensive part (AAD or analytic W-cache)
      // RANK-SAFE, at the ONE shared threshold (E3-B2 / G5, 2026-09-10): the same complete orthogonal
      // decomposition StreamingCalibrator::factor uses, so a rank-deficient square bundle gets the
      // minimum-norm pseudo-inverse here too (an un-thresholded QR produced |M| up to 6e14).
      Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
      cod.setThreshold(kRankThreshold);
      cod.compute(J);
      Eigen::MatrixXd M = cod.solve(Eigen::MatrixXd::Identity(n_res_, n_res_));
      {
        std::lock_guard<std::mutex> lk(res_m_);
        res_x_ = std::move(x);
        res_M_ = std::move(M);
        have_res_ = true;
      }
      computing_.store(false, std::memory_order_release);
    }
  }

  residual_engine_t<Problem> engine_;  // OWN engine (own scratch) -- not shared with the calibrator
  int n_res_;

  mutable std::mutex req_m_;
  std::condition_variable req_cv_;
  Eigen::VectorXd req_x_;
  bool have_req_ = false;
  bool stop_ = false;

  mutable std::mutex res_m_;
  Eigen::VectorXd res_x_;
  Eigen::MatrixXd res_M_;
  bool have_res_ = false;

  std::atomic<bool> computing_{false};
  std::thread worker_;  // constructed LAST (all state above is ready before the loop runs)
};

}  // namespace swaps::calibration
