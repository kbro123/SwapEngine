#pragma once
// Streaming re-calibrator for a live market feed (Stage 2). Holds a cached operator M built from the
// Jacobian at the last anchor point, and the current solution x_cur.
//
// Two modes (Options::exact):
//
//  EXACT (default -- accuracy-first, for a live pricer to sharp clients):
//    Every tick runs frozen-Jacobian Gauss-Newton to convergence, REUSING the cached M:
//        x <- x - M*r(x, q)   until ||dx||_inf < step_tol
//    The reused M is a Newton PRECONDITIONER, valid over a far larger market move than a single
//    linear step is accurate -- so each tick is solved EXACTLY (~step_tol), independent of move size.
//    J is recomputed ONLY when the frozen iteration stalls (M stale as a preconditioner), i.e. on the
//    staleness envelope (several bp), NOT the tiny linear-accuracy envelope. That is how Jacobian
//    recalcs are eliminated without any accuracy give-up.
//
//    WHEN THE FIXED POINT IS THE ANSWER. The iteration's fixed point is J_refᵀ r(x) = 0 (J_ref the frozen
//    Jacobian). For a SQUARE full-rank problem that is r = 0: the exact reprice, for any J_ref. For an
//    over-determined / banded / regularised problem the least-squares condition is J(x)ᵀ r(x) = 0 with
//    the CURRENT Jacobian, and the two agree only while J(x) == J_ref. With the plain par-rate rows J
//    moves at second order (measured ≤0.015 bp over 30 bp of drift), so the frozen operator is exact to
//    that order. A BAND is the case that matters: its residual slope is `decay` inside the band and 1
//    outside (problem.hpp band_residual), so a quote crossing an edge changes that row of J by exactly
//    that factor -- a 10x moving target if left frozen (147 bp knot error was measured with the old
//    smooth ramp). Because the change is a pure per-row SCALE of a frozen quote Jacobian ∂q/∂x, the
//    streamer tracks it without recomputing anything expensive: it keeps J_ref, the slope each banded row
//    had at the anchor, and the slope it has now; when a row's side of the band changes it re-scales that
//    row of J by slope_now/slope_ref and re-factorises M (tens of µs at desk scale, no W-cache walk, no
//    AAD). The quote Jacobian stays frozen; only its band scaling follows the market. A row whose anchor
//    slope was 0 (decay = 0 inside the band) cannot be re-scaled and forces a full refresh instead.
//
//  LINEAR (legacy, for comparison): a single linear step x = x_anchor + M*dq while drift < envelope,
//    re-anchoring (recompute J) on every envelope crossing. Fast (one matvec) but O(drift^2) error.
//
// A tick that does NOT converge (refresh cap hit) is reported (StreamTick::converged == false) and NOT
// committed: current() keeps the last converged solution, so the next tick warm-starts from a good point
// and a caller never reads a half-solved curve as the answer.

#include <Eigen/Dense>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

#include "swaps/calibration/background_jacobian.hpp"
#include "swaps/calibration/problem.hpp"  // band_slope / band_residual_d, QuoteKind
#include "swaps/calibration/residual_engine.hpp"

#ifdef SWAPS_STREAM_TRACE
#include <cstdio>
#define SWAPS_TRACE(...) std::printf(__VA_ARGS__)
#else
#define SWAPS_TRACE(...) ((void)0)
#endif

namespace swaps::calibration {

// Why a tick ended. Anything but Converged is a FAILED tick: reported, never committed (current() keeps
// the last converged solution) and the anchor is restored to that solution, so the next tick starts from a
// valid state and cannot report a stale curve as converged (probe C-divergent-repeat, 2026-09-09).
enum class StreamStatus {
  Converged,   // ||dx||_inf < step_tol
  StepCap,     // Options::max_steps corrector steps without convergence
  RefreshCap,  // Options::max_refresh Jacobian refreshes without convergence (stalled)
  RescaleCap,  // the per-tick band re-scale budget was exhausted (active-set cycling)
  NonFinite,   // a residual, state or Jacobian entry became NaN/inf during the tick
  Diverged,    // the residual grew by 1e3x over the tick's first evaluation, or |x| left [-10, 10]
};
inline const char* to_string(StreamStatus s) {
  switch (s) {
    case StreamStatus::Converged: return "converged";
    case StreamStatus::StepCap: return "step cap hit";
    case StreamStatus::RefreshCap: return "refresh cap hit";
    case StreamStatus::RescaleCap: return "band re-scale budget exhausted";
    case StreamStatus::NonFinite: return "non-finite residual, state or Jacobian";
    case StreamStatus::Diverged: return "diverged";
  }
  return "?";
}

struct StreamTick {
  StreamStatus status = StreamStatus::StepCap;
  const char* reason() const { return to_string(status); }
  bool refreshed = false;  // this tick recomputed the analytic Jacobian (>=1 refresh)
  int newton_steps = 0;    // total frozen Gauss-Newton steps taken this tick
  int refreshes = 0;       // analytic Jacobian recomputations this tick (0 on the pure fast path)
  int prefetched = 0;      // refreshes served from the BACKGROUND worker (no inline Jacobian this tick)
  double drift = 0;        // ||q_new - q_ref||_inf: market move since J was last computed
  bool converged = false;  // ||dx||_inf < step_tol was reached (false: refresh cap hit; x NOT committed)
  int rescales = 0;        // band-edge crossings handled by re-scaling frozen J rows + re-factorising M
};

template <class Problem = CalibrationProblem>
class StreamingCalibrator {
 public:
  struct Options {
    double envelope = 1e-4;  // LINEAR mode: drift (rate units) that forces a re-anchor
    double step_tol = 1e-9;  // EXACT mode: converged when ||dx||_inf < step_tol
    int max_frozen = 4;      // EXACT mode: frozen steps without convergence -> refresh M (staleness)
    int max_refresh = 6;     // safety cap on refreshes within one tick
    int max_steps = 64;      // hard cap on corrector steps per tick (then: converged = false, not committed)
    bool exact = true;       // EXACT (iterate to step_tol) vs LINEAR (single step + drift re-anchor)
    // ACCURACY refresh for problems whose fixed point is NOT r = 0 (over-determined, banded, regularised):
    // there the frozen operator's answer is exact only to second order in the drift, and along a weakly
    // determined direction (a decay-weighted band) that second-order term is amplified -- measured 0.4 bp
    // at 30 bp of drift on an unregularised ±0.25 bp banded strip, with no stall to trigger a refresh. So
    // once the market has drifted this far from the anchor, refresh J at the start of the tick regardless
    // of convergence speed. A SQUARE full-rank problem is exact for any drift and is exempt (its fixed point
    // is r = 0 whatever M is), so the pure fast path is untouched.
    double refresh_drift = 1e-3;  // 10 bp
    // SPECULATIVE background Jacobian (EXACT mode): a dedicated thread pre-computes M as drift grows, so
    // a refresh swaps in a ready M (~µs) instead of computing J+factorize inline (0.5-20ms). Pure
    // tail-latency win; correctness is identical (frozen-Newton is exact for any invertible M).
    bool prefetch = false;
    double prefetch_drift = 5e-4;  // request a background M once drift from the anchor exceeds this (5bp)
    // SMOOTHNESS REGULARISER (empty = off). R = λ·D, the second-difference operator over the chosen
    // curves' knots (swaps::calibration::second_difference_operator). When set, the frozen-Newton
    // operator becomes M = (JᵀJ + RᵀR)⁻¹Jᵀ and each step gains a B·x curvature-pull term, so a
    // RANK-DEFICIENT (basis-only) build streams directly -- selecting the smoothest curve consistent
    // with the market every tick, instead of a singular M. Empty keeps the exact previous behaviour.
    Eigen::MatrixXd regularizer;
  };

  StreamingCalibrator(const Problem& prob, const Eigen::VectorXd& x0,
                      const Eigen::VectorXd& q0, const Options& opt)
      : n_res_(prob.n_residuals()), engine_(prob), opt_(opt) {
    if (!x0.allFinite()) throw std::invalid_argument("StreamingCalibrator: the anchor state x0 contains a non-finite value");
    if (!q0.allFinite()) throw std::invalid_argument("StreamingCalibrator: the anchor market q0 contains a non-finite quote");
    if (opt_.prefetch && opt_.exact) bg_ = std::make_unique<BackgroundJacobian<Problem>>(prob);
    if (opt_.regularizer.size()) RtR_.noalias() = opt_.regularizer.transpose() * opt_.regularizer;
    // The banded rows (FX forwards are never banded): what the per-tick side tracking watches.
    for (int i = 0; i < n_res_; ++i) {
      const Instrument& ins = prob.instruments[i];
      // decay > 0 only: the inside branch must be invertible (q from r) and the frozen row non-zero. A
      // decay-0 (dead-zone) band is not tracked; it still works through the stall-triggered refresh.
      if (ins.band_upper > ins.band_lower && ins.quote != QuoteKind::FxForward && ins.band_decay > 0.0)
        bands_.push_back({i, ins.band_lower, ins.band_upper, ins.band_decay});
    }
    slope_ref_.assign(bands_.size(), 1.0);
    slope_cur_.assign(bands_.size(), 1.0);
    flips_.assign(bands_.size(), 0);
    state_.assign(bands_.size(), 0);
    s_pin_.assign(bands_.size(), 0.0);
    last_side_.assign(bands_.size(), +1);
    releases_.assign(bands_.size(), 0);
    onedge_.assign(bands_.size(), 0);
    // The drift-triggered accuracy refresh applies only where the fixed point is not r = 0.
    drift_refresh_ = (n_res_ != static_cast<int>(x0.size())) || !bands_.empty() || RtR_.size() > 0;
    if (!set_anchor(x0, q0))
      throw std::invalid_argument("StreamingCalibrator: the Jacobian at the anchor state is non-finite");
    x_cur_ = x0;
    q_cur_ = q0;
  }

  const Eigen::VectorXd& current() const { return x_cur_; }
  const Eigen::VectorXd& anchor_market() const { return q_anchor_; }
  const Eigen::MatrixXd& sensitivity() const { return M_; }
  int refresh_count() const { return refresh_count_; }
  int prefetch_hits() const { return prefetch_hits_; }  // refreshes served from the background worker
  int rescale_count() const { return rescale_count_; }  // band-edge re-scales (cheap M refactors) so far

  StreamTick update(const Eigen::VectorXd& q_new) {
    if (q_new.size() != n_res_) throw std::invalid_argument("StreamingCalibrator::update: market length does not match the instrument count");
    if (!q_new.allFinite()) throw std::invalid_argument("StreamingCalibrator::update: market contains a non-finite quote");
    return opt_.exact ? update_exact(q_new) : update_linear(q_new);
  }

 private:
  struct BandRow {
    int row;
    double lower, upper, decay;
  };

  // EXACT: frozen-Newton to step_tol, reusing M; recompute J only when the iteration stalls.
  StreamTick update_exact(const Eigen::VectorXd& q_new) {
    StreamTick t;
    t.drift = (q_new - q_anchor_).cwiseAbs().maxCoeff();
    x_ = x_cur_;              // warm start from the last exact solution (tick-to-tick move is tiny)
    Eigen::VectorXd& x = x_;  // reused scratch: the frozen-Newton loop below allocates nothing
    if (drift_refresh_ && t.drift > opt_.refresh_drift) {  // accuracy refresh (see Options::refresh_drift)
      if (!refresh(x, q_new, t)) return fail(t, StreamStatus::NonFinite);
      t.drift = 0.0;
    }
    int frozen = 0;
    const bool drift_refreshed = t.refreshes > 0;
    bool final_refresh_done = false;
    for (std::size_t k = 0; k < flips_.size(); ++k) {
      flips_[k] = 0;
      releases_[k] = 0;  // the per-tick release budget (verify_pins) starts fresh every tick
    }
    double r0_inf = -1.0;  // the tick's first residual size: the divergence yardstick
    for (;;) {
      // The residual is engine-defined against the live market q_new: model_rates - q_new for hard
      // instruments, the Huber band residual for soft (banded) ones. Driving THIS (not the raw reprice)
      // is what makes frozen-Newton solve the soft least-squares -- dx = J⁺·r -> 0 at the soft minimum.
      r_ = engine_.residuals_vs(x, q_new);
      if (!r_.allFinite()) return fail(t, StreamStatus::NonFinite);
      const double r_inf = r_.cwiseAbs().maxCoeff();
      if (r0_inf < 0.0) r0_inf = r_inf;
      if (r_inf > 1e3 * std::max(r0_inf, 1e-2) || x.cwiseAbs().maxCoeff() > 10.0) return fail(t, StreamStatus::Diverged);
      // The band re-scale budget bounds active-set cycling within one tick. Exhausting it is a FAILED
      // tick: the old code kept stepping with the breakpoint search switched off, which cycled, then
      // refreshed into 1/decay-amplified overshoots (probe C-band-crossing, 0.6 bp at decay 0.1).
      if (t.rescales >= max_rescales()) return fail(t, StreamStatus::RescaleCap);
      if (!bands_.empty()) {
        // ACTIVE-SET band tracking (the objective is a convex piecewise-quadratic: each banded row has
        // slope decay inside its band and 1 outside, problem.hpp band_residual). Three states per row:
        //   FREE   -- slope follows the side. The residual value alone says which side the model quote is
        //             on (r is monotone in q). A side change re-scales that frozen row and re-factorises M
        //             (no Jacobian recompute), unless the row is unusable (M from the background worker
        //             without its J), in which case a full refresh.
        //   BREAKPOINT LINE SEARCH -- a decay-slope row's Gauss-Newton step is 1/decay-amplified and would
        //             overshoot the whole band; instead the step is taken only as far as the FIRST row that
        //             would cross an edge (predicted from its frozen quote row), the row's slope switches
        //             there, and the iteration continues. That is the exact minimiser walk for a convex
        //             piecewise quadratic and it never invents a crossing.
        //   PINNED -- a row that sits ON its edge and wants to cross back is at a kink optimum (generic
        //             for a piecewise-linear penalty; no fixed-slope Gauss-Newton converges to a kink). It
        //             is pinned: its residual becomes the linear extension through the edge at the
        //             MULTIPLIER slope s ∈ [decay, 1], and at convergence s is re-solved from the other
        //             rows' gradient (KKT); the row lands exactly on the edge for the right s, and is
        //             released to a side only if s leaves [decay, 1] -- at most once per tick per row
        //             (a row that walks back onto its edge after a release is a kink optimum the
        //             linearised multiplier misjudged by a hair: it is re-pinned for good, s clamped).
        apply_pins(q_new);
        if (track_bands(q_new)) {
          if (need_full_) {
            if (!set_anchor(x, q_new)) return fail(t, StreamStatus::NonFinite);
            t.refreshed = true;
            ++t.refreshes;
          } else {
            factor(J_cur_);
            ++t.rescales;
            ++rescale_count_;
          }
          frozen = 0;
        }
      }
      dx_.noalias() = M_ * r_;
      if (RtR_.size()) dx_.noalias() += B_ * x;  // curvature pull toward the smoothest market-consistent curve
      double alpha = 1.0;
      std::size_t hit = 0;
      int hit_side = 0;
      if (!bands_.empty() && have_J_) alpha = breakpoint(q_new, &hit, &hit_side);
      x.noalias() -= alpha * dx_;
      ++t.newton_steps;
      if (!x.allFinite()) return fail(t, StreamStatus::NonFinite);
      SWAPS_TRACE("  step %d |dx|=%.2e alpha=%.3f pinned=%d rescales=%d refreshes=%d\n", t.newton_steps, dx_.cwiseAbs().maxCoeff(), alpha, n_pinned_, t.rescales, t.refreshes);
      if (alpha < 1.0) {  // stopped on a band edge: switch that row there and carry on
        switch_row(hit, hit_side);
        factor(J_cur_);
        ++t.rescales;
        ++rescale_count_;
        frozen = 0;
        continue;
      }
      if (dx_.cwiseAbs().maxCoeff() < opt_.step_tol) {  // converged for the CURRENT active set
        if (!bands_.empty() && n_pinned_ > 0 && verify_pins(x, q_new)) {
          factor(J_cur_);  // a pin was released onto its true side, or its multiplier moved: iterate on
          ++t.rescales;
          ++rescale_count_;
          frozen = 0;
          continue;
        }
        // A drift refresh anchored J at the PREVIOUS solution; the frozen-J answer is then exact only to
        // second order in the move. One more anchor at the converged point and a re-converge (a genuine
        // Newton step with the fresh J) removes that term -- two Jacobians per refresh_drift of drift.
        // Not with the background prefetch on: there tail latency is the contract, and the second inline
        // Jacobian would be exactly the spike the worker exists to hide.
        if (drift_refreshed && !final_refresh_done && !bg_ && t.refreshes < opt_.max_refresh) {
          final_refresh_done = true;
          if (!set_anchor(x, q_new)) return fail(t, StreamStatus::NonFinite);
          ++t.refreshes;
          frozen = 0;
          continue;
        }
        t.converged = true;
        t.status = StreamStatus::Converged;
        break;
      }
      if (t.newton_steps >= opt_.max_steps) return fail(t, StreamStatus::StepCap);
      if (++frozen >= opt_.max_frozen) {  // M stale as a preconditioner -> refresh
        if (t.refreshes >= opt_.max_refresh) return fail(t, StreamStatus::RefreshCap);
        if (!refresh(x, q_new, t)) return fail(t, StreamStatus::NonFinite);
        frozen = 0;
      }
    }
    x_cur_ = x;  // committed: the exact solution at q_new
    q_cur_ = q_new;
    // Speculatively pre-compute the NEXT M once the market has drifted enough that a refresh is plausibly
    // near -- so the worker is done (on a spare core) before the envelope is hit. Coalesces: a request
    // while one is in flight just updates the target x.
    if (bg_ && !bg_->computing() && t.drift > opt_.prefetch_drift) bg_->request(x_cur_);
    return t;
  }

  // A FAILED tick: reported (status), never committed. If the tick moved the anchor (a Jacobian refresh at
  // a divergent iterate, a band re-scale or a pin), the anchor is rebuilt at the last committed solution
  // against the market it solved -- otherwise the next tick would run frozen-Newton off a Jacobian taken
  // at the divergent point (rank-collapsed to M == 0 in the probe), see |dx| == 0 and report a 1289 bp
  // stale curve as converged. Costs one Jacobian, only on the failed tick.
  StreamTick fail(StreamTick& t, StreamStatus why) {
    t.converged = false;
    t.status = why;
    SWAPS_TRACE("  tick FAILED: %s (steps %d refreshes %d rescales %d)\n", to_string(why), t.newton_steps, t.refreshes, t.rescales);
    if (t.refreshes > 0 || t.rescales > 0) (void)set_anchor(x_cur_, q_cur_);
    return t;
  }

  // LINEAR (legacy): single linear step inside the envelope, re-anchor (recompute J) outside it.
  StreamTick update_linear(const Eigen::VectorXd& q_new) {
    StreamTick t;
    const Eigen::VectorXd d = q_new - q_anchor_;
    t.drift = d.cwiseAbs().maxCoeff();
    if (t.drift < opt_.envelope) {
      x_cur_.noalias() = x_anchor_ + M_ * d;  // one matvec, O(drift^2) error
      t.converged = true;
      t.status = StreamStatus::Converged;
      return t;
    }
    t.refreshed = true;
    Eigen::VectorXd x = x_anchor_ + M_ * d;
    int frozen = 0;
    for (;;) {
      const Eigen::VectorXd r = engine_.residuals_vs(x, q_new);
      const Eigen::VectorXd dx = M_ * r;
      x.noalias() -= dx;
      ++t.newton_steps;
      if (dx.cwiseAbs().maxCoeff() < opt_.step_tol) {
        t.converged = true;
        t.status = StreamStatus::Converged;
        break;
      }
      if (++frozen >= opt_.max_frozen) {
        if (t.refreshes >= opt_.max_refresh) break;
        (void)set_anchor(x, q_new);
        ++t.refreshes;
        frozen = 0;
      }
    }
    (void)set_anchor(x, q_new);  // legacy always refreshes at the converged solution
    x_cur_ = x;
    t.refreshes += 1;
    return t;
  }

  // Which side of its band is each banded row on, read off the residual r_ (monotone in the model quote:
  // r > decay·(upper − m) means above, r < decay·(lower − m) means below, m = the live market). Updates
  // J_cur_ rows whose side changed; returns true if M must be rebuilt. Sets need_full_ when a row cannot
  // be re-scaled (anchor slope 0 => the frozen row is all zeros; or no J behind M).
  // Which side of its band is row k's model quote on, from the residual value (Huber, monotone in q).
  // Returns +1 above, -1 below, 0 inside; `q_out` receives the model quote recovered by inverting the
  // residual (decay > 0 for every tracked row, so the inside branch is invertible).
  int side_of(std::size_t k, const Eigen::VectorXd& q, double* q_out) const {
    const BandRow& b = bands_[k];
    const double m = q[b.row], r = r_[b.row];
    if (r > b.decay * (b.upper - m)) {
      if (q_out) *q_out = b.upper + (r - b.decay * (b.upper - m));
      return +1;
    }
    if (r < b.decay * (b.lower - m)) {
      if (q_out) *q_out = b.lower + (r - b.decay * (b.lower - m));
      return -1;
    }
    if (q_out) *q_out = m + r / b.decay;
    return 0;
  }

  // Pinned rows are EQUALITY CONSTRAINTS q_model = edge, imposed as a stiff penalty row
  //     r = w·(q_model − edge),  J row = w·(quote row),  w = kPinWeight,
  // so the frozen-Newton solve is the constrained Gauss-Newton step to O(1/w²) and the Lagrange
  // multiplier is read off the converged gap for free: λ = w²·gap (stationarity of the penalised least
  // squares: g_rest + Σ w²·gap_i·∇q_i = 0). The kink test is then s = λ / r_edge ∈ [decay, 1] with
  // r_edge = decay·(edge − m) the Huber residual AT the edge (verify_pins). This replaced (2026-09-10) a
  // fixed-point/secant iteration in the multiplier slope s that wandered without closing the gap and
  // left the tick 14 % above the optimum (probe C-band-crossing, 0.6 bp at decay 0.1).
  static constexpr double kPinWeight = 1e3;  // gap ≈ λ/w² ~ 1e-11 (1e-7 bp); λ resolved to ~1e-8 relative
  void apply_pins(const Eigen::VectorXd& q) {
    if (n_pinned_ == 0) return;
    for (std::size_t k = 0; k < bands_.size(); ++k) {
      if (state_[k] == 0) continue;
      const BandRow& b = bands_[k];
      double qm = 0.0;
      side_of(k, q, &qm);
      const double edge = state_[k] > 0 ? b.upper : b.lower;
      r_[b.row] = kPinWeight * (qm - edge);
    }
  }

  // The first band edge the full step x − dx would cross, predicted from the frozen quote rows
  // (Δq_k = −Jq_k·dx). Returns the fraction α ∈ (0, 1] of the step to take (1 = no crossing); on a
  // crossing `*hit` is the row and `*hit_side` the side it is entering (0 inside, ±1 outside).
  double breakpoint(const Eigen::VectorXd& q, std::size_t* hit, int* hit_side) {
    double alpha = 1.0;
    for (std::size_t k = 0; k < bands_.size(); ++k) {
      if (state_[k] != 0 || onedge_[k]) continue;  // pinned, or sitting on its edge (handled by track_bands)
      const BandRow& b = bands_[k];
      double qm = 0.0;
      const int side = side_of(k, q, &qm);
      const double dq = -(J_ref_.row(b.row).dot(dx_)) / slope_ref_[k];  // predicted move of the model quote
      if (dq == 0.0) continue;
      double a = 2.0;
      int into = 0;
      if (side == 0) {
        if (dq > 0.0 && qm < b.upper) { a = (b.upper - qm) / dq; into = +1; }
        if (dq < 0.0 && qm > b.lower) { a = (b.lower - qm) / dq; into = -1; }
      } else if (side > 0 && dq < 0.0) {
        a = (b.upper - qm) / dq;
      } else if (side < 0 && dq > 0.0) {
        a = (b.lower - qm) / dq;
      }
      if (a > 0.0 && a < alpha) {
        alpha = a;
        *hit = k;
        *hit_side = into;
      }
    }
    return alpha;
  }

  // Row k has just been walked onto its edge (by the breakpoint step) and is entering `side`: switch its
  // slope there. A row walked onto an edge for the SECOND time in a tick, from the other side, is at a
  // kink optimum: pin it on that edge (the kink is where the two walks meet).
  void switch_row(std::size_t k, int side) {
    const BandRow& b = bands_[k];
    ++flips_[k];
    onedge_[k] = true;
    const bool pin = flips_[k] >= 2;  // twice onto the same edge in one tick: a kink optimum
    SWAPS_TRACE("  row %d walked onto edge, entering side %d, flips=%d pin=%d\n", b.row, side, flips_[k], (int)pin);
    if (pin) {
      // The edge it is on: the one it is crossing now (entering outside => that side's edge; entering
      // inside => the edge on the side it came from).
      const int edge_side = (side != 0) ? side : last_side_[k];
      state_[k] = edge_side > 0 ? +1 : -1;
      ++n_pinned_;
      s_pin_[k] = 0.0;  // the multiplier slope is known only at convergence (verify_pins)
      J_cur_.row(b.row) = J_ref_.row(b.row) * (kPinWeight / slope_ref_[k]);
      slope_cur_[k] = kPinWeight;
      return;
    }
    const double slope = side == 0 ? b.decay : 1.0;
    J_cur_.row(b.row) = J_ref_.row(b.row) * (slope / slope_ref_[k]);
    slope_cur_[k] = slope;
    if (side != 0) last_side_[k] = side;
  }

  // FREE rows: reconcile the installed slope with the side the model quote is ACTUALLY on (the breakpoint
  // prediction is linear; the true quote map is not, so a walked-onto edge can land a hair either side --
  // and a row not on an edge can still change side under a full step). Updates J_cur_ rows whose slope
  // changed; returns true if M must be rebuilt. Sets need_full_ when a row cannot be re-scaled (no J
  // behind M: it came from the background worker).
  bool track_bands(const Eigen::VectorXd& q) {
    bool changed = false;
    need_full_ = false;
    for (std::size_t k = 0; k < bands_.size(); ++k) {
      if (state_[k] != 0) continue;  // pinned: handled by apply_pins / verify_pins
      const BandRow& b = bands_[k];
      double qm = 0.0;
      const int side = side_of(k, q, &qm);
      const double slope = side == 0 ? b.decay : 1.0;
      // On-edge rows: the installed slope is the side we are ENTERING; the actual side may still read as
      // the side we came from by a rounding hair. Treat "within kink_tol of the edge" as on the edge.
      if (onedge_[k]) {
        const double kink_tol = 1e-3 * (b.upper - b.lower);
        if (std::abs(qm - b.upper) < kink_tol || std::abs(qm - b.lower) < kink_tol) continue;
        onedge_[k] = false;  // moved clearly off the edge: back to plain side tracking
      }
      if (slope == slope_cur_[k]) continue;
      changed = true;
      if (!have_J_) {
        need_full_ = true;
        slope_cur_[k] = slope;
        continue;
      }
      SWAPS_TRACE("  row %d actual side %d slope %.2f->%.2f\n", b.row, side, slope_cur_[k], slope);
      J_cur_.row(b.row) = J_ref_.row(b.row) * (slope / slope_ref_[k]);
      slope_cur_[k] = slope;
      if (side != 0) last_side_[k] = side;
    }
    return changed;
  }

  // KKT check of every pinned row at a converged point. The penalised least squares is stationary:
  //     g_rest + Σ_i λ_i ∇q_i = 0,   λ_i = w²·gap_i,   gap_i = q_model_i − edge_i,
  // the constrained problem's multipliers. Along the constrained-optimal path a move δ of q_i changes the
  // rest of the objective by −λ_i·δ and the row's own Huber term by r_edge·slope·δ (slope = decay on the
  // inside of the edge, 1 outside), so the edge is a kink optimum iff s_i = λ_i / r_edge_i ∈ [decay, 1]
  // (both edges, either sign of r_edge). Otherwise the true optimum lies on the side the multiplier
  // points to: release the row there (slope decay if s < decay, 1 if s > 1) and iterate on -- once per
  // row per tick; a row that walks back onto its edge after a release is re-pinned for good.
  // Returns true if any row was released (M must be rebuilt).
  bool verify_pins(const Eigen::VectorXd& x, const Eigen::VectorXd& q) {
    (void)x;
    r_ = engine_.residuals_vs(x, q);  // the engine's (Huber) residuals: pinned rows' q_model read off these
    bool changed = false;
    for (std::size_t k = 0; k < bands_.size(); ++k) {
      if (state_[k] == 0) continue;
      const BandRow& b = bands_[k];
      double qm = 0.0;
      side_of(k, q, &qm);
      const double edge = state_[k] > 0 ? b.upper : b.lower;
      const double gap = qm - edge;
      const double lambda = kPinWeight * kPinWeight * gap;
      const double r_edge = b.decay * (edge - q[b.row]);
      double sj;
      if (std::abs(r_edge) > 1e-14) {
        sj = lambda / r_edge;
      } else {
        // Market ON the edge: no kink (the Huber term is C1 there). Release toward the side the rest of
        // the objective pulls to: λ > 0 means it wants q_i higher (the + side of the edge).
        sj = (state_[k] * lambda > 0.0) ? 2.0 : 0.0;
      }
      s_pin_[k] = sj;
      SWAPS_TRACE("  verify row %d state %d s=%.4f (decay %.2f) gap=%.2e lambda=%.3e\n", b.row, state_[k], sj, b.decay, gap, lambda);
      // Hysteresis: the multiplier is read off a frozen-J solve, so a hair outside [decay, 1] is noise,
      // not a side (the old 1e-6 released at s = decay − 9e-4 and cycled).
      const double tol = 1e-3;
      if ((sj < b.decay - tol || sj > 1.0 + tol) && releases_[k] == 0) {
        const double slope = (sj < b.decay) ? b.decay : 1.0;
        state_[k] = 0;
        --n_pinned_;
        ++releases_[k];  // one release per row per tick; a re-pin after it is final (termination)
        J_cur_.row(b.row) = J_ref_.row(b.row) * (slope / slope_ref_[k]);
        slope_cur_[k] = slope;
        changed = true;
      }
    }
    return changed;
  }

  // Per-tick band re-scale budget (each is a factor()); exhausting it fails the tick (RescaleCap).
  int max_rescales() const { return 8 + 4 * static_cast<int>(bands_.size()); }

  void clear_pins() {
    for (std::size_t k = 0; k < bands_.size(); ++k) {
      state_[k] = 0;
      flips_[k] = 0;
      releases_[k] = 0;
      onedge_[k] = false;
    }
    n_pinned_ = 0;
  }

  // One Jacobian refresh (either trigger: stall or drift). Prefer a Jacobian the BACKGROUND worker
  // already computed (a ~µs matrix swap); fall back to an inline compute only if none is ready. Adopting
  // a slightly-stale background M is exact for a square problem -- it just sets the frozen-Newton rate --
  // so no freshness check is needed (a too-stale M merely triggers another refresh, bounded by
  // max_refresh). The worker hands over M only, so band rows cannot be re-scaled off it (have_J_).
  bool refresh(const Eigen::VectorXd& x, const Eigen::VectorXd& q, StreamTick& t) {
    if (bg_ && bg_->try_take(bg_x_, bg_M_)) {
      M_ = std::move(bg_M_);
      x_anchor_ = std::move(bg_x_);
      q_anchor_ = q;  // reset drift from here
      have_J_ = false;
      ++refresh_count_;
      ++t.prefetched;
      ++prefetch_hits_;
    } else {
      if (!set_anchor(x, q)) return false;  // inline compute (the spike the prefetch exists to hide)
    }
    t.refreshed = true;
    ++t.refreshes;
    return true;
  }

  // Full refresh: the engine's Jacobian at (x, q) -- band rows carry their slope AT x -- plus the anchor
  // slopes, then the operator.
  // Returns false (anchor UNCHANGED) if the Jacobian at (x, q) is not finite.
  bool set_anchor(const Eigen::VectorXd& x, const Eigen::VectorXd& q) {
    SWAPS_TRACE("  set_anchor\n");
    J_ref_ = engine_.jacobian_vs(x, q);  // band term consistent with residuals_vs(·,q)
    if (!J_ref_.allFinite()) {
      if (have_J_) J_ref_ = J_cur_;  // keep the previous anchor usable (J_cur_ is its re-scaled copy)
      return false;
    }
    x_anchor_ = x;
    q_anchor_ = q;
    if (!bands_.empty()) {
      const Eigen::VectorXd& mr = engine_.model_rates(x);
      for (std::size_t k = 0; k < bands_.size(); ++k) {
        const BandRow& b = bands_[k];
        slope_ref_[k] = band_slope(mr[b.row], b.lower, b.upper, b.decay);  // > 0: tracked rows have decay > 0
        slope_cur_[k] = slope_ref_[k];
        last_side_[k] = (mr[b.row] > b.upper) ? +1 : (mr[b.row] < b.lower ? -1 : +1);
      }
      clear_pins();
    }
    J_cur_ = J_ref_;
    have_J_ = true;
    factor(J_cur_);
    ++refresh_count_;
    return true;
  }

  // M = (JᵀJ)⁻¹Jᵀ (= J⁻¹ when square). With a smoothness regulariser R, M = (JᵀJ + RᵀR)⁻¹Jᵀ and
  // B = (JᵀJ + RᵀR)⁻¹RᵀR (the per-step curvature pull). BOTH branches are RANK-SAFE: a rank-thresholded
  // complete orthogonal decomposition (kRankThreshold) -- of J, or of the stacked [J; R] whose pseudo-
  // inverse P gives (JᵀJ + RᵀR)⁺ = P·Pᵀ. A bundle can be legitimately rank-deficient (a knot no instrument
  // pins, redundant xccy/basis rows), and R's null space (level + linear forward moves) can meet J's when
  // a curve has knots but no level-pinning row: a tolerance-free LDLT of that singular normal matrix used
  // to send every tick to the refresh cap and walk the unpinned curve to negative forwards. The COD
  // zeroes the null directions instead, so an unpinned state never moves off its anchor.
  void factor(const Eigen::MatrixXd& J) {
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
    cod.setThreshold(kRankThreshold);
    if (RtR_.size()) {
      Eigen::MatrixXd S(J.rows() + opt_.regularizer.rows(), J.cols());
      S << J, opt_.regularizer;
      cod.compute(S);
      const Eigen::MatrixXd P = cod.pseudoInverse();  // n_knots x (n_res + n_reg)
      M_ = P.leftCols(J.rows());
      B_.noalias() = (P * P.transpose()) * RtR_;
    } else {
      cod.compute(J);
      M_ = cod.solve(Eigen::MatrixXd::Identity(n_res_, n_res_));
    }
  }

  int n_res_;
  residual_engine_t<Problem> engine_;
  Options opt_;
  Eigen::VectorXd x_anchor_, q_anchor_, x_cur_;
  Eigen::VectorXd q_cur_;  // the market x_cur_ solves (a failed tick restores the anchor here)
  Eigen::MatrixXd M_;
  Eigen::MatrixXd RtR_, B_;  // smoothness regulariser: RᵀR and the per-step curvature pull B=(JᵀJ+RᵀR)⁻¹RᵀR
  int refresh_count_ = 0;
  int prefetch_hits_ = 0;
  int rescale_count_ = 0;
  // Band-edge tracking: the frozen anchor Jacobian, its current re-scaled copy, and per banded row the
  // slope at the anchor / now. have_J_ is false when M was adopted from the background worker.
  std::vector<BandRow> bands_;  // tracked banded rows (decay > 0; a decay-0 band is left to the stall refresh)
  std::vector<double> slope_ref_, slope_cur_;
  std::vector<int> flips_;      // per banded row: side flips within the current tick
  std::vector<int> state_;      // 0 free, +1 pinned on upper, -1 pinned on lower
  std::vector<double> s_pin_;   // pinned rows: the multiplier slope s = λ/r_edge read at the last KKT check
  std::vector<int> last_side_;  // last OUTSIDE side seen (+1 above / -1 below): which edge a flip crosses
  std::vector<int> releases_;   // pin releases by the KKT check this tick (budget 1: a re-pin is final)
  std::vector<char> onedge_;    // row was walked onto its edge by the breakpoint step (side = installed slope)
  int n_pinned_ = 0;
  Eigen::MatrixXd J_ref_, J_cur_;
  bool have_J_ = false;
  bool need_full_ = false;
  bool drift_refresh_ = false;  // Options::refresh_drift applies (non-square / banded / regularised)
  // Speculative background Jacobian (null unless opt_.prefetch): a dedicated thread computes the next M.
  std::unique_ptr<BackgroundJacobian<Problem>> bg_;
  Eigen::VectorXd bg_x_;  // scratch for a taken background result
  Eigen::MatrixXd bg_M_;
  // Per-tick scratch so the exact frozen-Newton loop allocates nothing (sized on first use).
  Eigen::VectorXd x_, r_, dx_;
};

}  // namespace swaps::calibration
