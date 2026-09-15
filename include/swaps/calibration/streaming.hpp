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
// A tick that does NOT converge (refresh cap hit) is reported (StreamTick::converged == false) and NOT
// committed: current() keeps the last converged solution, so the next tick warm-starts from a good point
// and a caller never reads a half-solved curve as the answer.

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
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

// HOT-PATH CENSUS (2026-09-15). Every stage a streaming tick can pass through, stamped into StreamTick::stages as one bit (an
// integer OR per event, no allocation). tests/hotpath_census_test.cpp drives a committed scenario per stage against
// tests/hotpath_census.lock (with the scenario's allocation / factorisation pins), and tools/check_hotpath_census.py locks the
// streamer's refresh / factorisation call sites: a new stage or site cannot land without a scenario and a pin.
enum class StreamStage : unsigned {
  DriftRefresh, TrackRescale, TrackRefresh, Breakpoint, Pin, Release, RescaleFallback, KinkDamp, Predicted, StallRefresh, FrozenCapRefresh, CommitReanchor, FinalRefresh, Truncated, PrefetchTake, Fail, FailRestore,
  kCount
};
inline const char* to_string(StreamStage s) {
  switch (s) {
    case StreamStage::DriftRefresh: return "DriftRefresh";
    case StreamStage::TrackRescale: return "TrackRescale";
    case StreamStage::TrackRefresh: return "TrackRefresh";
    case StreamStage::Breakpoint: return "Breakpoint";
    case StreamStage::Pin: return "Pin";
    case StreamStage::Release: return "Release";
    case StreamStage::RescaleFallback: return "RescaleFallback";
    case StreamStage::KinkDamp: return "KinkDamp";
    case StreamStage::Predicted: return "Predicted";
    case StreamStage::StallRefresh: return "StallRefresh";
    case StreamStage::FrozenCapRefresh: return "FrozenCapRefresh";
    case StreamStage::CommitReanchor: return "CommitReanchor";
    case StreamStage::FinalRefresh: return "FinalRefresh";
    case StreamStage::Truncated: return "Truncated";
    case StreamStage::PrefetchTake: return "PrefetchTake";
    case StreamStage::Fail: return "Fail";
    case StreamStage::FailRestore: return "FailRestore";
    case StreamStage::kCount: break;
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
  double drift = 0;        // ||q_new - q_anchor||_inf: the market move since the anchor this tick started from -- reported
                           // also when that move triggered the accuracy refresh (`refreshed`); until 2026-09-15 such a
                           // tick zeroed it (C8)
  bool converged = false;  // ||dx||_inf < step_tol was reached (false: refresh cap hit; x NOT committed)
  int rescales = 0;        // band-edge crossings handled by re-scaling frozen J rows + re-factorising M
  unsigned stages = 0;     // StreamStage bits this tick passed through (the hot-path census)
};

template <class Problem = CalibrationProblem>
class StreamingCalibrator {
 public:
  struct Options {
    double step_tol = 1e-9;  // EXACT mode: converged when ||dx||_inf < step_tol
    int max_frozen = 8;      // EXACT mode: hard cap on frozen steps without convergence -> refresh M
    int max_refresh = 6;     // safety cap on refreshes within one tick
    int max_steps = 64;      // hard cap on corrector steps per tick (then: converged = false, not committed)
    // PREDICTIVE CONVERGENCE (2026-09-10, E4.D): after two consecutive full steps under the SAME frozen
    // operator (no refresh, re-scale or breakpoint between them) the next step is bounded by the observed
    // contraction, |dx_{k+1}| <= (|dx_k| / |dx_{k-1}|) * |dx_k| (exact for the linear rate a frozen M
    // converges at; conservative for the quadratic rate of a fresh one). When that bound is below
    // step_tol / 20 the tick stops here instead of spending a third residual evaluation + solve just to
    // read a ~1e-15 step. The committed x is within ~step_tol / 10 of the fully iterated one (measured
    // 1.2e-10 at step_tol 1e-9 on the chain fixture; streaming_contract_test pins 1e-10) and its reprice
    // stays inside the step_tol contract. Ladder ticks 0.68x on every compiled rung. false = iterate to the
    // measured step every time (the reference behaviour).
    bool predict_convergence = true;
    // BAND RE-SCALE by a rank-one operator update (E3-C7, 2026-09-10; see rescale_row) instead of a full
    // re-factorisation. false = re-factorise on every re-scale (the reference the parity test compares to).
    bool rescale_update = true;
    // ANCHOR SIDES FROM THE JACOBIAN PASS (S2, 2026-09-15): a refresh classifies each tracked band row from the residual values the
    // engine's Jacobian pass already has (inverted as the tick's side_of does) instead of re-evaluating every model quote -- on desk_mixed
    // that re-evaluation cost 192 us and 23 allocations per refresh. false = the model_rates reference the parity test compares to.
    bool anchor_sides_from_jacobian = true;
    // ADAPTIVE STALL (E4.D, 2026-09-10): refresh M when the frozen iteration's REMAINING steps, predicted from
    // its observed contraction ρ = |dx_k|/|dx_{k-1}| as log(step_tol/|dx_k|)/log(ρ), would cost more than a
    // refresh. The break-even step count is MEASURED per calibrator at construction (one Jacobian +
    // factorisation vs one residual evaluation), so a 12-knot OIS curve (~50 steps per refresh) never refreshes
    // on a 25 bp move while a daily-averaged Fed-funds leg (~4) refreshes after a few slow steps. A fixed
    // count (the old max_frozen = 4) refreshed a converging OIS tick for nothing and let the desk's slow frozen
    // rate run; max_frozen is now only the hard cap. false = count-based stalls only.
    bool adaptive_stall = true;
    // > 0: the adaptive stall's refresh-vs-step break-even, FIXED (steps) instead of measured at construction. The
    // measurement is a wall-clock ratio, so the refresh schedule -- not the converged answer -- varies with machine load;
    // tests and callers that need run-to-run identical schedules pin it here. 0 = measured.
    double breakeven_steps = 0.0;
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
    // tail-latency win; correctness is identical (frozen-Newton is exact for any invertible M). Armed
    // ONLY for a square, unbanded, unregularised problem (the worker hands over M alone; see the ctor).
    bool prefetch = false;
    double prefetch_drift = 5e-4;  // request a background M once drift from the anchor exceeds this (5bp)
    // SMOOTHNESS REGULARISER (empty = off). R = λ·D, the second-difference operator over the chosen
    // curves' knots (swaps::calibration::second_difference_operator). When set, the frozen-Newton
    // operator becomes M = (JᵀJ + RᵀR)⁻¹Jᵀ and each step gains a B·x curvature-pull term, so a
    // RANK-DEFICIENT (basis-only) build streams directly -- selecting the smoothest curve consistent
    // with the market every tick, instead of a singular M. Empty keeps the exact previous behaviour.
    Eigen::MatrixXd regularizer;
  };

  using Engine = residual_engine_t<Problem>;

  // OWN engine (compiled from `prob`): tests and stand-alone callers.
  StreamingCalibrator(const Problem& prob, const Eigen::VectorXd& x0,
                      const Eigen::VectorXd& q0, const Options& opt)
      : StreamingCalibrator(nullptr, prob, x0, q0, opt) {}
  // SHARED engine (the object model, 2026-09-10): the calibrator BORROWS the caller's compiled engine --
  // one engine per compiled model, so a scalar set_quotes on it is seen by every consumer (calibrate, this
  // streamer, the risk operator). `engine` must outlive the calibrator; `prob` supplies the band table and
  // the background worker's own copy (a worker computes J on another thread, so it keeps private scratch).
  StreamingCalibrator(const Engine& engine, const Problem& prob, const Eigen::VectorXd& x0,
                      const Eigen::VectorXd& q0, const Options& opt)
      : StreamingCalibrator(&engine, prob, x0, q0, opt) {}

  // Re-anchor after the QUOTE RHS changed underneath the (shared) engine: bands re-read from `prob` (a band
  // edit changes which rows the active set tracks), the Jacobian re-taken at (x, q), the committed state set
  // to (x, q). An owned engine is pushed the new quotes first. Throws on a non-finite input/Jacobian.
  void resync(const Problem& prob, const Eigen::VectorXd& x, const Eigen::VectorXd& q) {
    if (!x.allFinite() || !q.allFinite()) throw std::invalid_argument("StreamingCalibrator::resync: non-finite state or market");
    if (owned_engine_) {
      if constexpr (requires(Engine& e, const Problem& p) { e.set_quotes(p); }) owned_engine_->set_quotes(prob);
    }
    collect_bands(prob);
    drift_refresh_ = (n_res_ != static_cast<int>(x.size())) || !bands_.empty() || RtR_.size() > 0;
    if (!set_anchor(x, q)) throw std::invalid_argument("StreamingCalibrator::resync: the Jacobian at the anchor is non-finite");
    x_cur_ = x;
    q_cur_ = q;
  }

 private:
  StreamingCalibrator(const Engine* shared, const Problem& prob, const Eigen::VectorXd& x0,
                      const Eigen::VectorXd& q0, const Options& opt)
      : n_res_(prob.n_residuals()), opt_(opt) {
    if (shared) {
      engine_ = shared;
    } else {
      owned_engine_ = std::make_unique<Engine>(prob);
      engine_ = owned_engine_.get();
    }
    if (!x0.allFinite()) throw std::invalid_argument("StreamingCalibrator: the anchor state x0 contains a non-finite value");
    if (!q0.allFinite()) throw std::invalid_argument("StreamingCalibrator: the anchor market q0 contains a non-finite quote");
    if (opt_.regularizer.size()) {
      RtR_.noalias() = opt_.regularizer.transpose() * opt_.regularizer;
      S_.resize(n_res_ + opt_.regularizer.rows(), x0.size());  // the stacked [J; R]: R rows written ONCE (factor writes J's)
      S_.bottomRows(opt_.regularizer.rows()) = opt_.regularizer;
    }
    cod_ = Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>(n_res_ + opt_.regularizer.rows(), x0.size());  // sized once
    collect_bands(prob);
    // The drift-triggered accuracy refresh applies only where the fixed point is not r = 0.
    drift_refresh_ = (n_res_ != static_cast<int>(x0.size())) || !bands_.empty() || RtR_.size() > 0;
    // The background worker hands over M ONLY (no B, no band re-scaling, and its private engine copy never
    // sees a later set_quotes), which is exact for a SQUARE, unbanded, unregularised problem and wrong for
    // the rest (E3 register S3 / G5): a regularised or banded stream would mix an un-regularised M with a
    // stale B and converge to the wrong fixed point. So the prefetch is armed only where it is exact;
    // elsewhere the option is ignored (prefetch_hits() stays 0).
    if (opt_.prefetch && !drift_refresh_) bg_ = std::make_unique<BackgroundJacobian<Problem>>(prob);
    if (!set_anchor(x0, q0))
      throw std::invalid_argument("StreamingCalibrator: the Jacobian at the anchor state is non-finite");
    {
      // The refresh-vs-step break-even for the adaptive stall: ONE (warm) Jacobian + factorisation against ONE
      // residual evaluation at a NEW state (the engines memoise DF on x, so the anchor state would time a cache
      // hit). A construction-time cost only; nothing on the tick.
      // MIN of a few repetitions on each side: single-shot timings of a 100 us residual varied 3x (caches).
      double step_ns = 1e300, refresh_ns = 1e300;
      for (int rep = 0; rep < 4; ++rep) {
        const Eigen::VectorXd xp = x0.array() + 1e-9 * (rep + 1);  // a NEW state each time (no memo hit)
        const auto t0 = std::chrono::steady_clock::now();
        r_ = engine_->residuals_vs(xp, q0);
        const auto t1 = std::chrono::steady_clock::now();
        step_ns = std::min(step_ns, std::chrono::duration<double, std::nano>(t1 - t0).count());
      }
      for (int rep = 0; rep < 2; ++rep) {
        const auto t1 = std::chrono::steady_clock::now();
        engine_->jacobian_vs_into(x0, q0, J_cur_);
        factor(J_cur_);
        const auto t2 = std::chrono::steady_clock::now();
        refresh_ns = std::min(refresh_ns, std::chrono::duration<double, std::nano>(t2 - t1).count());
      }
      breakeven_steps_ = std::min(64.0, std::max(2.0, refresh_ns / std::max(1.0, step_ns)));
      if (opt_.breakeven_steps > 0.0) breakeven_steps_ = opt_.breakeven_steps;  // pinned (Options::breakeven_steps)
    }
    x_cur_ = x0;
    q_cur_ = q0;
    last_step_.resize(x0.size());  // sized once: the corrector's 2-cycle check allocates nothing
  }

  // The banded rows (FX forwards are never banded): what the per-tick side tracking watches. decay > 0
  // only: the inside branch must be invertible (q from r) and the frozen row non-zero. A decay-0 (dead-
  // zone) band is not tracked; it still works through the stall-triggered refresh.
  void collect_bands(const Problem& prob) {
    bands_.clear();
    quote_lo_.assign(n_res_, 0.0);
    quote_up_.assign(n_res_, 0.0);
    band_eligible_.assign(n_res_, 0);
    for (int i = 0; i < n_res_; ++i) {
      const Instrument& ins = prob.instruments[i];
      quote_lo_[i] = ins.band_lower;
      quote_up_[i] = ins.band_upper;
      band_eligible_[i] = ins.quote != QuoteKind::FxForward;
    }
    for (int i = 0; i < n_res_; ++i) {
      const Instrument& ins = prob.instruments[i];
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
    n_pinned_ = 0;
  }

 public:
  // FOUR-NUMBER REQUOTE (K5', 2026-09-14): move every row's band in place for the next update(). The engine this streamer
  // prices on must already carry the new bands (BundleSession writes both). A tracked row whose band only MOVES, or changes a
  // non-zero decay, keeps its frozen Jacobian row: the next tick's side reconciliation (track_bands) re-scales the row if its
  // slope changed -- the rank-one operator update, no Jacobian. A pinned row whose band moved is released first (its edge is no
  // longer where it was pinned). Returns false and changes NOTHING when the set of tracked rows would change (a band appears or
  // disappears, or a decay crosses 0): the caller re-anchors (resync). Inputs are the caller's validated quotes.
  bool set_bands(const Eigen::VectorXd& lower, const Eigen::VectorXd& upper, const Eigen::VectorXd& decay) {
    if (lower.size() != n_res_ || upper.size() != n_res_ || decay.size() != n_res_)
      throw std::invalid_argument("StreamingCalibrator::set_bands: lengths must equal the instrument count");
    std::size_t k = 0;
    for (int i = 0; i < n_res_; ++i) {
      const bool banded = upper[i] > lower[i];
      const bool tracked = banded && band_eligible_[i] && decay[i] > 0.0;
      const bool was_tracked = k < bands_.size() && bands_[k].row == i;
      if (tracked != was_tracked || banded != (quote_up_[i] > quote_lo_[i])) return false;
      if (was_tracked) ++k;
    }
    k = 0;
    for (int i = 0; i < n_res_; ++i) {
      quote_lo_[i] = lower[i];
      quote_up_[i] = upper[i];
      if (k < bands_.size() && bands_[k].row == i) {
        BandRow& b = bands_[k];
        if (b.lower != lower[i] || b.upper != upper[i] || b.decay != decay[i]) {
          b.lower = lower[i];
          b.upper = upper[i];
          b.decay = decay[i];
          if (state_[k] != 0) {  // pinned on an edge that has moved: release (track_bands re-scales it off the pin weight)
            state_[k] = 0;
            --n_pinned_;
          }
          onedge_[k] = 0;
        }
        ++k;
      }
    }
    return true;
  }

  const Eigen::VectorXd& current() const { return x_cur_; }
  const Eigen::VectorXd& anchor_market() const { return q_anchor_; }
  const Eigen::MatrixXd& sensitivity() const { return M_; }
  int refresh_count() const { return refresh_count_; }
  int prefetch_hits() const { return prefetch_hits_; }  // refreshes served from the background worker
  int rescale_count() const { return rescale_count_; }  // band-edge re-scales (cheap M refactors) so far
  // GUARD COUNTERS (2026-09-15, G1): cumulative and allocation-free; the guard tests read deltas. factor_count is every factorisation
  // of M (construction, an anchor or refresh, and rescale_row's degenerate-update fallback); pin_count / release_count are the band
  // walk's kink pins and their KKT releases.
  int factor_count() const { return factor_count_; }
  int pin_count() const { return pin_count_; }
  int release_count() const { return release_count_; }
  double breakeven_steps() const { return breakeven_steps_; }  // the adaptive stall's refresh-vs-step ratio

  StreamTick update(const Eigen::VectorXd& q_new) {
    if (q_new.size() != n_res_) throw std::invalid_argument("StreamingCalibrator::update: market length does not match the instrument count");
    if (!q_new.allFinite()) throw std::invalid_argument("StreamingCalibrator::update: market contains a non-finite quote");
    // K5': a banded row's target must lie inside its band (validate_quote) -- refused before the tick touches any state.
    for (int i = 0; i < n_res_; ++i)
      if (quote_up_[i] > quote_lo_[i] && (q_new[i] < quote_lo_[i] || q_new[i] > quote_up_[i]))
        validate_quote(q_new[i], quote_lo_[i], quote_up_[i], 1.0, "StreamingCalibrator::update", i);
    return update_exact(q_new);
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
    tick_stages_ = 0;
    x_ = x_cur_;              // warm start from the last exact solution (tick-to-tick move is tiny)
    Eigen::VectorXd& x = x_;  // reused scratch: the frozen-Newton loop below allocates nothing
    if (drift_refresh_ && t.drift > opt_.refresh_drift) {  // accuracy refresh (see Options::refresh_drift)
      stamp(StreamStage::DriftRefresh);
      if (!refresh(x, q_new, t)) return fail(t, StreamStatus::NonFinite);
    }
    int frozen = 0;
    const bool drift_refreshed = t.refreshes > 0;
    bool final_refresh_done = false;
    for (std::size_t k = 0; k < flips_.size(); ++k) {
      flips_[k] = 0;
      releases_[k] = 0;  // the per-tick release budget (verify_pins) starts fresh every tick
    }
    double r0_inf = -1.0;  // the tick's first residual size: the divergence yardstick
    double dx_prev = -1.0;  // |dx| of the previous FULL step under the same operator (-1: none)
    have_last_step_ = false;  // kink 2-cycle detection: no previous full step yet this tick
    full_rank_ = false;       // two-threshold operator: walk at the walk threshold, commit at the shared one
    in_walk_ = true;
    struct WalkEnd { bool& w; ~WalkEnd() { w = false; } } walk_end{in_walk_};
    for (;;) {
      // The residual is engine-defined against the live market q_new: model_rates - q_new for hard
      // instruments, the Huber band residual for soft (banded) ones. Driving THIS (not the raw reprice)
      // is what makes frozen-Newton solve the soft least-squares -- dx = J⁺·r -> 0 at the soft minimum.
      r_ = engine_->residuals_vs(x, q_new);
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
            stamp(StreamStage::TrackRefresh);
            if (!set_anchor(x, q_new)) return fail(t, StreamStatus::NonFinite);
            t.refreshed = true;
            ++t.refreshes;
          } else {
            stamp(StreamStage::TrackRescale);
            ++t.rescales;  // the operator was updated row by row inside track_bands (rescale_row)
            ++rescale_count_;
          }
          frozen = 0;
          dx_prev = -1.0;  // a new operator: the contraction history no longer applies
          have_last_step_ = false;  // an active-set change legitimately turns the step
        }
      }
      dx_.noalias() = M_ * r_;
      if (RtR_.size()) dx_.noalias() += B_ * x;  // curvature pull toward the smoothest market-consistent curve
      double alpha = 1.0;
      std::size_t hit = 0;
      int hit_side = 0;
      if (!bands_.empty() && have_J_) alpha = breakpoint(q_new, &hit, &hit_side);
      // KINK 2-CYCLE (FLK2, 2026-09-14). On a piecewise-smooth residual (MonotoneCubic's Hyman filter) a Newton step can
      // jump a kink and the step from the far side jumps straight back: dx_{k+1} = -dx_k, |r| unchanged, and a refresh at
      // either point reproduces it until max_refresh fails the tick (desk_mixed: knot 11 alternating 0.0341513 / 0.0342819
      // at |dx| 1.306e-4). A full step that REVERSES the previous full step (anti-parallel, similar size) is halved: the
      // midpoint lands on the kink and the iteration converges. A contracting or merely turning iteration never matches.
      double damp = 1.0;
      if (alpha == 1.0) {
        if (have_last_step_) {
          const double n2 = dx_.squaredNorm();
          if (dx_.dot(last_step_) < -0.9 * n2 && last_step_.squaredNorm() < 1.21 * n2) {
            damp = 0.5;
            stamp(StreamStage::KinkDamp);
            SWAPS_TRACE("  kink 2-cycle: half step\n");
          }
        }
        last_step_ = dx_;  // same size after the first tick: no allocation on the hot path
        have_last_step_ = true;
      } else {
        have_last_step_ = false;  // a breakpoint step: the operator is about to change
      }
      x.noalias() -= (alpha * damp) * dx_;
      ++t.newton_steps;
      if (!x.allFinite()) return fail(t, StreamStatus::NonFinite);
      SWAPS_TRACE("  step %d |dx|=%.2e alpha=%.3f pinned=%d rescales=%d refreshes=%d\n", t.newton_steps, dx_.cwiseAbs().maxCoeff(), alpha, n_pinned_, t.rescales, t.refreshes);
      if (alpha < 1.0) {  // stopped on a band edge: switch that row there and carry on
        stamp(StreamStage::Breakpoint);
        switch_row(hit, hit_side);  // (updates the operator in place, rescale_row)
        ++t.rescales;
        ++rescale_count_;
        frozen = 0;
        dx_prev = -1.0;
        continue;
      }
      const double dx_inf = dx_.cwiseAbs().maxCoeff();
      // Predictive convergence (Options::predict_convergence): two full steps under one operator, contracting,
      // and the contraction-bounded next step already far below tolerance -- stop here.
      bool predicted = false;
      if (opt_.predict_convergence && dx_prev > 0.0 && dx_inf < dx_prev && dx_inf >= opt_.step_tol) {
        const double ratio = dx_inf / dx_prev;
        predicted = ratio < 0.5 && ratio * dx_inf < 0.05 * opt_.step_tol;  // 2x margin on the bound
        if (predicted) stamp(StreamStage::Predicted);
      }
      // Adaptive stall: with two full steps under one operator, would the steps still to come cost more than a
      // refresh? (ρ ≥ 1: the frozen iteration is not contracting at all.)
      bool slow = false;
      if (opt_.adaptive_stall && dx_prev > 0.0 && dx_inf >= opt_.step_tol && !predicted) {
        const double rho = dx_inf / dx_prev;
        // The observed ratio is the OPTIMISTIC one (the first frozen steps contract fastest; the stale-M error
        // then dominates and the rate halves, measured), so the remaining-steps estimate carries a factor 2.
        slow = rho >= 1.0 || 2.0 * std::log(opt_.step_tol / dx_inf) / std::log(rho) > breakeven_steps_;
        SWAPS_TRACE("  adaptive: |dx| %.2e prev %.2e rho %.3f steps_left %.1f K %.1f -> %s\n", dx_inf, dx_prev, rho,
                    rho < 1.0 ? 2.0 * std::log(opt_.step_tol / dx_inf) / std::log(rho) : -1.0, breakeven_steps_, slow ? "REFRESH" : "continue");
      }
      dx_prev = dx_inf;
      if (dx_inf < opt_.step_tol || predicted) {  // converged for the CURRENT active set
        if (!bands_.empty() && n_pinned_ > 0 && verify_pins(x, q_new)) {
          ++t.rescales;  // a pin was released onto its true side (rescale_row updated the operator): iterate on
          ++rescale_count_;
          frozen = 0;
          dx_prev = -1.0;
          continue;
        }
        // A drift refresh anchored J at the PREVIOUS solution; the frozen-J answer is then exact only to
        // second order in the move. One more anchor at the converged point and a re-converge (a genuine
        // Newton step with the fresh J) removes that term -- two Jacobians per refresh_drift of drift.
        // Not with the background prefetch on: there tail latency is the contract, and the second inline
        // Jacobian would be exactly the spike the worker exists to hide.
        if (truncated_ && !full_rank_) {
          // TWO-THRESHOLD COMMIT RULE: this point is a fixed point of a TRUNCATED operator -- exact only in the directions it
          // kept. Re-anchor here at the shared threshold and re-converge (the rest of the tick stays at full rank).
          full_rank_ = true;
          stamp(StreamStage::CommitReanchor);
          // ...and it IS this tick's final refresh: a Jacobian at a converged point of this market, which is all the final refresh
          // below guarantees. Taking both re-anchored twice one re-convergence apart (S3, 2026-09-15).
          final_refresh_done = true;
          if (!set_anchor(x, q_new)) return fail(t, StreamStatus::NonFinite);
          t.refreshed = true;
          ++t.refreshes;
          frozen = 0;
          dx_prev = -1.0;
          have_last_step_ = false;
          continue;
        }
        if (drift_refreshed && !final_refresh_done && !bg_ && t.refreshes < opt_.max_refresh) {
          final_refresh_done = true;
          stamp(StreamStage::FinalRefresh);
          if (!set_anchor(x, q_new)) return fail(t, StreamStatus::NonFinite);
          ++t.refreshes;
          frozen = 0;
          dx_prev = -1.0;
          continue;
        }
        t.converged = true;
        t.status = StreamStatus::Converged;
        break;
      }
      if (t.newton_steps >= opt_.max_steps) return fail(t, StreamStatus::StepCap);
      if (++frozen >= opt_.max_frozen || slow) {  // M stale as a preconditioner -> refresh
        stamp(slow ? StreamStage::StallRefresh : StreamStage::FrozenCapRefresh);
        if (t.refreshes >= opt_.max_refresh) return fail(t, StreamStatus::RefreshCap);
        if (!refresh(x, q_new, t)) return fail(t, StreamStatus::NonFinite);
        frozen = 0;
        dx_prev = -1.0;
      }
    }
    x_cur_ = x;  // committed: the exact solution at q_new
    q_cur_ = q_new;
    // Speculatively pre-compute the NEXT M once the market has drifted enough that a refresh is plausibly
    // near -- so the worker is done (on a spare core) before the envelope is hit. Coalesces: a request
    // while one is in flight just updates the target x.
    if (bg_ && !bg_->computing() && t.drift > opt_.prefetch_drift) bg_->request(x_cur_);
    t.stages = tick_stages_;
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
    stamp(StreamStage::Fail);
    SWAPS_TRACE("  tick FAILED: %s (steps %d refreshes %d rescales %d)\n", to_string(why), t.newton_steps, t.refreshes, t.rescales);
    if (t.refreshes > 0 || t.rescales > 0) stamp(StreamStage::FailRestore);
    if (t.refreshes > 0 || t.rescales > 0) (void)set_anchor(x_cur_, q_cur_);
    t.stages = tick_stages_;
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
      ++pin_count_;
      stamp(StreamStage::Pin);
      s_pin_[k] = 0.0;  // the multiplier slope is known only at convergence (verify_pins)
      rescale_row(k, kPinWeight);
      return;
    }
    const double slope = side == 0 ? b.decay : 1.0;
    rescale_row(k, slope);
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
      rescale_row(k, slope);
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
    r_ = engine_->residuals_vs(x, q);  // the engine's (Huber) residuals: pinned rows' q_model read off these
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
        ++release_count_;
        stamp(StreamStage::Release);
        ++releases_[k];  // one release per row per tick; a re-pin after it is final (termination)
        rescale_row(k, slope);
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
      stamp(StreamStage::PrefetchTake);
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
    bool sides_from_j = false;  // S2: the band sides below come from the Jacobian pass's residuals, when the engine returns them
    if constexpr (requires(const Engine& e, const Eigen::VectorXd& v, Eigen::MatrixXd& m, Eigen::VectorXd* p) { e.jacobian_vs_into(v, v, m, p); }) {
      sides_from_j = !bands_.empty() && opt_.anchor_sides_from_jacobian;
      if (sides_from_j) engine_->jacobian_vs_into(x, q, J_ref_, &r_anchor_);
    }
    if (!sides_from_j) engine_->jacobian_vs_into(x, q, J_ref_);  // in place (C6); band term consistent with residuals_vs(·,q)
    if (!J_ref_.allFinite()) {
      if (have_J_) J_ref_ = J_cur_;  // keep the previous anchor usable (J_cur_ is its re-scaled copy)
      return false;
    }
    x_anchor_ = x;
    q_anchor_ = q;
    if (!bands_.empty()) {
      // Each tracked row's side at the anchor. From the Jacobian pass's residuals (S2): the Huber residual is monotone in the model quote,
      // so r above the upper edge's residual decay·(upper − m) is above the band and below the lower edge's is below -- the tick's own
      // side_of test, read at (x, q). The reference (anchor_sides_from_jacobian = false) re-prices every model quote.
      const Eigen::VectorXd* mr = sides_from_j ? nullptr : &engine_->model_rates(x);
      for (std::size_t k = 0; k < bands_.size(); ++k) {
        const BandRow& b = bands_[k];
        int side = 0;
        if (mr) {
          side = ((*mr)[b.row] > b.upper) ? +1 : ((*mr)[b.row] < b.lower ? -1 : 0);
        } else {
          const double m = q[b.row], r = r_anchor_[b.row];
          side = r > b.decay * (b.upper - m) ? +1 : (r < b.decay * (b.lower - m) ? -1 : 0);
        }
        slope_ref_[k] = side == 0 ? b.decay : 1.0;  // band_slope; > 0: tracked rows have decay > 0
        slope_cur_[k] = slope_ref_[k];
        last_side_[k] = side < 0 ? -1 : +1;
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
  // TWO-THRESHOLD OPERATOR (2026-09-15). A direction below the walk threshold -- kWalkRankThreshold x sigma_max, measured against the
  // UNPINNED scale (a pinned row carries kPinWeight and would set sigma_max) -- is WEAK. An anchor factorisation (construction, resync, an
  // external set_anchor, the commit re-anchor) records how many weak directions the committed state has: those are STRUCTURAL (a
  // near-collinear pair of knots, an xccy bundle's weakly identified basis knots) and the walk keeps them. A factorisation met while
  // walking with MORE weak directions than the anchor is transiently near-singular (desk_mixed's MonotoneCubic long end mid-walk): its
  // operator drops them, and the commit rule (update_exact) re-converges at the shared threshold before the tick commits.
  double walk_threshold() const { return n_pinned_ > 0 ? kWalkRankThreshold / kPinWeight : kWalkRankThreshold; }
  void factor(const Eigen::MatrixXd& J) {
    ++factor_count_;
    // ONE decomposition for the calibrator's life (C6, 2026-09-15): sized (rows, cols) at construction, so compute() on a same-shaped
    // matrix writes into storage it already owns -- the same Eigen operations on the same data as a fresh local, bit-identical.
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>& cod = cod_;
    if (RtR_.size()) S_.topRows(J.rows()) = J;
    const Eigen::MatrixXd& A = RtR_.size() ? S_ : J;
    cod.setThreshold(kRankThreshold);
    cod.compute(A);
    const Eigen::Index full = cod.rank();
    cod.setThreshold(walk_threshold());
    const int weak = static_cast<int>(full - cod.rank());
    cod.setThreshold(kRankThreshold);
    truncated_ = false;
    if (!in_walk_ || full_rank_) {
      anchor_weak_ = weak;
    } else if (weak > anchor_weak_) {
      truncated_ = true;
      stamp(StreamStage::Truncated);
      cod.setThreshold(walk_threshold());
      cod.compute(A);
    }
    if (RtR_.size()) {
      const Eigen::MatrixXd P = cod.pseudoInverse();  // n_knots x (n_res + n_reg)
      M_ = P.leftCols(J.rows());
      G_.noalias() = P * P.transpose();  // (JᵀJ + RᵀR)⁺ -- kept for the O(n·m) band re-scale updates
      B_.noalias() = G_ * RtR_;
    } else {
      M_ = cod.solve(Eigen::MatrixXd::Identity(n_res_, n_res_));
      G_.noalias() = M_ * M_.transpose();  // (JᵀJ)⁺ = J⁺ J⁺ᵀ
    }
  }

  // A band re-scale (E3-C7, 2026-09-10): row `row` of the frozen Jacobian changes slope by c = new/old, a
  // RANK-ONE change of JᵀJ (β·u uᵀ, β = c² − 1, u = the old row). The operator is updated exactly by
  // Sherman–Morrison on G = (JᵀJ + RᵀR)⁺ (u lies in G's range, so the pseudo-inverse form holds):
  //     v = G u,  d = 1 + β uᵀv,   G' = G − (β/d) v vᵀ,
  //     M' = G' J'ᵀ = M − (β/d) v (J v)ᵀ + ((c−1)/d) v e_rowᵀ,   B' = G' RᵀR = B − (β/d) v (RᵀR v)ᵀ,
  // O(n·m) and allocation-free after first use, where a re-factorisation is O(n³) (300 µs on the desk rung
  // and 1.75 MB of temporaries, 12–16 times per 25 bp tick). Rounding in G moves no fixed point (any
  // non-singular preconditioner leaves the stationarity condition alone); the pins' 1e3 weight gives d ~ 1e6.
  void rescale_row(std::size_t k, double new_slope) {
    const int row = bands_[k].row;
    const double c = new_slope / slope_cur_[k];
    if (have_J_ && c != 1.0 && opt_.rescale_update) {
      u_ = J_cur_.row(row).transpose();
      v_.noalias() = G_ * u_;
      const double s = u_.dot(v_), beta = c * c - 1.0, d = 1.0 + beta * s;
      // Releasing a PIN (c = decay / 1e3) on a row whose leverage s is close to 1 cancels d = 1 + βs
      // catastrophically; such a row is rare (one release per row per tick) -- re-factorise instead.
      if (std::isfinite(d) && std::abs(d) > 1e-3) {
        jv_.noalias() = J_cur_ * v_;  // the OLD J
        const double f = beta / d;
        M_.noalias() -= f * v_ * jv_.transpose();
        M_.col(row) += ((c - 1.0) / d) * v_;
        if (RtR_.size()) {
          rv_.noalias() = RtR_ * v_;
          B_.noalias() -= f * v_ * rv_.transpose();
        }
        G_.noalias() -= f * v_ * v_.transpose();
        J_cur_.row(row) = J_ref_.row(row) * (new_slope / slope_ref_[k]);
        slope_cur_[k] = new_slope;
        return;
      }
    }
    J_cur_.row(row) = J_ref_.row(row) * (new_slope / slope_ref_[k]);
    slope_cur_[k] = new_slope;
    if (have_J_) stamp(StreamStage::RescaleFallback);
    if (have_J_) factor(J_cur_);  // degenerate update: fall back to a full factorisation
  }

  int n_res_;
  const Engine* engine_ = nullptr;          // the residual engine driven every tick (borrowed or owned)
  std::unique_ptr<Engine> owned_engine_;    // set only by the (prob, ...) constructor
  Options opt_;
  Eigen::VectorXd x_anchor_, q_anchor_, x_cur_;
  Eigen::VectorXd q_cur_;  // the market x_cur_ solves (a failed tick restores the anchor here)
  Eigen::MatrixXd M_;
  Eigen::MatrixXd RtR_, B_;  // smoothness regulariser: RᵀR and the per-step curvature pull B=(JᵀJ+RᵀR)⁻¹RᵀR
  Eigen::MatrixXd G_;        // (JᵀJ + RᵀR)⁺ from the last factor(), updated rank-one per band re-scale
  Eigen::VectorXd u_, v_, jv_, rv_;  // rescale_row scratch (no per-rescale allocation after first use)
  int refresh_count_ = 0;
  int prefetch_hits_ = 0;
  int rescale_count_ = 0;
  int factor_count_ = 0, pin_count_ = 0, release_count_ = 0;  // guard counters (factor_count() ...)
  unsigned tick_stages_ = 0;  // StreamStage bits of the tick in progress (copied into StreamTick::stages on return)
  void stamp(StreamStage s) { tick_stages_ |= 1u << static_cast<unsigned>(s); }
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
  std::vector<double> quote_lo_, quote_up_;  // per row: the band every tick's target must lie in (upper <= lower: none)
  std::vector<char> band_eligible_;          // per row: may carry a tracked band (FX forwards never do)
  int n_pinned_ = 0;
  Eigen::MatrixXd J_ref_, J_cur_;
  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod_;  // factor()'s decomposition, sized once at construction (C6)
  Eigen::MatrixXd S_;  // regularised only: the stacked [J; R] (R rows written at construction)
  bool have_J_ = false;
  bool need_full_ = false;
  bool drift_refresh_ = false;  // Options::refresh_drift applies (non-square / banded / regularised)
  double breakeven_steps_ = 8.0;  // adaptive stall: measured refresh cost in residual-evaluation units
  // Speculative background Jacobian (null unless opt_.prefetch): a dedicated thread computes the next M.
  std::unique_ptr<BackgroundJacobian<Problem>> bg_;
  Eigen::VectorXd bg_x_;  // scratch for a taken background result
  Eigen::MatrixXd bg_M_;
  // Per-tick scratch so the exact frozen-Newton loop allocates nothing (sized on first use).
  Eigen::VectorXd x_, r_, dx_;
  Eigen::VectorXd r_anchor_;  // set_anchor: the residuals at (x, q) the Jacobian pass returns, read for the band sides (S2)
  Eigen::VectorXd last_step_;     // the previous FULL step this tick (kink 2-cycle detection)
  bool have_last_step_ = false;
  // Two-threshold operator (see factor): the walk drops weak directions the anchor did not have; a tick that converged on a
  // truncated operator re-converges at the shared threshold before committing (update_exact).
  static constexpr double kWalkRankThreshold = 1e-4;
  bool full_rank_ = false;   // this tick has re-anchored at the shared threshold
  bool truncated_ = false;   // the current operator dropped a weak direction
  bool in_walk_ = false;     // inside update_exact (a factorisation here is not an anchor)
  int anchor_weak_ = 0;      // weak directions at the last anchor factorisation
};

// ---- K5' quote hand-off helpers (2026-09-15; PRINCIPLES.md P14: the session verbs call these, the behaviour lives here) -----------

// Every banded row's TARGET against its band -- refused before anything changes. The bands themselves were validated when they were
// set (validate_instrument / validate_quote), so only a target that left its band is re-checked (the cheap test first).
inline void validate_targets(const std::vector<Instrument>& ins, const Eigen::VectorXd& target, const char* where) {
  for (int i = 0; i < target.size(); ++i) {
    const Instrument& in = ins[static_cast<std::size_t>(i)];
    if (in.band_upper > in.band_lower && (target[i] < in.band_lower || target[i] > in.band_upper))
      validate_quote(target[i], in.band_lower, in.band_upper, in.band_decay, where, i);
  }
}

// A FOUR-NUMBER requote, validated whole before any of it lands: every row's (target, lower, upper, decay).
inline void validate_quotes(const Eigen::VectorXd& target, const Eigen::VectorXd& lower, const Eigen::VectorXd& upper,
                            const Eigen::VectorXd& decay, int n, const char* where) {
  if (target.size() != n || lower.size() != n || upper.size() != n || decay.size() != n)
    throw std::runtime_error(std::string(where) + ": target / lower / upper / decay lengths must equal the instrument count");
  for (int i = 0; i < n; ++i) validate_quote(target[i], lower[i], upper[i], decay[i], where, i);
}

// P3 (2026-09-15): lands a VALIDATED target vector on the session's quotes -- the instruments AND the shared compiled engine -- so
// whatever reads the session's market after a tick (resolve, residual, quote_diagnostics, result().rms_residual) reads the market the
// curve was solved to. The tick itself prices the q it is handed; this is the session's record of it.
template <class Engine>
void commit_targets(std::vector<Instrument>& ins, Engine* engine, const Eigen::VectorXd& target) {
  for (std::size_t i = 0; i < ins.size(); ++i) ins[i].market = target[static_cast<Eigen::Index>(i)];
  if (engine) engine->set_market(target);
}

// Lands a VALIDATED requote's bands: on the instruments, on the compiled engine (set_quote, row by row, only where a band moved) and
// on the streamer IN PLACE (set_bands). `has_band` is refreshed when any band moved. Returns true when the streamer must re-anchor
// instead -- WHICH rows are banded changed (one Jacobian before the next tick).
template <class Engine, class Streamer>
bool requote_bands(std::vector<Instrument>& ins, Engine* engine, Streamer* stream, const Eigen::VectorXd& lower,
                   const Eigen::VectorXd& upper, const Eigen::VectorXd& decay, bool& has_band) {
  bool moved = false;
  for (std::size_t i = 0; i < ins.size(); ++i) {
    Instrument& in = ins[i];
    const Eigen::Index r = static_cast<Eigen::Index>(i);
    if (in.band_lower == lower[r] && in.band_upper == upper[r] && in.band_decay == decay[r]) continue;
    moved = true;
    in.band_lower = lower[r];
    in.band_upper = upper[r];
    in.band_decay = decay[r];
    if (engine) engine->set_quote(static_cast<int>(i), in.market, lower[r], upper[r], decay[r]);  // the tick passes the targets
  }
  if (!moved) return false;
  has_band = false;
  for (const Instrument& in : ins) has_band = has_band || in.band_upper > in.band_lower;
  return stream != nullptr && !stream->set_bands(lower, upper, decay);
}

}  // namespace swaps::calibration
