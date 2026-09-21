// MARKET WALKS for the shape ladder (2026-09-21): a seeded, deterministic tick-by-tick TARGET generator for a Shape, so
// a rung can be driven along a long NON-REPEATING path instead of the two-point toggles (q0 <-> q_small / q_big) the
// perf benches use. Those toggles are right for a timing metric -- a repeatable tick is a measurable tick -- but they
// revisit the same two markets, so a scheme whose Jacobian jumps between value-dependent branch cells (MonotoneCubic's
// Hyman filter) is never asked to carry a frozen operator across a cell it has not seen. The first random walk over
// desk_mixed (scratchpad soak, 2026-09-21) failed within a few hundred ticks; nothing in the gate had ever driven it.
//
// Two walks, both starting AT q0 (so tick 0 is a null move) and both leaving a row the fixture holds fixed (an FX
// forward pinned by covered interest parity) where the MODEL puts it:
//
//   Factor  -- the realistic one. Per CURVE, three mean-reverting factors in KNOT space (level, slope, butterfly)
//              on top of x_true, a spread curve at 0.3x an outright's amplitude; the targets are the MODEL quotes at
//              that state (every row self-consistent, neighbouring tenors move together, FX forwards follow CIP by
//              construction) plus a fast-reverting per-row jitter of ~0.03 bp (bid/ask noise), and a +-8 bp level
//              shock on every curve every 997 ticks. This is the tools/stream_sim.cpp recipe on the ladder.
//   PerRow  -- the stress. Every rate row its own OU noise at 0.15 bp/tick scaled by that row's share of the fixture's
//              25 bp move, no cross-row correlation, a full-size per-row shock every 997 ticks. Neighbouring quotes
//              contradict each other every tick, which is exactly what walks a Hyman filter across its branch cells;
//              it finds the kinks fastest and overstates what a trading day does.
//
// A banded row's band travels with its target through Shape::requote (K5'), at the call site.
#pragma once

#include <Eigen/Core>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "shape_ladder.hpp"
#include "swaps/calibration/hybrid_residual.hpp"

namespace swaps::shapes {

enum class WalkKind { Factor, PerRow };
inline const char* to_string(WalkKind k) { return k == WalkKind::Factor ? "factor" : "per-row"; }

class MarketWalk {
 public:
  MarketWalk(const Shape& s, WalkKind kind, unsigned seed)
      : s_(s), kind_(kind), rng_(seed), scale_((s.q_big - s.q0).cwiseAbs()), q_(s.q0),
        a_(Eigen::VectorXd::Zero(s.q0.size())), jit_(Eigen::VectorXd::Zero(s.q0.size())) {
    if (kind_ == WalkKind::Factor) {
      eng_ = std::make_unique<calibration::HybridBundleResidual>(s.prob);
      x_ = s.x_true;
      const int nc = static_cast<int>(s.prob.curves.size());
      blocks_.resize(static_cast<std::size_t>(nc));
      for (int c = 0; c < nc; ++c) {
        Block& b = blocks_[static_cast<std::size_t>(c)];
        b.off = s.prob.offset(c);
        b.spread = s.prob.curves[static_cast<std::size_t>(c)].base >= 0;
        std::vector<double> t;
        for (const auto& reg : s.prob.curves[static_cast<std::size_t>(c)].regions) t.insert(t.end(), reg.knots.begin(), reg.knots.end());
        const int nk = static_cast<int>(t.size());
        b.level.resize(nk); b.slope.resize(nk); b.fly.resize(nk);
        const double t0 = nk ? t.front() : 0.0, t1 = nk ? t.back() : 0.0, span = t1 > t0 ? t1 - t0 : 1.0;
        for (int i = 0; i < nk; ++i) {
          const double u = t1 > t0 ? (t[static_cast<std::size_t>(i)] - t0) / span : 0.5;  // 0 at the first knot, 1 at the last
          b.level[i] = 1.0;
          b.slope[i] = 2.0 * u - 1.0;                 // -1 .. +1 across the curve
          b.fly[i] = 4.0 * u * (1.0 - u) - 2.0 / 3.0;  // belly up, wings down, ~zero mean
        }
      }
    }
  }

  // The next tick's targets, in residual order. Tick 0 returns q0 itself.
  const Eigen::VectorXd& next() {
    const long k = tick_++;
    if (k == 0) return q_;
    const bool shock = (k % 997) == 0;
    if (kind_ == WalkKind::PerRow) {
      for (int i = 0; i < q_.size(); ++i) {
        if (scale_[i] == 0.0) continue;  // a row the fixture holds fixed (FX forwards)
        a_[i] += -0.002 * a_[i] + 0.006 * scale_[i] * z_(rng_);  // 0.15 bp/tick per 25 bp of that row's move
        if (shock) a_[i] += (z_(rng_) > 0 ? 1.0 : -1.0) * scale_[i];
      }
      q_ = s_.q0 + a_;
      return q_;
    }
    // Factor: knot-space factors per curve, model quotes, jitter.
    x_ = s_.x_true;
    for (Block& b : blocks_) {
      const double amp = b.spread ? 0.3 : 1.0;
      b.aL += -kappa_ * b.aL + amp * 0.15e-4 * z_(rng_);
      b.aS += -kappa_ * b.aS + amp * 0.08e-4 * z_(rng_);
      b.aF += -kappa_ * b.aF + amp * 0.05e-4 * z_(rng_);
      if (shock) b.aL += (z_(rng_) > 0 ? 1.0 : -1.0) * amp * 8e-4;
      for (int i = 0; i < b.level.size(); ++i)
        x_[b.off + i] += b.aL * b.level[i] + b.aS * b.slope[i] + b.aF * b.fly[i];
    }
    q_ = eng_->model_rates(x_);
    for (int i = 0; i < q_.size(); ++i) {
      if (scale_[i] == 0.0) continue;  // FX forwards: exactly where CIP puts them
      jit_[i] += -0.2 * jit_[i] + 0.03e-4 * z_(rng_);
      q_[i] += jit_[i];
    }
    return q_;
  }

  long tick() const { return tick_; }
  WalkKind kind() const { return kind_; }
  // Factor walk only: the knot state the last targets were generated at (before jitter).
  const Eigen::VectorXd& generating_state() const { return x_; }

 private:
  struct Block {
    int off = 0;
    bool spread = false;
    Eigen::VectorXd level, slope, fly;
    double aL = 0, aS = 0, aF = 0;
  };
  const Shape& s_;
  WalkKind kind_;
  std::mt19937 rng_;
  std::normal_distribution<double> z_{0.0, 1.0};
  const double kappa_ = 0.001;  // stationary sd ~ sigma / sqrt(2 kappa): a level band of ~3.4 bp around x_true
  Eigen::VectorXd scale_, q_, a_, jit_, x_;
  std::vector<Block> blocks_;
  std::unique_ptr<calibration::HybridBundleResidual> eng_;
  long tick_ = 0;
};

}  // namespace swaps::shapes
