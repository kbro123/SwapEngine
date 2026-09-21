// E5 taxonomy: T5 properties + value pins (failed-tick counts per ladder rung under seeded market walks; may only decrease)
// STREAMING SOAK GATE (2026-09-21). Every rung of the shape ladder is streamed through the SHIPPED seam (api::BundleSession)
// along a long, seeded, non-repeating market path -- the FACTOR walk (a realistic day: correlated level / slope / butterfly
// moves per curve, model-consistent quotes, bid/ask jitter, shocks) and the PER-ROW walk (the stress: every quote its own
// noise, neighbours contradicting each other) -- see bench/fixtures/market_walk.hpp.
//
// WHY THIS EXISTS. Until 2026-09-21 every driver of the ladder toggled between two fixed markets (q0 <-> q_small / q_big,
// 10-26 ticks), which is right for a timing metric and useless for robustness: a toggle revisits markets it has already
// solved. The first random walk over desk_mixed (the SOFR long end on MonotoneCubic) fell back to a full LM on ~1 tick in
// 5 at ~9 ms per tick against a 36 us healthy tick; nothing in the gate had ever asked.
//
// WHAT IT FOUND (the diagnosis, so nobody re-derives it). The market quotes at a 6-pillar long end (10y..30y par swaps) are
// INTEGRALS of the forward, so a forward ZIGZAG between pillars is nearly invisible to them: 0.03 bp of bid/ask jitter on
// the quotes was amplified into a +-40 bp second difference of the knot forwards on desk (Hermite) and +-80 bp on
// desk_mixed -- a cold LM from scratch delivers the same zigzag, so it is the market, not the streamer's path. On a
// Hermite long end that is merely an ugly curve at 34 us. On a Hyman-filtered long end the zigzag flips the secants'
// signs, which parks the curve ON the filter's branch boundaries (17-37 W re-takes within one tick), where the residual is
// kinked and every solver crawls (a damped LM needed 33-53 iterations for a 0.4 bp move; a fresh Jacobian at the same
// point diverged like the frozen one; a monotone-step backtracking experiment made it worse). The fix is the one
// CLAUDE.md Phase 3 named: the smoothness regulariser pins the invisible direction to the smoothest curve. It also found
// that the TENSION operator's shape functions are undefined on a value-dependent region (it evaluated the curve on unit
// knot vectors, which the Hyman filter clamps) -- at Smoothing::Strong it DROVE desk_mixed into a 149 bp zigzag; fixed in
// regularize.hpp (a value-dependent region contributes a discrete tension energy, ASSUMPTIONS.md D17).
//
// THE RECIPES, measured 400 ticks per rung (seed 20260921), mixed_scheme / desk_mixed failed ticks and desk_mixed mean us:
//   unsmoothed                       factor 49 / 85 @ 8,400 us     per-row 14 / 11 @ 4,500 us
//   Light TENSION (the default until 2026-09-21)  factor 82 / 22 @ 1,400 us   per-row 59 / 48 @ 5,000 us   (after the operator fix)
//   Strong tension                   factor  0 /  0 @    36 us     per-row  6 /  4 @ 1,600 us
//   Light SECOND-DIFFERENCE (the API default since 2026-09-21)  factor 0 / 0 @ 35 us   per-row 0 / 1 @ 812 us   zigzag <= 3 bp
//     (the divided-difference row, D18; the spacing-blind stencil it replaced read 0 / 0 @ 268 us on the per-row walk, at
//     the price of biasing every fit on non-uniform pillars)
// The tension presets are much weaker than the second-difference ones on 5-year knot spacings (the energy of a second
// difference D over spacing h is ~D^2/h^3, so the same weight buys ~1/125 of the discrete row's strength at h = 5).
//
// So the gate pins three things per rung:
//   SMOOTHED (the API's Light preset, i.e. second-difference over every curve -- the owner made it the default operator on
//     2026-09-21 on these numbers) -- the recipe that streams EVERY rung: ZERO failed ticks on the realistic walk. The invariant.
//   LIGHT TENSION (the opt-in operator, the default until 2026-09-21) -- its measured counts on the factor walk, so the
//     tension operator's cost on a value-dependent region is on record and may only DECREASE.
//   UNSMOOTHED -- the stress case, its measured counts, may only decrease.
// Plus, on every row: ACCURACY of the delivered curve in QUOTE space (CLAUDE.md 3b: never knot-space distance on a
// weakly-identified bundle) -- its model quotes against those of a cold solve of the SAME objective (targets, bands,
// regulariser) refined FROM it, every `check_every` ticks (null: a square rung's converged tick is exact, ~1e-12); and,
// informational, the largest second difference of the first curve's knot forwards (the zigzag amplitude).
//
// SWAPS_SOAK_TICKS (default 400) lengthens the walks for a local soak (pins are per-400-tick counts scaled to the run);
// SWAPS_SOAK_REG=second|tension and SWAPS_SOAK_LAMBDA override the smoothed rows' recipe for local experiments only.
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "market_walk.hpp"
#include "shape_ladder.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/calibration/regularize.hpp"

namespace api = swaps::api;
namespace cal = swaps::calibration;
using swaps::shapes::MarketWalk;
using swaps::shapes::Shape;
using swaps::shapes::WalkKind;

namespace {

const std::vector<Shape>& ladder() {
  static const std::vector<Shape> L = swaps::shapes::ladder();
  return L;
}

long soak_ticks() {
  if (const char* e = std::getenv("SWAPS_SOAK_TICKS")) return std::max(10L, std::atol(e));
  return 400;
}

enum class Recipe { Smoothed, LightTension, Unsmoothed };
const char* to_string(Recipe r) {
  switch (r) {
    case Recipe::Smoothed: return "SMOOTHED (the API's Light preset: second-difference over every curve)";
    case Recipe::LightTension: return "LIGHT TENSION (opt-in operator; the default until 2026-09-21)";
    case Recipe::Unsmoothed: return "UNSMOOTHED (the stress case)";
  }
  return "";
}

cal::RegSpec reg_for(Recipe r, const Shape& s) {
  const int nc = static_cast<int>(s.prob.curves.size());
  switch (r) {
    case Recipe::Unsmoothed: return cal::RegSpec{};
    case Recipe::LightTension: return cal::smoothing_preset(cal::Smoothing::Light, nc, /*tension=*/true);
    case Recipe::Smoothed: {
      cal::RegSpec reg = cal::smoothing_preset(cal::Smoothing::Light, nc);  // THE shipped default
      if (const char* e = std::getenv("SWAPS_SOAK_REG")) {  // local experiments only
        reg.tension = std::string(e) != "second";
        reg.lambda = cal::smoothing_lambda(cal::Smoothing::Light, reg.tension);
        reg.sigma = 0.0;
      }
      if (const char* e = std::getenv("SWAPS_SOAK_LAMBDA")) reg.lambda = std::atof(e);
      return reg;
    }
  }
  return cal::RegSpec{};
}

struct Outcome {
  long ticks = 0, failed = 0, maxrun = 0;
  double mean_us = 0, p99_us = 0, worst_dq = 0, worst_zig = 0;
  std::map<std::string, long> reasons;  // why each failed tick failed (last_reason)
};

// The largest second difference of the first curve's interpolation knots at x: the zigzag amplitude (rate units).
double zigzag(const Shape& s, const Eigen::VectorXd& x) {
  int nk = 0;
  for (const auto& reg : s.prob.curves[0].regions) nk += static_cast<int>(reg.knots.size());
  const int off = s.prob.offset(0);
  double z = 0;
  for (int i = 1; i + 1 < nk; ++i) z = std::max(z, std::abs(x[off + i + 1] - 2.0 * x[off + i] + x[off + i - 1]));
  return z;
}

Outcome soak(const Shape& s, WalkKind kind, const cal::RegSpec& reg, long n, unsigned seed, int check_every) {
  Outcome o;
  api::BundleSession sess(s.prob);
  sess.calibrate(s.x0, reg);
  sess.start_streaming(reg);
  MarketWalk walk(s, kind, seed);
  std::vector<double> us;
  us.reserve(static_cast<std::size_t>(n));
  long run = 0;
  double sum = 0;
  for (long k = 0; k < n; ++k) {
    const Eigen::VectorXd& q = walk.next();
    const Shape::Requote rq = s.requote(q);
    const auto t0 = std::chrono::steady_clock::now();
    const Eigen::VectorXd& x = s.has_bands ? sess.stream_update(rq.target, rq.lower, rq.upper, rq.decay) : sess.stream_update(rq.target);
    const double dt = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    sum += dt;
    us.push_back(dt);
    ++o.ticks;
    if (!sess.last_converged()) {
      ++o.failed;
      ++run;
      o.maxrun = std::max(o.maxrun, run);
      ++o.reasons[sess.last_reason()];
    } else {
      run = 0;
    }
    o.worst_zig = std::max(o.worst_zig, zigzag(s, x));
    if (k % check_every == check_every - 1) {  // the delivered curve vs a cold solve of the SAME objective refined from it
      cal::BundleProblem p = s.prob;
      for (int i = 0; i < p.n_residuals(); ++i) {
        auto& in = p.instruments[static_cast<std::size_t>(i)];
        in.market = rq.target[i]; in.band_lower = rq.lower[i]; in.band_upper = rq.upper[i]; in.band_decay = rq.decay[i];
      }
      api::BundleSession ref(p);
      const Eigen::VectorXd xr = ref.calibrate(x, reg).x;
      const Eigen::VectorXd rd = p.residuals<double>(x), rr = p.residuals<double>(xr);  // same market: the difference is in model quotes
      o.worst_dq = std::max(o.worst_dq, (rd - rr).cwiseAbs().maxCoeff());
    }
  }
  std::sort(us.begin(), us.end());
  o.mean_us = sum / static_cast<double>(n);
  o.p99_us = us[static_cast<std::size_t>(0.99 * static_cast<double>(n - 1))];
  return o;
}

// PINS: failed ticks per 400-tick walk, measured 2026-09-21 (seed 20260921) with the divided-difference operator (D18). The
// SMOOTHED FACTOR walk -- the realistic day under the shipped default -- is pinned at ZERO on every rung: that is the
// invariant. The SMOOTHED PER-ROW walk (contradictory neighbours every tick) is zero on every linear rung and 1 on
// desk_mixed (measured; the divided row is ~1/h^3 weaker at the 5y long-end spacing than the spacing-blind stencil that
// gave 0). The LIGHT TENSION and UNSMOOTHED rows carry the value-dependent rungs' measured counts on the factor walk
// (tension: mixed_scheme 82, desk_mixed 22; unsmoothed: 49, 85) plus ~25 % slack. Every pin may only DECREASE: lower one
// when you fix the cause, never raise one.
struct Pin { const char* name; long smoothed_per_row, tension_factor, unsmoothed_factor; };
const Pin kPins[] = {{"mixed_scheme", 0, 103, 62}, {"desk_mixed", 1, 28, 108}};
long pin_for(const std::string& name, Recipe r, WalkKind kind, long n) {
  if (r == Recipe::Smoothed && kind == WalkKind::Factor) return 0;
  for (const Pin& p : kPins)
    if (name == p.name) {
      const long per400 = r == Recipe::Smoothed ? p.smoothed_per_row : r == Recipe::LightTension ? p.tension_factor : p.unsmoothed_factor;
      return (per400 * n + 399) / 400;
    }
  return 0;
}

void run_walk(WalkKind kind, Recipe recipe) {
  const long n = soak_ticks();
  const int check_every = 40;
  std::printf("  [soak] %s walk, %s, %ld ticks per rung, accuracy every %d\n", swaps::shapes::to_string(kind), to_string(recipe), n, check_every);
  std::printf("  %-20s %6s %6s %6s %10s %10s %11s %8s  %s\n", "rung", "ticks", "failed", "maxrun", "mean us", "p99 us", "max|dq|",
              "zig bp", "failures by reason");
  for (const Shape& s : ladder()) {
    const Outcome o = soak(s, kind, reg_for(recipe, s), n, 20260921u, check_every);
    std::string why;
    for (const auto& [r, c] : o.reasons) why += (why.empty() ? "" : "; ") + std::to_string(c) + "x " + r;
    std::printf("  %-20s %6ld %6ld %6ld %10.1f %10.1f %11.2e %8.1f  %s\n", s.name.c_str(), o.ticks, o.failed, o.maxrun, o.mean_us,
                o.p99_us, o.worst_dq, 1e4 * o.worst_zig, why.c_str());
    EXPECT_LE(o.failed, pin_for(s.name, recipe, kind, n)) << s.name << " (" << swaps::shapes::to_string(kind) << " walk, " << to_string(recipe)
                                                    << "): failed ticks above the pin";
    // 0.01 bp between the delivered curve's quotes and the cold solve's on the realistic walk. The per-row walk's contradictory
    // quotes enlarge the frozen operator's second-order term (its fixed point is J_anchor^T r + R^T R x = 0, the cold solve's
    // J(x)^T r + R^T R x = 0): 0.013 bp measured on the compiled rungs under Light, judged at 0.05 bp. A value-dependent rung
    // on the per-row walk, or under a recipe that still fails ticks, is judged at 0.2 bp: there the fallback LM itself stalls
    // on the kinked landscape (0.11 bp measured on mixed_scheme, per-row smoothed).
    const bool clean = s.expect_compiled || (recipe == Recipe::Smoothed && kind == WalkKind::Factor);
    const double dq_tol = !clean ? 2e-5 : (kind == WalkKind::PerRow ? 5e-6 : 1e-6);
    EXPECT_LT(o.worst_dq, dq_tol) << s.name << ": the delivered curve's quotes differ from the cold solve's by more than the tolerance";
  }
  std::fflush(stdout);
}

}  // namespace

TEST(StreamingSoak, FactorWalkSmoothedEveryRung) { run_walk(WalkKind::Factor, Recipe::Smoothed); }
TEST(StreamingSoak, PerRowWalkSmoothedEveryRung) { run_walk(WalkKind::PerRow, Recipe::Smoothed); }
TEST(StreamingSoak, FactorWalkUnderLightTensionIsOnRecord) { run_walk(WalkKind::Factor, Recipe::LightTension); }
TEST(StreamingSoak, UnsmoothedFactorWalkIsTheStressCase) { run_walk(WalkKind::Factor, Recipe::Unsmoothed); }
