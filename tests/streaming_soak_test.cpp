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
// So the gate pins two things per rung:
//   SMOOTHED (the API's Light preset: the curvature penalty over every curve) -- the recipe that streams EVERY rung: ZERO
//     failed ticks on the realistic walk. The invariant, on every platform.
//   UNSMOOTHED -- the stress case, its measured counts, may only decrease -- PER NUMERIC FINGERPRINT (see PINS below).
// (A LIGHT TENSION row was pinned here from 2026-09-21 until the tension-energy operator was retired on 2026-09-22: at
// equal weight it was indistinguishable from the curvature penalty -- 0 / 0 failed ticks and 34.8 vs 34.9 us on
// desk_mixed's factor walk -- so it was removed as a feature. Its numbers above are the historical record.)
// Plus, on every row: ACCURACY of the delivered curve in QUOTE space (CLAUDE.md 3b: never knot-space distance on a
// weakly-identified bundle) -- its model quotes against those of a cold solve of the SAME objective (targets, bands,
// regulariser) refined FROM it, every `check_every` ticks (null: a square rung's converged tick is exact, ~1e-12); and,
// informational, the largest second difference of the first curve's knot forwards (the zigzag amplitude).
//
// SWAPS_SOAK_TICKS (default 400) lengthens the walks for a local soak (pins are per-400-tick counts scaled to the run);
// SWAPS_SOAK_LAMBDA overrides the smoothed rows' weight for local experiments only.
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <boost/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "market_walk.hpp"
#include "shape_ladder.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/calibration/regularize.hpp"
#include "swaps/simd_config.hpp"

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

enum class Recipe { Smoothed, Unsmoothed };
const char* to_string(Recipe r) {
  switch (r) {
    case Recipe::Smoothed: return "SMOOTHED (the API's Light preset: second-difference over every curve)";
    case Recipe::Unsmoothed: return "UNSMOOTHED (the stress case)";
  }
  return "";
}

cal::RegSpec reg_for(Recipe r, const Shape& s) {
  const int nc = static_cast<int>(s.prob.curves.size());
  switch (r) {
    case Recipe::Unsmoothed: return cal::RegSpec{};
    case Recipe::Smoothed: {
      cal::RegSpec reg = cal::smoothing_preset(cal::Smoothing::Light, nc);  // THE shipped default
      if (const char* e = std::getenv("SWAPS_SOAK_LAMBDA")) reg.lambda = std::atof(e);  // local experiments only
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

// PINS. Two kinds (2026-09-22):
//   UNIVERSAL -- platform-independent by construction, hard-coded here: every LINEAR rung on every walk and recipe, and
//     EVERY rung on the smoothed FACTOR walk (the realistic day under the shipped default), at ZERO failed ticks; the
//     delivered curve within the class dq tolerance (0.01 bp factor / 0.05 bp per-row: the frozen operator's second-order
//     term on contradictory quotes, 0.013 bp measured on the compiled rungs under Light).
//   PER NUMERIC FINGERPRINT -- baselines/soak_pins.json, keyed OS|compiler|arch flags|ISA exactly as the perf baselines
//     are keyed by machine+toolchain: a VALUE-DEPENDENT rung (MonotoneCubic) on the per-row stress walk or under no
//     smoothing sits on Hyman branch boundaries where an ulp decides the branch, so its failed-tick count (and how far a
//     kinked fallback LM lands from another) is deterministic PER instruction stream and differs across them: the same
//     seeded walk read 0 on mixed_scheme under Apple clang and 2 under GCC/glibc (CI, 2026-09-22). A row with no entry
//     for the running key is REPORT-ONLY and the test prints the stanza to paste. Pins only ever DECREASE per key.
// (The 2026-09-21 Mac measurements: smoothed per-row mixed_scheme 0 / desk_mixed 1; unsmoothed factor 62 / 108 incl. ~25 %
// slack -- the divided row is ~1/h^3 weaker at the 5y long-end spacing than the spacing-blind stencil that gave 0.)
std::string numeric_key() {
#if defined(__APPLE__)
  const char* os = "Darwin";
#elif defined(__linux__)
  const char* os = "Linux";
#elif defined(_WIN32)
  const char* os = "Windows";
#else
  const char* os = "unknown";
#endif
  namespace d = swaps::simd::detected;
  return std::string(os) + "|" + d::compiler_id + " " + d::compiler_version + "|" + d::arch_flags + "|" + d::isa_name;
}
struct RowPin { long failed; double dq; };
struct PinTable {
  std::string key;
  bool known = false;
  std::map<std::string, RowPin> rows;  // "<rung>/<row>" -> pin
};
const PinTable& pins() {
  static const PinTable table = [] {
    PinTable t;
    t.key = numeric_key();
    const std::string path = std::string(SWAPS_BASELINES_DIR) + "/soak_pins.json";
    std::ifstream in(path);
    if (!in.good()) throw std::runtime_error("missing " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    const boost::json::value doc = boost::json::parse(ss.str());
    const boost::json::object& keys = doc.at("keys").as_object();
    if (const auto it = keys.find(t.key); it != keys.end()) {
      t.known = true;
      for (const auto& rung : it->value().at("rungs").as_object())
        for (const auto& row : rung.value().as_object())
          t.rows[std::string(rung.key()) + "/" + std::string(row.key())] = {row.value().at("failed").as_int64(),
                                                                          row.value().at("dq").to_number<double>()};
    }
    return t;
  }();
  return table;
}
const char* row_name(WalkKind kind, Recipe r) {
  const bool per_row = kind == WalkKind::PerRow;
  return r == Recipe::Smoothed ? (per_row ? "smoothed_per_row" : "smoothed_factor") : (per_row ? "unsmoothed_per_row" : "unsmoothed_factor");
}

void run_walk(WalkKind kind, Recipe recipe) {
  const long n = soak_ticks();
  const int check_every = 40;
  std::printf("  [soak] %s walk, %s, %ld ticks per rung, accuracy every %d\n", swaps::shapes::to_string(kind), to_string(recipe), n, check_every);
  std::printf("  [soak] numeric fingerprint %s: %s\n", pins().key.c_str(), pins().known ? "pinned in baselines/soak_pins.json" : "NO ENTRY -- value-dependent rows are report-only");
  std::printf("  %-20s %6s %6s %6s %10s %10s %11s %8s  %s\n", "rung", "ticks", "failed", "maxrun", "mean us", "p99 us", "max|dq|",
              "zig bp", "failures by reason");
  std::map<std::string, Outcome> unpinned;
  for (const Shape& s : ladder()) {
    const Outcome o = soak(s, kind, reg_for(recipe, s), n, 20260921u, check_every);
    std::string why;
    for (const auto& [r, c] : o.reasons) why += (why.empty() ? "" : "; ") + std::to_string(c) + "x " + r;
    std::printf("  %-20s %6ld %6ld %6ld %10.1f %10.1f %11.2e %8.1f  %s\n", s.name.c_str(), o.ticks, o.failed, o.maxrun, o.mean_us,
                o.p99_us, o.worst_dq, 1e4 * o.worst_zig, why.c_str());
    const bool universal = s.expect_compiled || (recipe == Recipe::Smoothed && kind == WalkKind::Factor);
    if (universal) {
      EXPECT_EQ(o.failed, 0) << s.name << " (" << swaps::shapes::to_string(kind) << " walk, " << to_string(recipe)
                             << "): a failed tick on a platform-independent row";
      const double dq_tol = kind == WalkKind::PerRow ? 5e-6 : 1e-6;
      EXPECT_LT(o.worst_dq, dq_tol) << s.name << ": the delivered curve's quotes differ from the cold solve's by more than the tolerance";
      continue;
    }
    const std::string id = s.name + "/" + row_name(kind, recipe);
    const auto it = pins().rows.find(id);
    if (it == pins().rows.end()) {
      std::printf("  %-20s REPORT-ONLY: no pin for this row under the running fingerprint\n", "");
      unpinned[id] = o;
      continue;
    }
    EXPECT_LE(o.failed, (it->second.failed * n + 399) / 400) << id << ": failed ticks above the pin for " << pins().key;
    EXPECT_LT(o.worst_dq, it->second.dq) << id << ": the delivered curve's quotes differ from the cold solve's by more than the pinned tolerance for " << pins().key;
  }
  if (!unpinned.empty()) {  // the stanza to paste under this key (measured values; give dq ~1.5x slack)
    std::printf("  [soak] measured, unpinned rows for \"%s\":\n", pins().key.c_str());
    for (const auto& [id, o] : unpinned) std::printf("    %-36s {\"failed\": %ld, \"dq\": %.1e}\n", id.c_str(), o.failed, o.worst_dq);
  }
  std::fflush(stdout);
}

}  // namespace

TEST(StreamingSoak, FactorWalkSmoothedEveryRung) { run_walk(WalkKind::Factor, Recipe::Smoothed); }
TEST(StreamingSoak, PerRowWalkSmoothedEveryRung) { run_walk(WalkKind::PerRow, Recipe::Smoothed); }
TEST(StreamingSoak, UnsmoothedFactorWalkIsTheStressCase) { run_walk(WalkKind::Factor, Recipe::Unsmoothed); }
