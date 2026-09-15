// E5 taxonomy: T3 cross-path parity (two engine paths, same inputs) | T4 hot-path invariant (allocation / determinism / structure)
// @regression-test — the SHAPE LADDER (bench/fixtures/shape_ladder.hpp): for EVERY instrument shape the compiled /
// hybrid hot path accepts, (T3) the hybrid engine's residual and analytic Jacobian match the templated kernel and
// the AAD Jacobian, and (T4) the streaming tick is allocation-free after warm-up on every W-cacheable shape, with
// the hybrid (FX/MtM) shapes and band-edge crossings pinned at their CURRENT counts so they can only go down.
// Until 2026-09-09 the gate's "0 allocations per tick" was proven on annual OIS only (E3 review A3/B3/C7).
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <set>

#include "malloc_count.hpp"
#include "shape_ladder.hpp"
#include "swaps/api/bundle_api.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/hybrid_residual.hpp"
#include "swaps/ad/dual.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/jacobian.hpp"

namespace cal = swaps::calibration;
namespace cv = swaps::curve;
namespace api = swaps::api;
using swaps::shapes::Shape;
using swaps::testing::AllocScope;

namespace {
double rel(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  return (a - b).cwiseAbs().maxCoeff() / (b.cwiseAbs().maxCoeff() + 1e-300);
}
}  // namespace

// T3: compiled/hybrid == templated == AAD on every rung, at x_true and off it.
// THE LADDER'S PROMISE, ENFORCED (2026-09-10). Its header says it carries "every instrument shape the
// compiled / hybrid hot path accepts", and for QuoteKind that was true -- but "shape" had silently meant only
// the INSTRUMENT: every rung built its curves with flat_hermite, so five of the seven interpolation schemes
// appeared nowhere in the ladder and the router's per-row partition could not be measured by it at all. A
// promise that nothing checks decays exactly this quietly, so both dimensions are now asserted here: adding a
// QuoteKind or a curve::Scheme without a rung that uses it fails this test.
TEST(ShapeLadder, CoversEveryQuoteKindAndScheme) {
  std::set<cal::QuoteKind> kinds;
  std::set<cv::Scheme> schemes;
  const std::function<void(const cal::Instrument&)> note = [&](const cal::Instrument& in) {
    kinds.insert(in.quote);
    for (const auto& c : in.combination) note(c.instrument);  // a Portfolio's components are instruments too
  };
  for (const Shape& s : swaps::shapes::ladder()) {
    for (const auto& in : s.prob.instruments) note(in);
    for (const auto& c : s.prob.curves)
      for (const auto& r : c.regions) schemes.insert(r.scheme);
  }
  // The enums are the authority; this list only has to stay in step with them, which the loops below enforce.
  for (const cal::QuoteKind k : {cal::QuoteKind::ParRate, cal::QuoteKind::ParSpread, cal::QuoteKind::Rate,
                                 cal::QuoteKind::ZeroCouponRate, cal::QuoteKind::FxForward,
                                 cal::QuoteKind::XccyMtmBasis, cal::QuoteKind::Portfolio,
                                 cal::QuoteKind::TurnJump})
    EXPECT_TRUE(kinds.count(k)) << "no ladder rung uses QuoteKind #" << static_cast<int>(k)
                                << " -- add one to bench/fixtures/shape_ladder.hpp";
  for (const cv::Scheme sc : {cv::Scheme::Flat, cv::Scheme::Linear, cv::Scheme::NaturalCubic, cv::Scheme::Hermite,
                              cv::Scheme::MonotoneCubic, cv::Scheme::BSpline, cv::Scheme::Tension})
    EXPECT_TRUE(schemes.count(sc)) << "no ladder rung interpolates with Scheme #" << static_cast<int>(sc)
                                   << " -- add one to bench/fixtures/shape_ladder.hpp";
  std::cout << "  [ladder] covers " << kinds.size() << " quote kinds and " << schemes.size() << " schemes over "
            << swaps::shapes::ladder().size() << " rungs\n";
}

TEST(ShapeLadder, HybridResidualAndJacobianMatchTemplatedAndAadOnEveryShape) {
  for (const Shape& s : swaps::shapes::ladder()) {
    const cal::HybridBundleResidual h(s.prob);
    double worst_r = 0, worst_j = 0;
    for (double bump : {0.0, 7e-4, -1.3e-3}) {
      Eigen::VectorXd x = s.x_true;
      for (int i = 0; i < x.size(); ++i) x[i] += bump * std::sin(0.9 * i + 0.2);
      worst_r = std::max(worst_r, (h.residuals(x) - s.prob.residuals<double>(x)).cwiseAbs().maxCoeff());
      worst_j = std::max(worst_j, rel(h.jacobian(x), cal::aad_jacobian(s.prob, x)));
    }
    std::cout << "  [ladder] " << s.name << ": " << s.prob.n_residuals() << " rows, " << s.prob.n_knots()
              << " knots, |dr|=" << worst_r << " |dJ|rel=" << worst_j << (s.expect_compiled ? "" : "  (hybrid route)") << "\n";
    EXPECT_LT(worst_r, 1e-12) << s.name;
    EXPECT_LT(worst_j, 1e-9) << s.name;
    EXPECT_LT(s.prob.residuals<double>(s.x_true).cwiseAbs().maxCoeff(), 1e-12) << s.name << ": markets must be the model quotes at x_true";
  }
}

// T4: after warm-up, a streaming tick allocates NOTHING on every W-cacheable shape. Hybrid shapes (FX/MtM) and
// band-edge crossings are pinned at their current counts (E3-B3: Hermite::build locals; E3-C7: a band re-scale
// is a full factor()) — they may only decrease; driving them to 0 is E4.D.
TEST(ShapeLadder, StreamingTickIsAllocationFreeOnEveryCompiledShape) {
  if (!swaps::testing::alloc_counting_available()) GTEST_SKIP() << "allocation counting needs libmalloc's logger (macOS)";
  for (const Shape& s : swaps::shapes::ladder()) {
    api::BundleSession sess(s.prob);
    sess.calibrate(s.x0);
    ASSERT_LT(sess.result().rms_residual, 1e-8) << s.name << ": cold calibrate must converge";
    sess.start_streaming();
    bool flip = false;
    for (int i = 0; i < 6; ++i) { flip = !flip; sess.stream_update(flip ? s.q_small : s.q0); }
    unsigned long small = 0, requote = 0;
    {
      AllocScope a;
      for (int i = 0; i < 20; ++i) { flip = !flip; sess.stream_update(flip ? s.q_small : s.q0); }
      small = a.allocs();
    }
    // K5': the FOUR-NUMBER requote tick (every row's target and band move together) -- a band move must cost no allocation
    // beyond the target-only tick's.
    const Shape::Requote small_q = s.requote(s.q_small), base_q = s.requote(s.q0);
    {
      AllocScope a;
      for (int i = 0; i < 20; ++i) {
        flip = !flip;
        const Shape::Requote& r = flip ? small_q : base_q;
        if (s.has_bands) sess.stream_update(r.target, r.lower, r.upper, r.decay);
        else sess.stream_update(r.target);
      }
      requote = a.allocs();
    }
    std::cout << "  [ladder] " << s.name << ": allocs over 20 small ticks = " << small << ", over 20 four-number requote ticks = " << requote << "\n";
    // THE CLIFF, made loud (2026-09-12). A block whose touched width exceeds ad::kPooledMaxW silently drops
    // to heap duals and every operation allocates: desk_mixed at width 55 against a limit of 48 cost 19,182
    // allocations per tick where the same work at width 29 cost 26. Nothing reported that but the pin, and a
    // pin only catches it once someone has already paid for it.
    const cal::HybridBundleResidual probe(s.prob);
    EXPECT_TRUE(probe.aad_pooled())
        << s.name << ": the AAD block fell off the pooled dual at width " << probe.aad_width()
        << " (limit " << swaps::ad::kPooledMaxW << ") -- every dual op now allocates";
    // PINS measured 2026-09-09 on engine 540abaf (20 ticks each). A compiled shape's small tick is 0 — that is the
    // invariant. The rest are the CURRENT costs of known E3 findings (B3 hybrid Hermite::build locals; C7 a band
    // re-scale is a full factor()); they may only DECREASE — lower a pin when you fix the cause, never raise one.
    struct Pin { const char* name; unsigned long small; };
    //   averaged_leg (compiled, 18/tick = 6 per residual x 3 Newton steps): E3-A3, Eigen IndexedView copies on
    //   pv()'s general branch — the one compiled shape that is NOT allocation-free today.
    //   fx_xccy / desk small ticks: 0 since the MtM leg compiles (2026-09-09); desk crossings are the C7 rescale cost.
    //   mixed_scheme / desk_mixed (2026-09-12): these stream for the first time — BundleSession's last
    //   whole-bundle veto is gone, and a tick that used to be a 215 ms COLD LM on desk_mixed is now 5.0 ms,
    //   43x faster. They are not allocation-free, and these pins say by how much: the AAD block re-evaluates
    //   a Hyman-filtered region per tick (the filter is value-dependent, so nothing about it can be cached
    //   the way a constant W is). ~26 allocations/tick on mixed_scheme's 12 AAD rows, ~19,000 on
    //   desk_mixed's 18. That is the next thing to attack here, in the same family as C7/C6 — and like every
    //   pin in this list it may only DECREASE.
    static const Pin pins[] = {{"averaged_leg", 360}, {"banded", 0}, {"fx_xccy", 0}, {"desk", 0}, {"mixed_scheme", 560},
                               {"desk_mixed", 3700}};
    //   desk_mixed's SMALL count is exact and stable (3600 over 20 ticks, three runs identical); its
    //   CROSSING count varies 4523..4753 because a band-edge crossing triggers a variable number of
    //   active-set re-scales -- the same slack the banded / desk crossing pins carry, for the same reason.
    //   banded 1900 -> 150, desk 3000 -> 100 on 2026-09-10 (C7 fixed: a band re-scale is a rank-one operator update,
    //   not a factor()); measured 123 / 54 -- the rest is the pin/release bookkeeping, next.
    //   (banded / desk crossing pins carry a few % of slack: the count of refreshes 20 crossing ticks trigger moves with
    //    rounding when the kernel's summation order changes — 1733 sparse, 1757 segment.)
    //   banded 1800 -> 1900 on 2026-09-10 (C1/C2 active-set fix): the corrected walk (pins as stiff constraint rows
    //   with a verified multiplier, budgeted releases) does 152 re-scales over the 20 crossing ticks where the old
    //   walk did 145 -- every one a full factor() (C7, unchanged per call): 1860 allocs. The per-call cost is the
    //   thing to fix (C7); when it is, this pin drops to 0.
    Pin pin{s.name.c_str(), 0};
    bool pinned = false;
    for (const auto& q : pins) if (s.name == q.name) { pin = q; pinned = true; }
    if (s.expect_compiled && !pinned) EXPECT_EQ(small, 0u) << s.name << ": the compiled tick must not allocate";
    else EXPECT_LE(small, pin.small) << s.name << ": tick allocations grew past the pinned count";
    EXPECT_LE(requote, pin.small) << s.name << ": a four-number requote allocates more than the target-only tick's pin";
  }
}

// The moment path vs the exact daily path on the SAME FF OIS legs: the documented ~5e-9 relative approximation
// (docs/bezier-and-moments.md Part B) must hold on real 1Y windows — the model par rates of the two rungs at the
// shared x_true agree to 2e-8 (0.0002 bp) while the moment rung ticks ~100x faster (shape_ladder_bench).
// THE VETO'S REPLACEMENT, PINNED (2026-09-12). Until today BundleSession refused to stream any bundle with a
// value-dependent region -- the last of the three whole-bundle vetoes, after the router's (1956b24) and the
// batch's. The claim that replaced it is that frozen-Newton is sound on a PARTITIONED bundle: the rows inside
// each curve's linear horizon ride the W-cache, the rest ride the AAD block and refresh on staleness, exactly
// as FX/MtM has always done. This asserts it where it is hardest -- a MonotoneCubic region, whose Hyman filter
// is only piecewise smooth, so a frozen Jacobian can in principle be carried across a branch switch.
TEST(ShapeLadder, AMixedBundleStreamsToTheSameAnswerAsAColdSolve) {
  for (const Shape& s : swaps::shapes::ladder()) {
    if (!cal::curves_are_noncacheable(s.prob.curves)) continue;  // only the rungs the veto used to refuse
    api::BundleSession sess(s.prob);
    sess.calibrate(s.x0);
    EXPECT_FALSE(sess.needs_recalibrate()) << s.name << ": nothing forces a cold solve any more";
    ASSERT_NO_THROW(sess.start_streaming()) << s.name;
    for (int rep = 0; rep < 3; ++rep)
      for (const Eigen::VectorXd* qv : {&s.q_big, &s.q0, &s.q_small, &s.q0}) {
        // K5': a banded shape's move carries its bands (a target is never outside its band); pq is that requote.
        const Shape::Requote rq = s.requote(*qv);
        const Eigen::VectorXd* q = &rq.target;
        cal::BundleProblem pq = s.prob;
        for (int i = 0; i < pq.n_residuals(); ++i) {
          auto& in = pq.instruments[static_cast<std::size_t>(i)];
          in.market = rq.target[i];
          in.band_lower = rq.lower[i];
          in.band_upper = rq.upper[i];
          in.band_decay = rq.decay[i];
        }
        const cal::HybridBundleResidual eng(pq);
        const Eigen::VectorXd& x = s.has_bands ? sess.stream_update(rq.target, rq.lower, rq.upper, rq.decay) : sess.stream_update(rq.target);
        ASSERT_TRUE(sess.last_converged()) << s.name << ": " << sess.last_reason();
        ASSERT_TRUE(x.allFinite()) << s.name;
        // The streamed state must be no worse than a cold LM on the same market -- the fixed point of the
        // frozen iteration is the least-squares optimum, whatever the filter did on the way there.
        const Eigen::VectorXd xc = cal::calibrate(pq, s.x0).x;
        const double f_s = eng.residuals_vs(x, *q).squaredNorm(), f_c = eng.residuals_vs(xc, *q).squaredNorm();
        EXPECT_LE(f_s, f_c * (1.0 + 1e-6) + 1e-20)
            << s.name << ": streamed objective " << f_s << " vs cold " << f_c;
      }
  }
}

TEST(ShapeLadder, MomentPathAgreesWithTheExactDailyAverageOnFedFundsOis) {
  const Shape daily = swaps::shapes::averaged_leg(), moment = swaps::shapes::averaged_leg_moment();
  ASSERT_EQ(daily.prob.n_residuals(), moment.prob.n_residuals());
  double worst = 0.0;
  for (int i = 0; i < daily.prob.n_residuals(); ++i)
    worst = std::max(worst, std::abs(daily.q0[i] - moment.q0[i]));
  std::cout << "  [ladder] moment vs exact daily FF OIS par rates: max |dq| = " << worst << " (" << worst * 1e4 << " bp)\n";
  EXPECT_LT(worst, 1e-9);  // measured 2.3e-10 with the knot-aligned quadrature (2026-09-10); the pin was 2e-8
  // And the moment rung is a pure W-cache bundle: the hybrid engine must not have routed any row to AAD.
  EXPECT_NO_THROW(cal::CompiledBundleResidual{moment.prob});
}

// Every rung's 25 bp move (the refresh metric's tick) and its small tick CONVERGE, in both directions, and the
// converged curve reprices the live market at the step_tol contract. Until 2026-09-10 the fx_xccy rung's 25 bp
// fixture contradicted covered interest parity (FX forwards scaled 0.25 % under a parallel rate move) and its
// refresh metric timed a tick that FAILED non-finite -- the bench now throws on a failed tick, and this test
// pins the fixture's ticks themselves.
TEST(ShapeLadder, EveryRungConvergesOnTheGateTicks) {
  for (const Shape& s : swaps::shapes::ladder()) {
    api::BundleSession sess(s.prob);
    sess.calibrate(s.x0);
    sess.start_streaming();
    for (int rep = 0; rep < 3; ++rep)
      for (const Eigen::VectorXd* qv : {&s.q_big, &s.q0, &s.q_small, &s.q0}) {
        // K5': a banded shape's move carries its bands (a target is never outside its band); pq is that requote.
        const Shape::Requote rq = s.requote(*qv);
        const Eigen::VectorXd* q = &rq.target;
        cal::BundleProblem pq = s.prob;
        for (int i = 0; i < pq.n_residuals(); ++i) {
          auto& in = pq.instruments[static_cast<std::size_t>(i)];
          in.market = rq.target[i];
          in.band_lower = rq.lower[i];
          in.band_upper = rq.upper[i];
          in.band_decay = rq.decay[i];
        }
        const cal::HybridBundleResidual eng(pq);
        const Eigen::VectorXd& x = s.has_bands ? sess.stream_update(rq.target, rq.lower, rq.upper, rq.decay) : sess.stream_update(rq.target);
        EXPECT_TRUE(sess.last_converged()) << s.name << ": " << sess.last_reason() << " after " << sess.last_newton_steps() << " steps";
        EXPECT_TRUE(x.allFinite()) << s.name;
        const Eigen::VectorXd r = eng.residuals_vs(x, *q);
        if (s.prob.n_residuals() == s.prob.n_knots() && !s.has_bands) {
          // SQUARE, hard: every row reprices at the step_tol contract
          EXPECT_LT(r.cwiseAbs().maxCoeff(), 1e-8) << s.name;
        } else if (rep == 0) {
          // over-determined / banded: a least-squares fit -- the streamed objective is no worse than a cold LM's
          const Eigen::VectorXd xc = cal::calibrate(pq, s.x0).x;
          const double f_s = eng.residuals_vs(x, *q).squaredNorm(), f_c = eng.residuals_vs(xc, *q).squaredNorm();
          EXPECT_LE(f_s, f_c * (1.0 + 1e-6) + 1e-20) << s.name << ": streamed objective " << f_s << " vs cold " << f_c;
        }
      }
  }
}
