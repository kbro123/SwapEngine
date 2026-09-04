// @oracle-test — validates against QuantLib cashflow-for-cashflow. DO NOT DELETE OR WEAKEN
// without reproducing the QuantLib comparison. See tests/ORACLE_TESTS.md.
// Stage 3 gate on REALISTIC curve builds: four curves, each flat-forward meeting-date front + par-swap
// back (the production SOFR structure), calibrated simultaneously and validated against QuantLib.
//   SOFR   : real reference market -- 12x1M + 8x3M SOFR futures + 9 par swaps (build_market/build_problem)
//   FF     : 12x1M FF futures (front) + FF-SOFR basis at 18m/2y/3y + swap maturities (replaces 3M futures)
//   PRIME  : basis over FF at 3m/6m/9m/12m + 18m/2y/3y + swap maturities (basis only, no futures)
//   PRIME2 : basis over PRIME, same tenors
// Quotes are generated from a known set of forwards, so the joint solve must recover them. The MULTI-CURVE
// basis pricing (the new Stage-3 kernel) is additionally oracle-checked against QuantLib's own multi-curve
// OIS fair spread. SOFR/FF futures reuse the single-curve kernel already validated to 1e-16 (pricing_test).

#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include <chrono>
#include <thread>

#include "reference_bundle.hpp"
#include "reference_curve.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/bundle_stage.hpp"
#include "swaps/parallel/thread_pool.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/calibration/streaming.hpp"
#include "swaps/calibration/warm.hpp"
#include "swaps/curve/curve_module.hpp"
#include "swaps/ql/ql_term_structure.hpp"
#include "swaps/ql/extract.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace cv = swaps::curve;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace rb = swaps::refbuild;
namespace rm = swaps::refmkt;

struct BundleRealistic : ::testing::Test {
  RelinkableHandle<YieldTermStructure> hS;         // SOFR handle (the discount curve for all)
  rb::Market mk = rb::build_market(hS);
  Date today = mk.today;
  DayCounter dc = mk.dc;
  Calendar cal = UnitedStates(UnitedStates::GovernmentBond);
  enum { SOFR = 0, FF = 1, PRIME = 2, PRIME2 = 3, NC = 4 };

  std::vector<RelinkableHandle<YieldTermStructure>> h;
  std::vector<ext::shared_ptr<OvernightIndex>> idx;   // FF/PRIME/PRIME2 forecast indices
  std::vector<std::unique_ptr<cal::CurveHandle<double>>> curve_handles;  // actual fwd curves (base+spread)
  std::vector<ext::shared_ptr<YieldTermStructure>> ts;
  cal::BundleProblem prob;
  Eigen::VectorXd x_true;
  std::vector<int> off;  // stacked offset per curve
  // Exact instrument tenors as QuantLib Periods (no double->Period rounding).
  std::vector<Period> back_periods{18 * Months, 2 * Years, 3 * Years, 4 * Years, 5 * Years, 7 * Years,
                                   10 * Years,  12 * Years, 15 * Years, 20 * Years, 25 * Years, 30 * Years};
  std::vector<Period> mon_periods{3 * Months, 6 * Months, 9 * Months, 12 * Months};

  void SetUp() override {
    Settings::instance().evaluationDate() = today;
    const cal::CalibrationProblem sofr = rb::build_problem(mk);  // real SOFR: meetings + 1M/3M futures + swaps

    // ---- knot specs (all curves: flat meeting-date front + par-swap back). Knot times are the exact
    // date-based year fractions of each tenor. SOFR: reference knots. FF: meeting front + basis-tenor
    // back. PRIME/PRIME2: monthly front + same back. ----
    std::vector<double> back_t;
    for (const Period& p : back_periods) back_t.push_back(dc.yearFraction(today, today + p));
    std::vector<double> mon_t;
    for (const Period& p : mon_periods) mon_t.push_back(dc.yearFraction(today, today + p));

    // Convention (matches a trading desk): only SOFR is an OUTRIGHT curve; every other curve is a
    // SPREAD to the one below it, so its free variables are forward SPREADS (base < curve index):
    //   FF = SOFR + spread, PRIME = FF + spread, PRIME2 = PRIME + spread.
    prob.curves.resize(NC);
    prob.curves[SOFR] = swaps::pricing::CurveStructure{.base = -1, .regions = swaps::curve::flat_hermite(sofr.meeting_times, sofr.back_times)};        // outright
    prob.curves[FF] = swaps::pricing::CurveStructure{.base = SOFR, .regions = swaps::curve::flat_hermite(sofr.meeting_times, back_t)};                 // spread over SOFR
    prob.curves[PRIME] = swaps::pricing::CurveStructure{.base = FF, .regions = swaps::curve::flat_hermite(mon_t, back_t)};                            // spread over FF
    prob.curves[PRIME2] = swaps::pricing::CurveStructure{.base = PRIME, .regions = swaps::curve::flat_hermite(mon_t, back_t)};                        // spread over PRIME
    off.assign(NC, 0);
    for (int c = 1; c < NC; ++c) off[c] = off[c - 1] + prob.curves[c - 1].n_knots();
    const int N = off[NC - 1] + prob.curves[NC - 1].n_knots();

    // x_true: SOFR is a FORWARD level (~4.3%); FF/PRIME/PRIME2 are forward SPREADS to their base
    // (FF ~ -6bp vs SOFR, PRIME ~ +306bp vs FF, PRIME2 ~ +50bp vs PRIME -- so the resulting forwards
    // are SOFR ~4.3%, FF ~4.24%, PRIME ~7.3%, PRIME2 ~7.8%, as before). Gentle slope on each block.
    const double base[] = {0.0430, -0.0006, 0.0306, 0.0050};
    x_true.resize(N);
    for (int c = 0; c < NC; ++c) {
      const int nk = prob.curves[c].n_knots();
      const double slope = (c == SOFR) ? 0.0003 : 0.0001;  // spreads slope gently
      for (int i = 0; i < nk; ++i) x_true[off[c] + i] = base[c] + slope * i;
    }

    // A ParSpread basis instrument from a QuantLib OIS: the spread (quoted) leg forecasts `fwd_fc`, the
    // benchmark leg forecasts `bench_fc` (SAME coupons), both discount `disc`. This is exactly the old
    // basis_par_spread(sched, fwd_fc, bench_fc, disc) as one generic instrument.
    auto basis_inst = [&](const OvernightIndexedSwap& o, int fwd_fc, int bench_fc, int disc) {
      const auto fl = swaps::qlx::extract_float_leg(o.overnightLeg(), today, dc);
      const auto fx = swaps::qlx::extract_fixed_leg(o.fixedLeg(), today, dc);
      cal::Instrument ins;
      ins.quote = cal::QuoteKind::ParSpread;
      ins.fwd = {fl, fwd_fc, disc};
      ins.bench = {fl, bench_fc, disc};
      ins.fixed = {fx, disc};
      return ins;
    };

    // ---- SOFR instruments (curve 0): the reference market as generic Instruments. build_problem's
    // instruments already carry role 0 = SOFR (single-curve => every role is curve 0), so reuse them. ----
    for (const auto& ins : sofr.instruments) prob.instruments.push_back(ins);

    // ---- FF/PRIME/PRIME2 indices ----
    const char* names[] = {"", "FFx", "PRIMEx", "PRIME2x"};
    h.resize(NC);
    idx.resize(NC);
    h[SOFR] = hS;
    for (int c = 1; c < NC; ++c) {
      h[c].linkTo(ext::make_shared<FlatForward>(today, 0.04, dc, Continuous));
      idx[c] = ext::make_shared<OvernightIndex>(names[c], 0, USDCurrency(), cal, Actual360(), h[c]);
    }

    // FF front: 12 1M FF averaging futures (Rate; observation from the SOFR 1M-future dates, forecast
    // off FF -- the observation is curve-agnostic, only the forecast role selects the curve).
    for (int i = 0; i < 12; ++i) {
      const auto& q = rm::futures_1m[i];
      const Date s = rb::sofr_start(Month(q.ref_month), q.ref_year, Monthly),
                 e = rb::sofr_end(Month(q.ref_month), q.ref_year, Monthly);
      cal::Instrument ins;
      ins.quote = cal::QuoteKind::Rate;
      ins.obs = rb::avg_future_obs(mk, rb::Future{s, e, false, q.price});
      ins.forecast = FF;
      prob.instruments.push_back(ins);
    }
    // FF back: FF-SOFR basis (spread leg forecasts FF, benchmark SOFR, both discount SOFR).
    for (const Period& p : back_periods) {
      auto o = ext::shared_ptr<OvernightIndexedSwap>(MakeOIS(p, idx[FF], 0.03).withDiscountingTermStructure(hS));
      prob.instruments.push_back(basis_inst(*o, FF, SOFR, SOFR));
    }
    // PRIME / PRIME2: basis over the previous curve at the monthly front + back tenors.
    std::vector<Period> all = mon_periods;
    all.insert(all.end(), back_periods.begin(), back_periods.end());
    for (int c = PRIME; c <= PRIME2; ++c)
      for (const Period& p : all) {
        auto o = ext::shared_ptr<OvernightIndexedSwap>(MakeOIS(p, idx[c], 0.03).withDiscountingTermStructure(hS));
        prob.instruments.push_back(basis_inst(*o, c, c - 1, SOFR));
      }

    // ---- Actual forward curves at x_true (SOFR outright; FF/PRIME/PRIME2 = base+spread), exposed to
    // QuantLib as YieldTermStructures. build_bundle_curves resolves the spread chain exactly, and the
    // resulting CurveHandle (virtual discount) plugs straight into CurveTermStructure. ----
    curve_handles = cal::build_bundle_curves<double>(
        prob.curves, [&](int c, int i) { return x_true[off[c] + i]; });
    ts.resize(NC);
    for (int c = 0; c < NC; ++c) {
      auto t = ext::make_shared<swaps::qlx::CurveTermStructure<cal::CurveHandle<double>>>(
          today, dc, curve_handles[c].get());
      t->enableExtrapolation();
      ts[c] = t;
      h[c].linkTo(t);
    }
    for (auto& s : mk.swaps) s->deepUpdate();

    // Market quotes generated from x_true (self-consistent), so the joint solve recovers x_true.
    const Eigen::VectorXd r0 = prob.residuals<double>(x_true);
    for (int i = 0; i < static_cast<int>(prob.instruments.size()); ++i) prob.instruments[i].market += r0[i];
  }
};

TEST_F(BundleRealistic, JointAndStagedRecoverAllFourCurves) {
  std::cout << "  [bundle-real] knots per curve: SOFR=" << prob.curves[SOFR].n_knots()
            << " FF=" << prob.curves[FF].n_knots() << " PRIME=" << prob.curves[PRIME].n_knots()
            << " PRIME2=" << prob.curves[PRIME2].n_knots() << "  total knots=" << prob.n_knots()
            << " instruments=" << prob.n_residuals() << "\n";
  EXPECT_LT(prob.residuals<double>(x_true).cwiseAbs().maxCoeff(), 1e-12);

  Eigen::VectorXd x0(prob.n_knots());  // per-curve flat start near each curve's level
  const double start[] = {0.043, -0.0006, 0.0306, 0.0050};  // SOFR forward; FF/PRIME/PRIME2 SPREADS
  for (int c = 0; c < NC; ++c) x0.segment(off[c], prob.curves[c].n_knots()).setConstant(start[c]);
  const auto joint = cal::calibrate(prob, x0);
  const auto staged = cal::calibrate_staged(prob, x0);
  for (int c = 0; c < NC; ++c) {
    const int nk = prob.curves[c].n_knots();
    std::cout << "  [bundle-real] curve " << c << " joint err="
              << (joint.x.segment(off[c], nk) - x_true.segment(off[c], nk)).cwiseAbs().maxCoeff()
              << " staged err="
              << (staged.x.segment(off[c], nk) - x_true.segment(off[c], nk)).cwiseAbs().maxCoeff() << "\n";
  }
  std::cout << "  [bundle-real] joint  ||x*-xtrue||=" << (joint.x - x_true).cwiseAbs().maxCoeff()
            << " iters=" << joint.iterations << " info=" << joint.info << " rms=" << joint.rms_residual
            << " stat=" << joint.stationarity << "\n";
  std::cout << "  [bundle-real] staged ||x*-xtrue||=" << (staged.x - x_true).cwiseAbs().maxCoeff()
            << " iters=" << staged.iterations << "\n";
  EXPECT_LT((joint.x - x_true).cwiseAbs().maxCoeff(), 1e-6) << "joint solve recovers all four realistic curves";
  EXPECT_LT((staged.x - x_true).cwiseAbs().maxCoeff(), 1e-6) << "staged solve recovers all four realistic curves";
}

TEST_F(BundleRealistic, MultiCurveBasisMatchesQuantLib) {
  // The FF-SOFR basis (forecast FF, discount SOFR) must equal QuantLib's OIS fair rate spread:
  // s = fairRate(FF OIS, SOFR-disc) - fairRate(SOFR OIS, SOFR-disc), at x_true.
  double worst = 0.0;
  int n = 0;
  for (const Period& p : back_periods) {
    auto ff = ext::shared_ptr<OvernightIndexedSwap>(MakeOIS(p, idx[FF], 0.03).withDiscountingTermStructure(hS));
    auto sf = ext::shared_ptr<OvernightIndexedSwap>(MakeOIS(p, mk.sofr, 0.03).withDiscountingTermStructure(hS));
    ff->deepUpdate();
    sf->deepUpdate();
    // Our basis_par_spread(fwd, bench, disc) = r_bench - r_fwd, so with fwd=FF, bench=SOFR it is
    // r_SOFR - r_FF = sf.fairRate - ff.fairRate.
    const double ql = sf->fairRate() - ff->fairRate();
    const auto fl = swaps::qlx::extract_float_leg(ff->overnightLeg(), today, dc);
    const auto fx = swaps::qlx::extract_fixed_leg(ff->fixedLeg(), today, dc);
    // par_spread(fwd_leg, bench_leg, annuity, fwd, bench, disc) = r_bench - r_fwd; fwd=FF, bench=SOFR.
    const double ours =
        swaps::pricing::par_spread<double>(fl, fl, fx, *curve_handles[FF], *curve_handles[SOFR],
                                           *curve_handles[SOFR]);
    worst = std::max(worst, std::abs(ours - ql));
    ++n;
  }
  std::cout << "  [bundle-real] FF-SOFR basis max |ours - QuantLib| = " << worst << " over " << n << " tenors\n";
  EXPECT_LT(worst, swaps::tol::curve_rel) << "multi-curve basis pricing must match QuantLib";
}

TEST_F(BundleRealistic, CompiledBundleResidualMatchesAad) {
  // The multi-curve W-cache residual must reproduce the templated BundleProblem residual to machine
  // precision, and its ANALYTIC block Jacobian must match the AAD Jacobian -- this is what lets the
  // warm/streaming re-calibrators drive the bundle on the fast path instead of an AAD sweep.
  cal::CompiledBundleResidual cr(prob);
  double worst_r = 0, worst_j = 0;
  for (double bump : {0.0, 7e-4, -1.3e-3}) {  // at x_true and two off-solution points
    Eigen::VectorXd x = x_true;
    for (int i = 0; i < x.size(); ++i) x[i] += bump * std::sin(0.9 * i + 0.2);
    worst_r = std::max(worst_r, (cr.residuals(x) - prob.residuals<double>(x)).cwiseAbs().maxCoeff());
    worst_j = std::max(worst_j, (cr.jacobian(x) - cal::aad_jacobian(prob, x)).cwiseAbs().maxCoeff());
  }
  std::cout << "  [bundle-compiled] worst |residual - AAD|=" << worst_r
            << "  worst |Jacobian - AAD|=" << worst_j << "\n";
  EXPECT_LT(worst_r, 1e-12) << "compiled bundle residual must match the templated residual";
  EXPECT_LT(worst_j, 1e-8) << "analytic block Jacobian must match AAD";
}

TEST_F(BundleRealistic, WarmRecalMatchesColdResolveOnMinorPerturbation) {
  // Base cold solve (per-curve flat start), then a MINOR (~1bp) market perturbation. The warm
  // frozen-Jacobian re-cal reusing the base Jacobian must land on an INDEPENDENT cold LM re-solve of
  // the perturbed market -- and the one-matvec linear update must be first-order accurate. This is the
  // 4-curve-bundle analogue of the single-curve Warm gate: same WarmCalibrator, driven via BundleProblem.
  Eigen::VectorXd x0(prob.n_knots());
  const double start[] = {0.043, -0.0006, 0.0306, 0.0050};  // SOFR forward; FF/PRIME/PRIME2 SPREADS
  for (int c = 0; c < NC; ++c) x0.segment(off[c], prob.curves[c].n_knots()).setConstant(start[c]);
  const Eigen::VectorXd x_solved = cal::calibrate(prob, x0).x;

  Eigen::VectorXd dq(prob.n_residuals());  // ~1bp move, varied sign/shape across quotes (residual order)
  for (int i = 0; i < dq.size(); ++i) dq[i] = 1e-4 * std::sin(0.7 * i + 0.3);

  // Ground truth: an independent COLD LM re-solve of the perturbed market (market += dq, residual order).
  cal::BundleProblem pert = prob;
  for (int i = 0; i < static_cast<int>(pert.instruments.size()); ++i) pert.instruments[i].market += dq[i];
  const Eigen::VectorXd x_cold = cal::calibrate(pert, x_solved).x;

  // WARM: reuse the base Jacobian; only refresh if the move leaves the envelope (1bp should not).
  cal::WarmCalibrator<cal::BundleProblem> wc(prob, x_solved);
  const cal::WarmResult wr = wc.recalibrate(dq);
  const Eigen::VectorXd x_lin = wc.recalibrate_linear(dq);

  const double warm_err = (wr.x - x_cold).cwiseAbs().maxCoeff();
  const double lin_err = (x_lin - x_cold).cwiseAbs().maxCoeff();
  std::cout << "  [bundle-warm] steps=" << wr.steps << " refreshes=" << wr.jacobian_refreshes
            << " converged=" << wr.converged << "  warm ||x-x_cold||=" << warm_err
            << "  linear ||x-x_cold||=" << lin_err << "\n";
  EXPECT_TRUE(wr.converged);
  EXPECT_LT(warm_err, 1e-7) << "warm frozen-Newton must match an independent cold LM re-solve";
  EXPECT_LT(lin_err, 1e-5) << "one-matvec linear update is first-order accurate at ~1bp";
}

TEST_F(BundleRealistic, StreamingPrefetchHidesTheRefreshSpike) {
  // The speculative background Jacobian on the multi-curve bundle, where a refresh recomputes the
  // (analytic) block Jacobian -- ~ms, vs a ~µs fast tick. With a live-cadence feed (ticks paced so the
  // worker has wall-clock time between refreshes, as on a real desk), the prefetch serves refreshes off
  // the background thread, so the WORST prefetch tick is a µs matrix-swap while the WORST sync tick pays
  // the full inline Jacobian. Exactness is identical (frozen-Newton is exact for any invertible M).
  Eigen::VectorXd x0(prob.n_knots());
  const double start[] = {0.043, -0.0006, 0.0306, 0.0050};
  for (int c = 0; c < NC; ++c) x0.segment(off[c], prob.curves[c].n_knots()).setConstant(start[c]);
  const Eigen::VectorXd x_solved = cal::calibrate(prob, x0).x;
  const Eigen::VectorXd mkt = prob.market();
  auto model_rates = [&](const Eigen::VectorXd& x) { return (prob.residuals<double>(x) + mkt).eval(); };

  cal::StreamingCalibrator<cal::BundleProblem> sync(prob, x_solved, model_rates(x_solved),
                                                    cal::StreamingCalibrator<cal::BundleProblem>::Options{});
  cal::StreamingCalibrator<cal::BundleProblem>::Options popt;
  popt.prefetch = true;
  popt.prefetch_drift = 3e-4;
  cal::StreamingCalibrator<cal::BundleProblem> pref(prob, x_solved, model_rates(x_solved), popt);

  double max_sync = 0, max_pref = 0, worst_rt = 0;
  int hits = 0, refreshes_pref = 0;
  for (int t = 1; t <= 200; ++t) {
    Eigen::VectorXd xp = x_solved;  // a realistic SLOW trend (~0.3bp/tick) that crosses the envelope a
    for (int i = 0; i < xp.size(); ++i)   // few times -- the worker has ample lead between refreshes.
      xp[i] += 60e-4 * (t / 200.0) * (0.7 + 0.3 * std::sin(0.5 * i));
    const Eigen::VectorXd q = model_rates(xp);
    auto a = std::chrono::steady_clock::now();
    sync.update(q);
    auto b = std::chrono::steady_clock::now();
    const auto tick = pref.update(q);
    auto c = std::chrono::steady_clock::now();
    max_sync = std::max(max_sync, std::chrono::duration<double, std::micro>(b - a).count());
    max_pref = std::max(max_pref, std::chrono::duration<double, std::micro>(c - b).count());
    hits += tick.prefetched;
    refreshes_pref += tick.refreshes;
    worst_rt = std::max(worst_rt, (model_rates(pref.current()) - q).cwiseAbs().maxCoeff());
    std::this_thread::sleep_for(std::chrono::microseconds(600));  // live-cadence pacing: give the worker time
  }
  std::cout << "  [bundle-prefetch] refreshes=" << refreshes_pref << " of which prefetch-served=" << hits << "\n";
  std::cout << "  [bundle-prefetch] worst tick: sync=" << max_sync << "us prefetch=" << max_pref
            << "us  prefetch_hits=" << hits << " round-trip=" << worst_rt << "\n";
  EXPECT_LT(worst_rt, 1e-8) << "prefetch bundle stream must still reprice exactly every tick";
  EXPECT_GT(hits, 0) << "the background Jacobian must have served refreshes";
  EXPECT_LT(max_pref, max_sync) << "the worst prefetch tick must beat the worst sync tick (spike hidden)";
}

TEST_F(BundleRealistic, StreamingExactPathRoundTripsTheBundle) {
  // The EXACT streaming path, driven by BundleProblem, must reprice the whole 4-curve instrument set
  // back to each tick's quotes. Quotes are generated from a perturbed curve (so they are achievable
  // even though the bundle is over-determined), and every tick must round-trip to the Newton tolerance.
  Eigen::VectorXd x0(prob.n_knots());
  const double start[] = {0.043, -0.0006, 0.0306, 0.0050};  // SOFR forward; FF/PRIME/PRIME2 SPREADS
  for (int c = 0; c < NC; ++c) x0.segment(off[c], prob.curves[c].n_knots()).setConstant(start[c]);
  const Eigen::VectorXd x_solved = cal::calibrate(prob, x0).x;

  const Eigen::VectorXd mkt = prob.market();
  auto model_rates = [&](const Eigen::VectorXd& x) { return (prob.residuals<double>(x) + mkt).eval(); };

  cal::StreamingCalibrator<cal::BundleProblem>::Options opt;  // exact = true
  cal::StreamingCalibrator sc(prob, x_solved, model_rates(x_solved), opt);

  double worst_rt = 0;
  for (int t = 1; t <= 120; ++t) {
    Eigen::VectorXd xp = x_solved;  // a smoothly-drifting curve move (up to ~15bp), quotes from it
    for (int i = 0; i < xp.size(); ++i) xp[i] += 15e-4 * std::sin(0.05 * t) * std::sin(0.5 * i + 1.0);
    const Eigen::VectorXd q = model_rates(xp);
    sc.update(q);
    worst_rt = std::max(worst_rt, (model_rates(sc.current()) - q).cwiseAbs().maxCoeff());
  }
  std::cout << "  [bundle-stream] worst round-trip=" << worst_rt << " recalcs=" << sc.refresh_count() - 1
            << "\n";
  EXPECT_LT(worst_rt, 1e-8) << "every tick must reprice the 4-curve bundle back to the input quotes";
}

// ---- Spread-parameterized bundle curves -------------------------------------------------------
// A curve's DEFINITION carries whether it is OUTRIGHT (own forwards) or a SPREAD over another bundle
// curve (its free vars are forward spreads); the calibration engine does the right thing off that spec
// alone. These are QuantLib-free machinery tests: hand-built annual OIS schedules, self-consistent
// market from a known x, recovered by the joint + staged solvers.
namespace {
// An annual OIS in the generic coupon model: each float coupon is one telescoped sub-period
// [prev, u] (tau cancels via k == 1), each fixed coupon accrues (u - prev). This is make_annual_ois's
// legacy OisSwap re-expressed as generic legs.
struct AnnualOis {
  std::vector<px::FloatCoupon> flt;
  std::vector<px::FixedCoupon> fix;
};
AnnualOis make_annual_ois(double T) {
  AnnualOis s;
  std::vector<double> t;
  for (double u = 1.0; u < T - 1e-9; u += 1.0) t.push_back(u);
  t.push_back(T);
  double prev = 0.0;
  for (double u : t) {
    px::FloatCoupon c;
    c.obs.sub_start = {prev};
    c.obs.sub_end = {u};
    c.obs.tau_index = u - prev;
    c.pay = u;
    c.tau_pay = u - prev;
    s.flt.push_back(c);
    s.fix.push_back({u, u - prev});
    prev = u;
  }
  return s;
}
// An annual OIS as a ParRate instrument (fwd leg forecasts `fc`, discounts `dc`).
cal::Instrument annual_par_rate(double T, int fc, int dc) {
  const AnnualOis s = make_annual_ois(T);
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParRate;
  ins.fwd = {s.flt, fc, dc};
  ins.fixed = {s.fix, dc};
  return ins;
}
// An annual OIS basis as a ParSpread instrument (spread leg forecasts `fwd_fc`, benchmark `bench_fc`).
cal::Instrument annual_basis(double T, int fwd_fc, int bench_fc, int dc) {
  const AnnualOis s = make_annual_ois(T);
  cal::Instrument ins;
  ins.quote = cal::QuoteKind::ParSpread;
  ins.fwd = {s.flt, fwd_fc, dc};
  ins.bench = {s.flt, bench_fc, dc};
  ins.fixed = {s.fix, dc};
  return ins;
}
}  // namespace

TEST(BundleSpread, JointAndStagedRecoverSpreadCurve) {
  // Curve 0 = outright base (pinned by its own OIS swaps); curve 1 = base + forward-spread (pinned by
  // basis swaps over curve 0). The spread block is coupled to the base block through the base discount.
  cal::BundleProblem prob;
  const std::vector<double> meeting{0.5}, back{1.0, 2.0, 3.0, 5.0, 10.0};
  prob.curves.resize(2);
  prob.curves[0] = {.base = -1, .regions = swaps::curve::flat_hermite(meeting, back)};  // outright
  prob.curves[1] = {.base = 0, .regions = swaps::curve::flat_hermite(meeting, back)};   // spread over curve 0
  const int nk = prob.curves[0].n_knots();
  const std::vector<double> mats{0.5, 1.0, 2.0, 3.0, 5.0, 10.0};
  for (double T : mats) prob.instruments.push_back(annual_par_rate(T, 0, 0));     // pin the base
  for (double T : mats) prob.instruments.push_back(annual_basis(T, 1, 0, 0));     // pin the spread

  Eigen::VectorXd x_true(2 * nk);
  for (int i = 0; i < nk; ++i) {
    x_true[i] = 0.040 + 0.001 * i;         // base forwards
    x_true[nk + i] = 0.0050 + 0.0003 * i;  // forward spreads
  }
  const Eigen::VectorXd r0 = prob.residuals<double>(x_true);  // make x_true the exact solution
  for (int i = 0; i < static_cast<int>(prob.instruments.size()); ++i) prob.instruments[i].market += r0[i];
  ASSERT_LT(prob.residuals<double>(x_true).cwiseAbs().maxCoeff(), 1e-14);

  Eigen::VectorXd x0(2 * nk);
  x0.head(nk).setConstant(0.04);
  x0.tail(nk).setConstant(0.005);
  const auto joint = cal::calibrate(prob, x0);
  const auto staged = cal::calibrate_staged(prob, x0);
  std::cout << "  [bundle-spread] joint ||x*-xtrue||=" << (joint.x - x_true).cwiseAbs().maxCoeff()
            << " iters=" << joint.iterations << " stat=" << joint.stationarity << "  staged ||x*-xtrue||="
            << (staged.x - x_true).cwiseAbs().maxCoeff() << " iters=" << staged.iterations << "\n";
  EXPECT_LT((joint.x - x_true).cwiseAbs().maxCoeff(), 1e-8) << "joint solve recovers base + spread";
  EXPECT_LT((staged.x - x_true).cwiseAbs().maxCoeff(), 1e-8) << "staged solve recovers base then spread";
}

TEST(BundleParallel, StarTopologyParallelIsBitIdenticalToSerial) {
  // Branch-parallel case (CLAUDE.md §7b): curve 0 is the outright base; curves 1..K are each a spread
  // straight off 0 and mutually independent (a STAR). The dependency waves are [{0}, {1..K}], so wave 1
  // is K independent block solves run concurrently. Because a spread block's residuals reference only its
  // own curve + the (already-solved) base, thread order cannot change any block's inputs -> the parallel
  // result must be BIT-IDENTICAL to the serial staged solve. That determinism IS the correctness gate.
  const int K = 5;
  cal::BundleProblem prob;
  const std::vector<double> meeting{0.5}, back{1.0, 2.0, 3.0, 5.0, 10.0};
  prob.curves.resize(K + 1);
  prob.curves[0] = {.base = -1, .regions = swaps::curve::flat_hermite(meeting, back)};  // outright base
  for (int c = 1; c <= K; ++c) prob.curves[c] = {.base = 0, .regions = swaps::curve::flat_hermite(meeting, back)};  // spread straight off the base
  const int nk = prob.curves[0].n_knots();
  const std::vector<double> mats{0.5, 1.0, 2.0, 3.0, 5.0, 10.0};
  for (double T : mats) prob.instruments.push_back(annual_par_rate(T, 0, 0));            // pin the base
  for (int c = 1; c <= K; ++c)
    for (double T : mats) prob.instruments.push_back(annual_basis(T, c, 0, 0));          // pin each spread

  Eigen::VectorXd x_true((K + 1) * nk);
  for (int i = 0; i < nk; ++i) x_true[i] = 0.040 + 0.001 * i;                            // base forwards
  for (int c = 1; c <= K; ++c)
    for (int i = 0; i < nk; ++i) x_true[c * nk + i] = 0.004 * c + 0.0003 * i;            // distinct spreads
  const Eigen::VectorXd r0 = prob.residuals<double>(x_true);
  for (int i = 0; i < static_cast<int>(prob.instruments.size()); ++i) prob.instruments[i].market += r0[i];
  ASSERT_LT(prob.residuals<double>(x_true).cwiseAbs().maxCoeff(), 1e-13);

  // Wave structure: a base wave of one SCC, then a fat wave of K independent SCCs.
  const auto sccs = cal::bundle_dependency_order(prob);
  const auto waves = cal::bundle_waves(prob, sccs);
  ASSERT_EQ(waves.size(), 2u) << "star topology => exactly two dependency waves";
  std::size_t fat = std::max(waves[0].size(), waves[1].size());
  EXPECT_EQ(fat, static_cast<std::size_t>(K)) << "the parallel wave must hold all K independent spreads";

  Eigen::VectorXd x0((K + 1) * nk);
  x0.head(nk).setConstant(0.04);
  for (int c = 1; c <= K; ++c) x0.segment(c * nk, nk).setConstant(0.004 * c);
  const auto serial = cal::calibrate_staged(prob, x0);
  const auto parallel = cal::calibrate_staged_parallel(prob, x0);
  swaps::parallel::ThreadPool pool(4);
  const auto pooled = cal::calibrate_staged_parallel(prob, x0, true, &pool);  // pool path == serial too
  EXPECT_EQ((pooled.x - serial.x).cwiseAbs().maxCoeff(), 0.0) << "thread-pool staged solve == serial";
  const double diff = (serial.x - parallel.x).cwiseAbs().maxCoeff();
  std::cout << "  [bundle-parallel] K=" << K << " |x_parallel - x_serial|=" << diff
            << " iters(serial=" << serial.iterations << ", parallel=" << parallel.iterations << ")"
            << " ||x*-xtrue||=" << (parallel.x - x_true).cwiseAbs().maxCoeff() << "\n";
  EXPECT_EQ(diff, 0.0) << "branch-parallel staged solve must be BIT-IDENTICAL to the serial staged solve";
  EXPECT_EQ(serial.iterations, parallel.iterations) << "same blocks, same iterations";
  EXPECT_LT((parallel.x - x_true).cwiseAbs().maxCoeff(), 1e-8) << "and it recovers the whole star";
}

TEST(BundleParallel, RealisticStarIsBitIdenticalToSerialAndRecovers) {
  // The canonical realistic model (reference_bundle.hpp) as a STAR: a full-structure SOFR base
  // (6 meetings + 12x1M/8x3M futures + swaps) with K basis curves each spread straight off SOFR
  // (12x1M futures + basis swaps). Waves = [{SOFR}, {K basis}], so the parallel solve runs K
  // realistically-sized block solves concurrently. It must be BIT-IDENTICAL to serial and recover x_true.
  using namespace QuantLib;
  RelinkableHandle<YieldTermStructure> hh;
  rb::Market m = rb::build_market(hh);
  rb::RealisticBundle b = rb::build_realistic_bundle(m, hh, /*n_basis=*/6, rb::BundleTopology::Star);
  ASSERT_LT(b.prob.residuals<double>(b.x_true).cwiseAbs().maxCoeff(), 1e-10) << "x_true zeroes the residual";

  const auto sccs = cal::bundle_dependency_order(b.prob);
  const auto waves = cal::bundle_waves(b.prob, sccs);
  ASSERT_EQ(waves.size(), 2u) << "star => a SOFR wave then a fat basis wave";
  EXPECT_EQ(std::max(waves[0].size(), waves[1].size()), 6u) << "6 independent basis SCCs in the parallel wave";

  const auto serial = cal::calibrate_staged(b.prob, b.x0);
  const auto parallel = cal::calibrate_staged_parallel(b.prob, b.x0);
  const double diff = (serial.x - parallel.x).cwiseAbs().maxCoeff();
  std::cout << "  [bundle-parallel-realistic] |x_par - x_ser|=" << diff
            << " ||x*-xtrue||=" << (parallel.x - b.x_true).cwiseAbs().maxCoeff()
            << " iters(ser=" << serial.iterations << ",par=" << parallel.iterations << ")\n";
  EXPECT_EQ(diff, 0.0) << "realistic branch-parallel solve must be BIT-IDENTICAL to serial";
  EXPECT_LT((parallel.x - b.x_true).cwiseAbs().maxCoeff(), 1e-6) << "and recover the realistic star";
}

TEST(BundleSpread, CompiledResidualHandlesSpreadCurves) {
  // The W-cache must handle a SPREAD curve too: DF_spread = exp(-(W_base x_base + W_spread x_spread)),
  // i.e. its W_all rows carry base-ancestry columns. Compiled residual + Jacobian must still match AAD.
  cal::BundleProblem prob;
  const std::vector<double> meeting{0.5}, back{1.0, 2.0, 3.0, 5.0, 10.0};
  prob.curves.resize(2);
  prob.curves[0] = {.base = -1, .regions = swaps::curve::flat_hermite(meeting, back)};  // outright base
  prob.curves[1] = {.base = 0, .regions = swaps::curve::flat_hermite(meeting, back)};   // spread over curve 0
  const int nk = prob.curves[0].n_knots();
  const std::vector<double> mats{0.5, 1.0, 2.0, 3.0, 5.0, 10.0};
  for (double T : mats) prob.instruments.push_back(annual_par_rate(T, 0, 0));
  for (double T : mats) prob.instruments.push_back(annual_basis(T, 1, 0, 0));

  Eigen::VectorXd x(2 * nk);
  for (int i = 0; i < nk; ++i) {
    x[i] = 0.040 + 0.001 * i;
    x[nk + i] = 0.0050 + 0.0003 * i;
  }
  cal::CompiledBundleResidual cr(prob);
  const double dr = (cr.residuals(x) - prob.residuals<double>(x)).cwiseAbs().maxCoeff();
  const double dj = (cr.jacobian(x) - cal::aad_jacobian(prob, x)).cwiseAbs().maxCoeff();
  std::cout << "  [bundle-spread] compiled |residual - AAD|=" << dr << " |Jacobian - AAD|=" << dj << "\n";
  EXPECT_LT(dr, 1e-14);
  EXPECT_LT(dj, 1e-9);
}

TEST(BundleSpread, SpreadHandleMatchesBasePlusSpread) {
  // The spread handle must be EXACTLY forward = base + spread, DF = base_DF * exp(-int spread).
  const std::vector<double> meeting{0.5}, back{1.0, 2.0, 3.0, 5.0, 10.0};
  std::vector<cal::BundleCurveSpec> specs{{.base = -1, .regions = swaps::curve::flat_hermite(meeting, back)}, {.base = 0, .regions = swaps::curve::flat_hermite(meeting, back)}};
  const int nk = specs[0].n_knots();
  Eigen::VectorXd base_f(nk), spread_f(nk);
  for (int i = 0; i < nk; ++i) {
    base_f[i] = 0.04 + 0.001 * i;
    spread_f[i] = 0.005 + 0.0003 * i;
  }
  const auto C = cal::build_bundle_curves<double>(
      specs, [&](int c, int i) { return c == 0 ? base_f[i] : spread_f[i]; });

  auto base_curve = cv::make_modular_curve<double>(cv::flat_hermite(meeting, back));
  base_curve.set_forwards(base_f);
  auto spread_curve = cv::make_modular_curve<double>(cv::flat_hermite(meeting, back));
  spread_curve.set_forwards(spread_f);
  double wf = 0.0, wd = 0.0;
  for (double t : {0.1, 0.5, 0.9, 1.5, 3.0, 7.0, 10.0}) {
    wf = std::max(wf, std::abs(C[1]->forward(t) - (base_curve.forward(t) + spread_curve.forward(t))));
    const double df = base_curve.discount(t) * std::exp(-spread_curve.integral(t));
    wd = std::max(wd, std::abs(C[1]->discount(t) - df));
  }
  std::cout << "  [bundle-spread] SpreadHandle max fwd err=" << wf << " df err=" << wd << "\n";
  EXPECT_LT(wf, 1e-14);
  EXPECT_LT(wd, 1e-14);
}
