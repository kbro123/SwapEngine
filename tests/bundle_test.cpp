// Stage 3 gate on REALISTIC curve builds: four curves, each flat-forward meeting-date front + par-swap
// back (the production SOFR structure), calibrated simultaneously and validated against QuantLib.
//   SOFR   : real reference market -- 12x1M + 8x3M SOFR futures + 9 par swaps (build_market/build_problem)
//   FF     : 12x1M FF futures (front) + FF-SOFR basis at 18m/2y/3y + swap maturities (replaces 3M futures)
//   PRIME  : basis over FF at 3m/6m/9m/12m + 18m/2y/3y + swap maturities (basis only, no futures)
//   PRIME2 : basis over PRIME, same tenors
// Quotes are generated from a known set of forwards, so the joint solve must recover them. The DUAL-CURVE
// basis pricing (the new Stage-3 kernel) is additionally oracle-checked against QuantLib's own multi-curve
// OIS fair spread. SOFR/FF futures reuse the single-curve kernel already validated to 1e-16 (pricing_test).

#include <gtest/gtest.h>
#include <ql/quantlib.hpp>

#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "reference_curve.hpp"
#include "swaps/calibration/bundle_problem.hpp"
#include "swaps/calibration/bundle_stage.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/calibration_curve.hpp"
#include "swaps/curve/ql_term_structure.hpp"
#include "swaps/ql/extract.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace cv = swaps::curve;
namespace cal = swaps::calibration;
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
  std::vector<cv::CalibrationCurve<double>> curves;
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

    prob.curves.resize(NC);
    prob.curves[SOFR] = {sofr.meeting_times, sofr.back_times};
    prob.curves[FF] = {sofr.meeting_times, back_t};  // FOMC meeting-date front (same as SOFR)
    prob.curves[PRIME] = {mon_t, back_t};
    prob.curves[PRIME2] = {mon_t, back_t};
    off.assign(NC, 0);
    for (int c = 1; c < NC; ++c) off[c] = off[c - 1] + prob.curves[c - 1].n_knots();
    const int N = off[NC - 1] + prob.curves[NC - 1].n_knots();

    // Known forwards: SOFR ~4.3%, FF -6bp, PRIME +300bp, PRIME2 +50bp (gentle upward slope each).
    const double base[] = {0.0430, 0.0424, 0.0730, 0.0780};
    x_true.resize(N);
    for (int c = 0; c < NC; ++c) {
      const int nk = prob.curves[c].n_knots();
      for (int i = 0; i < nk; ++i) x_true[off[c] + i] = base[c] + 0.0003 * i;
    }

    // ---- SOFR instruments (curve 0): reuse the reference futures + swaps ----
    for (const auto& a : sofr.avg_futs) prob.avg_futs.push_back({SOFR, a.sched, 0.0, 0.0});
    for (const auto& cf : sofr.comp_futs) prob.comp_futs.push_back({SOFR, cf.sched, 0.0, 0.0});
    for (const auto& s : sofr.swaps) prob.swaps.push_back({SOFR, SOFR, s.sched, 0.0});

    // ---- FF/PRIME/PRIME2 indices ----
    const char* names[] = {"", "FFx", "PRIMEx", "PRIME2x"};
    h.resize(NC);
    idx.resize(NC);
    h[SOFR] = hS;
    for (int c = 1; c < NC; ++c) {
      h[c].linkTo(ext::make_shared<FlatForward>(today, 0.04, dc, Continuous));
      idx[c] = ext::make_shared<OvernightIndex>(names[c], 0, USDCurrency(), cal, Actual360(), h[c]);
    }

    // FF front: 12 1M FF averaging futures (schedule from the SOFR 1M-future dates, forecast off FF).
    for (int i = 0; i < 12; ++i) {
      const auto& q = rm::futures_1m[i];
      const Date s = rb::sofr_start(Month(q.ref_month), q.ref_year, Monthly),
                 e = rb::sofr_end(Month(q.ref_month), q.ref_year, Monthly);
      prob.avg_futs.push_back({FF, swaps::qlx::extract_averaged_future(mk.sofr, s, e, today, dc), 0.0, 0.0});
    }
    // FF back: FF-SOFR basis at the back-tenor knots.
    for (const Period& p : back_periods) {
      auto o = ext::shared_ptr<OvernightIndexedSwap>(MakeOIS(p, idx[FF], 0.03).withDiscountingTermStructure(hS));
      prob.bases.push_back({FF, SOFR, SOFR, swaps::qlx::extract_ois_swap(*o, today, dc), 0.0});
    }
    // PRIME / PRIME2: basis over the previous curve at the monthly front + back tenors.
    std::vector<Period> all = mon_periods;
    all.insert(all.end(), back_periods.begin(), back_periods.end());
    for (int c = PRIME; c <= PRIME2; ++c)
      for (const Period& p : all) {
        auto o = ext::shared_ptr<OvernightIndexedSwap>(MakeOIS(p, idx[c], 0.03).withDiscountingTermStructure(hS));
        prob.bases.push_back({c, c - 1, SOFR, swaps::qlx::extract_ois_swap(*o, today, dc), 0.0});
      }

    // ---- Curves at x_true, exposed to QuantLib as YieldTermStructures ----
    curves.reserve(NC);
    for (int c = 0; c < NC; ++c) curves.push_back(cv::make_calibration_curve<double>(prob.curves[c].meeting, prob.curves[c].back));
    for (int c = 0; c < NC; ++c) {
      Eigen::VectorXd xi = x_true.segment(off[c], prob.curves[c].n_knots());
      curves[c].set_forwards(xi);
    }
    ts.resize(NC);
    for (int c = 0; c < NC; ++c) {
      auto t = ext::make_shared<swaps::qlx::CurveTermStructure<cv::CalibrationCurve<double>>>(today, dc, &curves[c]);
      t->enableExtrapolation();
      ts[c] = t;
      h[c].linkTo(t);
    }
    for (auto& s : mk.swaps) s->deepUpdate();

    // Market quotes generated from x_true (self-consistent), so the joint solve recovers x_true.
    const Eigen::VectorXd r0 = prob.residuals<double>(x_true);
    int i = 0;
    for (auto& a : prob.avg_futs) a.market_rate += r0[i++];
    for (auto& c : prob.comp_futs) c.market_rate += r0[i++];
    for (auto& s : prob.swaps) s.market_rate += r0[i++];
    for (auto& b : prob.bases) b.market_rate += r0[i++];
  }
};

TEST_F(BundleRealistic, JointAndStagedRecoverAllFourCurves) {
  std::cout << "  [bundle-real] knots per curve: SOFR=" << prob.curves[SOFR].n_knots()
            << " FF=" << prob.curves[FF].n_knots() << " PRIME=" << prob.curves[PRIME].n_knots()
            << " PRIME2=" << prob.curves[PRIME2].n_knots() << "  total knots=" << prob.n_knots()
            << " instruments=" << prob.n_residuals() << "\n";
  EXPECT_LT(prob.residuals<double>(x_true).cwiseAbs().maxCoeff(), 1e-12);

  Eigen::VectorXd x0(prob.n_knots());  // per-curve flat start near each curve's level
  const double start[] = {0.043, 0.043, 0.073, 0.078};
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

TEST_F(BundleRealistic, DualCurveBasisMatchesQuantLib) {
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
    const auto sched = swaps::qlx::extract_ois_swap(*ff, today, dc);
    const double ours =
        swaps::pricing::basis_par_spread<double>(sched, curves[FF], curves[SOFR], curves[SOFR]);
    worst = std::max(worst, std::abs(ours - ql));
    ++n;
  }
  std::cout << "  [bundle-real] FF-SOFR basis max |ours - QuantLib| = " << worst << " over " << n << " tenors\n";
  EXPECT_LT(worst, swaps::tol::curve_rel) << "dual-curve basis pricing must match QuantLib";
}
