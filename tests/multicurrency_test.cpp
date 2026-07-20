// Multi-currency EUR bundle (plan: composed-tickling-snowglobe, Phase 1). Validates the FULL EUR block
// -- ESTR OIS (compounded overnight), 3M/6M EURIBOR IRS (the previously-untested IBOR/par-coupon path),
// 3s6s tenor basis, a payment-delay OIS, and an EONIA default-spread curve -- FIRST-PRINCIPLES (self-
// consistent recovery + compiled-vs-AAD) AND against QuantLib core to 1e-10 (fairRate off our curves).
//
// Everything is a generic Instrument built via QuantLib MakeOIS / MakeVanillaSwap and the coupon-type-
// dispatch extractors; the engine names no currency (CLAUDE.md §1). The per-index default discount
// (EURIBOR -> ESTR) and its per-trade override are a builder-layer resolution rule -- no engine change.

#include <gtest/gtest.h>

#include <ql/quantlib.hpp>

#include <Eigen/Core>

#include <memory>
#include <vector>

#include "reference_multicurrency.hpp"
#include "swaps/calibration/bundle_stage.hpp"
#include "swaps/calibration/compiled_bundle.hpp"
#include "swaps/calibration/jacobian.hpp"
#include "swaps/calibration/lm.hpp"
#include "swaps/curve/calibration_curve.hpp"
#include "swaps/curve/ql_term_structure.hpp"
#include "swaps/pricing/cashflows.hpp"
#include "swaps/ql/extract.hpp"
#include "tolerances.hpp"

using namespace QuantLib;
namespace rb = swaps::refbuild;
namespace cal = swaps::calibration;
namespace px = swaps::pricing;
namespace qlx = swaps::qlx;

namespace {

// Our ParRate model quote for a freshly-built QuantLib swap, priced off the bundle's spread-aware
// CurveHandles at x_true (forecast on `fc`, discount on `dc`).
double our_par_rate(const rb::MultiCcyBundle& b, const Leg& floatLeg, const Leg& fixedLeg, int fc, int dc) {
  const auto fl = qlx::extract_float_leg(floatLeg, b.today, b.dc);
  const auto fx = qlx::extract_fixed_leg(fixedLeg, b.today, b.dc);
  return px::par_rate<double>(fl, fx, *b.curve_handles[fc], *b.curve_handles[dc]);
}

}  // namespace

// ---- Self-consistency: x_true zeroes the residual (the market was generated from it). ----
TEST(MultiCcyEur, MarketIsSelfConsistentAtXTrue) {
  const rb::MultiCcyBundle b = rb::build_eur_bundle();
  const double r = b.prob.residuals<double>(b.x_true).cwiseAbs().maxCoeff();
  std::cout << "  [mc-eur] curves=" << b.n_curves() << " knots=" << b.prob.n_knots()
            << " instruments=" << b.prob.n_residuals() << " ||r(x_true)||inf=" << r << "\n";
  EXPECT_LT(r, 1e-12) << "self-consistent EUR market must zero the residual at x_true";
}

// ---- Joint + staged solves recover x_true. ESTR is the base SCC; EUR3M then EUR6M stage after it. ----
TEST(MultiCcyEur, JointAndStagedRecoverAllThreeCurves) {
  const rb::MultiCcyBundle b = rb::build_eur_bundle();
  const auto joint = cal::calibrate(b.prob, b.x0);
  const auto staged = cal::calibrate_staged(b.prob, b.x0);
  const double dj = (joint.x - b.x_true).cwiseAbs().maxCoeff();
  const double ds = (staged.x - b.x_true).cwiseAbs().maxCoeff();
  std::cout << "  [mc-eur] joint ||x*-xtrue||=" << dj << " iters=" << joint.iterations
            << " stat=" << joint.stationarity << "  staged ||x*-xtrue||=" << ds
            << " iters=" << staged.iterations << "\n";
  EXPECT_LT(dj, 1e-6) << "joint solve recovers ESTR + EUR3M + EUR6M";
  EXPECT_LT(ds, 1e-6) << "staged solve recovers ESTR then EUR3M then EUR6M";
}

// ---- QuantLib-core oracle: our par rate == QuantLib fairRate off the SAME curves, to 1e-10, across
// ESTR OIS, 3M EURIBOR IRS and 6M EURIBOR IRS. This is the decisive proof the EURIBOR/IBOR path is
// correct (its first exercise anywhere) and that ESTR compounded overnight matches. ----
TEST(MultiCcyEur, ParRatesMatchQuantLibCore) {
  const rb::MultiCcyBundle b = rb::build_eur_bundle();
  double worst_ois = 0, worst_3m = 0, worst_6m = 0;

  for (int y : {1, 2, 3, 5, 7, 10, 15, 20, 30}) {
    auto o = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(y * Years, b.estr, 0.03).withDiscountingTermStructure(b.h[b.ESTR]));
    o->deepUpdate();
    const double ours = our_par_rate(b, o->overnightLeg(), o->fixedLeg(), b.ESTR, b.ESTR);
    worst_ois = std::max(worst_ois, std::abs(ours - o->fairRate()));
  }
  auto ibor = [&](const ext::shared_ptr<IborIndex>& idx, int fc, int y) {
    auto s = ext::shared_ptr<VanillaSwap>(MakeVanillaSwap(y * Years, idx, 0.03)
                                              .withDiscountingTermStructure(b.h[b.ESTR])
                                              .withFixedLegDayCount(Thirty360(Thirty360::BondBasis))
                                              .withFixedLegTenor(1 * Years)
                                              .withFixedLegConvention(ModifiedFollowing)
                                              .withFixedLegCalendar(TARGET())
                                              .withFloatingLegCalendar(TARGET()));
    s->deepUpdate();
    return std::abs(our_par_rate(b, s->floatingLeg(), s->fixedLeg(), fc, b.ESTR) - s->fairRate());
  };
  for (int y : {1, 2, 3, 5, 7, 10}) {
    worst_3m = std::max(worst_3m, ibor(b.eur3m, b.EUR3M, y));
    worst_6m = std::max(worst_6m, ibor(b.eur6m, b.EUR6M, y));
  }
  std::cout << "  [mc-eur] max |ours - QuantLib fairRate|:  ESTR OIS=" << worst_ois
            << "  3M EURIBOR=" << worst_3m << "  6M EURIBOR=" << worst_6m << "\n";
  EXPECT_LT(worst_ois, swaps::tol::curve_rel) << "ESTR OIS par rate must match QuantLib";
  EXPECT_LT(worst_3m, swaps::tol::curve_rel) << "3M EURIBOR IRS par rate must match QuantLib (IBOR path)";
  EXPECT_LT(worst_6m, swaps::tol::curve_rel) << "6M EURIBOR IRS par rate must match QuantLib (IBOR path)";
}

// ---- The multi-curve W-cache residual + its analytic block Jacobian must match the templated /
// AAD paths -- so the EURIBOR and 3s6s instruments ride the compiled fast path correctly. ----
TEST(MultiCcyEur, CompiledBundleResidualMatchesAad) {
  const rb::MultiCcyBundle b = rb::build_eur_bundle();
  cal::CompiledBundleResidual cr(b.prob);
  const double dr = (cr.residuals(b.x_true) - b.prob.residuals<double>(b.x_true)).cwiseAbs().maxCoeff();
  const double dj = (cr.jacobian(b.x_true) - cal::aad_jacobian(b.prob, b.x_true)).cwiseAbs().maxCoeff();
  std::cout << "  [mc-eur] compiled |residual - templated|=" << dr << " |Jacobian - AAD|=" << dj << "\n";
  EXPECT_LT(dr, 1e-13) << "compiled multi-curve residual matches the templated kernel";
  EXPECT_LT(dj, 1e-9) << "analytic block Jacobian matches AAD";
}

// ---- Payment delay: an ESTR OIS with a 2-business-day payment lag has its pay date AFTER the accrual
// end, that lag is captured in the extracted coupon's `pay`, and it still prices to QuantLib to 1e-10. ----
TEST(MultiCcyEur, PaymentDelayIsCapturedAndPriced) {
  const rb::MultiCcyBundle b = rb::build_eur_bundle();
  auto lag = ext::shared_ptr<OvernightIndexedSwap>(
      MakeOIS(2 * Years, b.estr, 0.03).withDiscountingTermStructure(b.h[b.ESTR]).withPaymentLag(2));
  auto nolag = ext::shared_ptr<OvernightIndexedSwap>(
      MakeOIS(2 * Years, b.estr, 0.03).withDiscountingTermStructure(b.h[b.ESTR]));
  lag->deepUpdate();
  nolag->deepUpdate();

  // The lagged coupon's pay time must sit strictly after its accrual-end (last value date), and after
  // the no-lag coupon's pay time.
  const auto lc = qlx::extract_float_leg(lag->overnightLeg(), b.today, b.dc);
  const auto nc = qlx::extract_float_leg(nolag->overnightLeg(), b.today, b.dc);
  ASSERT_FALSE(lc.empty());
  const px::FloatCoupon& last_lag = lc.back();
  const px::FloatCoupon& last_nolag = nc.back();
  EXPECT_GT(last_lag.pay, last_lag.obs.sub_end.back())
      << "a payment delay must put the pay date after the accrual end (last value date)";
  EXPECT_GT(last_lag.pay, last_nolag.pay) << "the lagged pay date must be later than the un-lagged one";

  // And it still prices to QuantLib exactly (DF(pay) discounts the lagged date).
  const double ours = our_par_rate(b, lag->overnightLeg(), lag->fixedLeg(), b.ESTR, b.ESTR);
  std::cout << "  [mc-eur] payment-lag pay-time delta=" << (last_lag.pay - last_nolag.pay)
            << "  |ours - QuantLib|=" << std::abs(ours - lag->fairRate()) << "\n";
  EXPECT_LT(std::abs(ours - lag->fairRate()), swaps::tol::curve_rel)
      << "a payment-delayed OIS must still match QuantLib";
}

// ---- EONIA as a DEFAULT fixed spread over ESTR (ESTR + 8.5bp), the "default the spread, overridable"
// mechanism: build it as a SpreadHandle with a flat 8.5bp spread and confirm forward == ESTR + 8.5bp
// exactly (and DF = ESTR_DF * exp(-int spread)). No calibration instruments => a fixed base spread. ----
TEST(MultiCcyEur, EoniaDefaultSpreadOverEstr) {
  const rb::MultiCcyBundle b = rb::build_eur_bundle();
  constexpr double kEoniaSpread = 0.00085;  // EONIA = ESTR + 8.5bp (legacy transition spread)

  // ESTR curve at x_true, then EONIA = ESTR + flat 8.5bp forward spread.
  const auto& estr = *b.curve_handles[b.ESTR];
  const std::vector<double> meet{0.5}, back{1, 2, 3, 5, 7, 10, 15, 20, 30};
  auto sp = swaps::curve::make_calibration_curve<double>(meet, back);
  Eigen::VectorXd s(static_cast<int>(meet.size() + back.size()));
  s.setConstant(kEoniaSpread);
  sp.set_forwards(s);
  cal::SpreadHandle<double> eonia(std::move(sp), &estr);

  double worst_fwd = 0, worst_df = 0;
  for (double T : {0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0}) {
    worst_fwd = std::max(worst_fwd, std::abs(eonia.forward(T) - (estr.forward(T) + kEoniaSpread)));
    worst_df = std::max(worst_df, std::abs(eonia.discount(T) - estr.discount(T) * std::exp(-kEoniaSpread * T)));
  }
  std::cout << "  [mc-eur] EONIA=ESTR+8.5bp  max|fwd-(ESTR+spr)|=" << worst_fwd
            << "  max|DF-ESTR*exp(-spr t)|=" << worst_df << "\n";
  EXPECT_LT(worst_fwd, 1e-14) << "EONIA forward must be ESTR + the default 8.5bp spread exactly";
  EXPECT_LT(worst_df, 1e-14) << "EONIA discount must be ESTR_DF * exp(-int spread) exactly";
}

// ===============================================================================================
// USD block (Phase 2): SOFR + Fed Funds (compounded FF-OIS basis + arithmetic FF averaging future).
// ===============================================================================================

TEST(MultiCcyUsd, MarketIsSelfConsistentAtXTrue) {
  const rb::MultiCcyBundle b = rb::build_usd_bundle();
  const double r = b.prob.residuals<double>(b.x_true).cwiseAbs().maxCoeff();
  std::cout << "  [mc-usd] curves=" << b.n_curves() << " knots=" << b.prob.n_knots()
            << " instruments=" << b.prob.n_residuals() << " ||r(x_true)||inf=" << r << "\n";
  EXPECT_LT(r, 1e-10) << "self-consistent USD market must zero the residual at x_true";
}

TEST(MultiCcyUsd, JointAndStagedRecoverSofrAndFf) {
  const rb::MultiCcyBundle b = rb::build_usd_bundle();
  const auto joint = cal::calibrate(b.prob, b.x0);
  const auto staged = cal::calibrate_staged(b.prob, b.x0);
  const double dj = (joint.x - b.x_true).cwiseAbs().maxCoeff();
  const double ds = (staged.x - b.x_true).cwiseAbs().maxCoeff();
  std::cout << "  [mc-usd] joint ||x*-xtrue||=" << dj << " iters=" << joint.iterations
            << "  staged ||x*-xtrue||=" << ds << " iters=" << staged.iterations << "\n";
  EXPECT_LT(dj, 1e-6) << "joint solve recovers SOFR + FF";
  EXPECT_LT(ds, 1e-6) << "staged solve recovers SOFR then the FF spread";
}

// Fed Funds OIS (COMPOUNDED, like SOFR OIS) and the FF/SOFR basis must match QuantLib to 1e-10 off our
// curves -- the proof FF-OIS compounds correctly on the FedFunds (FederalReserve) calendar.
TEST(MultiCcyUsd, FedFundsOisAndBasisMatchQuantLib) {
  const rb::MultiCcyBundle b = rb::build_usd_bundle();
  double worst_ois = 0, worst_basis = 0;
  for (int y : {1, 2, 3, 5, 10, 30}) {
    auto ffo = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(y * Years, b.fedfunds, 0.03).withDiscountingTermStructure(b.h[b.SOFR]));
    auto so = ext::shared_ptr<OvernightIndexedSwap>(
        MakeOIS(y * Years, b.sofr, 0.03).withDiscountingTermStructure(b.h[b.SOFR]));
    ffo->deepUpdate();
    so->deepUpdate();
    // FF-OIS fair rate (FF forecast, SOFR-discounted).
    worst_ois = std::max(worst_ois,
                         std::abs(our_par_rate(b, ffo->overnightLeg(), ffo->fixedLeg(), b.FF, b.SOFR) -
                                  ffo->fairRate()));
    // FF/SOFR basis = r_SOFR - r_FF; our par_spread(fwd=FF, bench=SOFR, disc=SOFR).
    const auto ff_fl = qlx::extract_float_leg(ffo->overnightLeg(), b.today, b.dc);
    const auto so_fl = qlx::extract_float_leg(so->overnightLeg(), b.today, b.dc);
    const auto so_fx = qlx::extract_fixed_leg(so->fixedLeg(), b.today, b.dc);
    const double ours = px::par_spread<double>(ff_fl, so_fl, so_fx, *b.curve_handles[b.FF],
                                               *b.curve_handles[b.SOFR], *b.curve_handles[b.SOFR]);
    worst_basis = std::max(worst_basis, std::abs(ours - (so->fairRate() - ffo->fairRate())));
  }
  std::cout << "  [mc-usd] max |ours - QuantLib|: FF-OIS=" << worst_ois << "  FF/SOFR basis=" << worst_basis << "\n";
  EXPECT_LT(worst_ois, swaps::tol::curve_rel) << "compounded FedFunds OIS must match QuantLib";
  EXPECT_LT(worst_basis, swaps::tol::curve_rel) << "FF/SOFR basis must match QuantLib";
}

// The Fed Funds 1M ARITHMETIC averaging future must match QuantLib's OvernightIndexFuture
// (RateAveraging::Simple) to machine precision -- the proof FF futures average the daily EFFR correctly.
TEST(MultiCcyUsd, FedFundsAveragingFutureMatchesQuantLib) {
  const rb::MultiCcyBundle b = rb::build_usd_bundle();
  const Calendar fcal = b.fedfunds->fixingCalendar();
  double worst = 0;
  // Fully-forward monthly contracts (eval is 8 Jul 2026).
  for (const auto& mo : {std::make_pair(September, 2026), std::make_pair(October, 2026),
                         std::make_pair(November, 2026)}) {
    const Date s = fcal.adjust(Date(1, mo.first, mo.second));
    const Date e = fcal.adjust(Date(1, Month(int(mo.first) % 12 + 1), mo.second + (mo.first == December)));
    OvernightIndexFuture fut(b.fedfunds, s, e, Handle<Quote>(), RateAveraging::Simple);
    const double ql = 1.0 - fut.NPV() / 100.0;  // NPV = 100·(1 − (convexity + rate)); convexity = 0
    const px::RateObservation obs = rb::avg_future_obs_idx(b.fedfunds, b.today, b.dc, s, e);
    const double ours = px::rate<double>(obs, *b.curve_handles[b.FF]);
    worst = std::max(worst, std::abs(ours - ql));
  }
  std::cout << "  [mc-usd] FF 1M averaging future max |ours - QuantLib averagedRate| = " << worst << "\n";
  EXPECT_LT(worst, swaps::tol::curve_rel) << "FF arithmetic averaging future must match QuantLib";
}

TEST(MultiCcyUsd, CompiledBundleResidualMatchesAad) {
  const rb::MultiCcyBundle b = rb::build_usd_bundle();
  cal::CompiledBundleResidual cr(b.prob);
  const double dr = (cr.residuals(b.x_true) - b.prob.residuals<double>(b.x_true)).cwiseAbs().maxCoeff();
  const double dj = (cr.jacobian(b.x_true) - cal::aad_jacobian(b.prob, b.x_true)).cwiseAbs().maxCoeff();
  std::cout << "  [mc-usd] compiled |residual - templated|=" << dr << " |Jacobian - AAD|=" << dj << "\n";
  EXPECT_LT(dr, 1e-12) << "compiled USD residual matches the templated kernel";
  EXPECT_LT(dj, 1e-8) << "analytic block Jacobian matches AAD";
}

// PRIME as a DEFAULT fixed spread over Fed Funds (FF + ~300bp), overridable -- same mechanism as EONIA.
TEST(MultiCcyUsd, PrimeDefaultSpreadOverFedFunds) {
  const rb::MultiCcyBundle b = rb::build_usd_bundle();
  constexpr double kPrimeSpread = 0.0300;  // PRIME ~= Fed Funds + 300bp
  const auto& ff = *b.curve_handles[b.FF];
  const std::vector<double> meet{0.5}, back{1, 2, 3, 5, 7, 10, 15, 20, 30};
  auto sp = swaps::curve::make_calibration_curve<double>(meet, back);
  Eigen::VectorXd s(static_cast<int>(meet.size() + back.size()));
  s.setConstant(kPrimeSpread);
  sp.set_forwards(s);
  cal::SpreadHandle<double> prime(std::move(sp), &ff);

  double worst_fwd = 0;
  for (double T : {0.25, 1.0, 5.0, 10.0, 30.0})
    worst_fwd = std::max(worst_fwd, std::abs(prime.forward(T) - (ff.forward(T) + kPrimeSpread)));
  std::cout << "  [mc-usd] PRIME=FF+300bp  max|fwd-(FF+spr)|=" << worst_fwd << "\n";
  EXPECT_LT(worst_fwd, 1e-14) << "PRIME forward must be Fed Funds + the default 300bp spread exactly";
}

// The multi-currency BOOK: USD (SOFR+FF) and EUR (ESTR+EURIBOR) are INDEPENDENT problems -- the across-
// problem parallelism CLAUDE.md §7b calls out. Once built (their instruments are pure data), each
// calibrates with no shared state and recovers its own x_true, so a desk can solve them concurrently.
TEST(MultiCcy, UsdAndEurBooksCalibrateIndependently) {
  const rb::MultiCcyBundle usd = rb::build_usd_bundle();   // sets eval to the SOFR reference date
  const rb::MultiCcyBundle eur = rb::build_eur_bundle();   // sets eval to its own date; USD data already extracted
  const auto u = cal::calibrate(usd.prob, usd.x0);
  const auto e = cal::calibrate(eur.prob, eur.x0);
  const double du = (u.x - usd.x_true).cwiseAbs().maxCoeff();
  const double de = (e.x - eur.x_true).cwiseAbs().maxCoeff();
  std::cout << "  [mc-book] USD ||x*-xtrue||=" << du << "  EUR ||x*-xtrue||=" << de << "\n";
  EXPECT_LT(du, 1e-6) << "the USD book recovers independently";
  EXPECT_LT(de, 1e-6) << "the EUR book recovers independently";
}
